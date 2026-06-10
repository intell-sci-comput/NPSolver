import time
import os
import os.path as osp
import logging
import swanlab
import torch
import numpy as np
from tqdm import tqdm

from .utils import AverageMeter, RMSELoss, RelativeL2Loss


class NPSolver3DTrainer:
    """Field Trainer."""
    def __init__(self, model, device, cfg, optimizer=None, scheduler=None, phy_loss_func=None,
                 data_loss_func=None):
        """Initialize the trainer.
        Args:
            model (Model): Model for field optimization
            optimizer (Optimizer): Optimizer for trainable fields
            scheduler (Scheduler): Scheduler for optimizer
            loss_func (callable): Loss function to compute the loss
            device (torch.device): Device to run the training
            cfg (Config): Configuration object
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.phy_loss_func = phy_loss_func
        self.data_loss_func = data_loss_func
        self.device = device
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self, tr_loader):
        """Train the field.
        """
        if self.cfg.project.continous_train:
            self.load_checkpoint()
        start_epoch = self.cfg.params.start_epoch
        end_epoch = self.cfg.params.start_epoch + self.cfg.params.num_epochs
        min_tr_loss = 1.0e+6
        for epoch in range(start_epoch, end_epoch):
            phy_loss = self._train_loop(tr_loader, epoch)
            info_str = f'[Epoch {epoch:4d}/{end_epoch-1:4d}] tr_loss: {phy_loss:.2e}'
            if epoch == start_epoch or epoch % self.cfg.params.save_freq == 0:
                self.save_checkpoint()
            if epoch == start_epoch or epoch % self.cfg.params.print_freq == 0:
                if phy_loss < min_tr_loss:
                    min_tr_loss = phy_loss
                    self.save_checkpoint(min=True)
                    info_str += ' [MIN]'
                self.logger.info(info_str)

    def _train_loop(self, tr_loader, epoch):
        self.model.train()
        tr_loss = 0
        step_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        data_start_time = time.time()
        step_start_time = data_start_time
        batch_per_epoch = len(tr_loader)
        iters_per_epoch = batch_per_epoch // self.cfg.params.grad_accum

        pbar = tqdm(total=iters_per_epoch)
        for b_i, batch in enumerate(tr_loader):
            x, f, mask, x_bd, mask_bd, mesh = batch
            x, f = x.to(self.device), f.to(self.device)
            x_bd, mask_bd = x_bd.to(self.device), mask_bd.to(self.device)
            mask, mesh = mask.to(self.device), mesh.to(self.device)
            data_time_meter.update(time.time() - data_start_time)
            p_predicted = self.model(x, f, x_bd, mask, mask_bd)  # (b, n_max, 1)
            p_predicted = p_predicted[mask]  # (n_total, 1)
            f = f[mask]  # (n_total, 1)
            loss = self.phy_loss_func(p_outputs=p_predicted, f_samples=f, mesh=mesh)
            accum_loss = loss / self.cfg.params.grad_accum
            accum_loss.backward()
            if (b_i + 1) % self.cfg.params.grad_accum == 0:
                before_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.cfg.params.grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                
                step_time_meter.update(time.time() - step_start_time)
                iters = (epoch - 1) * iters_per_epoch + pbar.n + 1
                if iters == 1 or iters % self.cfg.params.batch_print_freq == 0:
                    swanlab.log({'phy_loss': loss.detach().item(),
                                'lr': self.optimizer.param_groups[0]['lr']},
                                step=iters)
                    pbar.set_postfix(tr_loss=loss.detach().item(),
                                     data_time=f'{data_time_meter.val:.2f} ({data_time_meter.avg:.2f})',
                                     step_time=f'{step_time_meter.val:.2f} ({step_time_meter.avg:.2f})',
                                     grad_norm=f'{before_grad_norm:.2f}')
                pbar.update(1)
                step_start_time = time.time()
            tr_loss += loss.item()
            data_start_time = time.time()
        pbar.close()
        return tr_loss / len(tr_loader)

    @torch.no_grad()
    def test(self, dataset):
        self.load_checkpoint(min=True)
        self.model.eval()
        length = len(dataset)
        rmse_func = RMSELoss()
        rel_l2_func = RelativeL2Loss()
        x_dict, p_pred_dict, f_dict, y_dict,  r_dict, time_dict = {}, {}, {}, {}, {}, {}
        name_dict = {}
        rmse_dict, rel_l2_dict = {}, {}
        for i in tqdm(range(length)):
            x, f, x_bd, mesh, y, category = dataset[i]
            x = x.unsqueeze(dim=0)
            f = f.unsqueeze(dim=0)
            x_bd = x_bd.unsqueeze(dim=0)
            x, f, x_bd = x.to(self.device), f.to(self.device), x_bd.to(self.device)
            mesh, y = mesh.to(self.device), y.to(self.device)
            start = time.time()
            p_pred = self.model(x, f, x_bd).squeeze(dim=0)  # (num_cells, 1)
            end = time.time()
            f = f.squeeze(dim=0)
            r = self.phy_loss_func(p_outputs=p_pred, f_samples=f, mesh=mesh)
            rmse = rmse_func(p_pred, y)
            rel_l2 = rel_l2_func(p_pred, y)
            if category not in p_pred_dict.keys():
                x_dict[category] = []
                p_pred_dict[category] = []
                f_dict[category] = []
                y_dict[category] = []
                r_dict[category] = []
                time_dict[category] = []
                rmse_dict[category] = []
                rel_l2_dict[category] = []
                name_dict[category] = []
            x_dict[category].append(x.squeeze(dim=0).cpu().numpy())
            p_pred_dict[category].append(p_pred.cpu().numpy())
            f_dict[category].append(f.cpu().numpy())
            y_dict[category].append(y.cpu().numpy())
            r_dict[category].append(r.item())
            time_dict[category].append(end - start)
            rmse_dict[category].append(rmse.item())
            rel_l2_dict[category].append(rel_l2.item())
            name_dict[category].append(mesh.case_name)
        
        mean_r_dict, mean_time_dict, mean_rmse_dict, mean_rel_l2_dict = {}, {}, {}, {}
        for category in p_pred_dict.keys():
            mean_r_dict[category] = np.mean(r_dict[category])
            mean_time_dict[category] = np.mean(time_dict[category])
            mean_rmse_dict[category] = np.mean(rmse_dict[category])
            mean_rel_l2_dict[category] = np.mean(rel_l2_dict[category])
        return x_dict, f_dict, p_pred_dict, y_dict, name_dict, mean_r_dict, mean_time_dict, mean_rmse_dict, mean_rel_l2_dict

    def save_checkpoint(self, min=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        if min:
            ckpt_path = osp.join(self.cfg.output.path, 'ckpt_min')
        else:
            ckpt_path = osp.join(self.cfg.output.path, 'ckpt')
        if not osp.exists(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(checkpoint, osp.join(ckpt_path, 'checkpoint.pth'))
    
    def load_checkpoint(self, min=False):
        if min:
            ckpt_path = osp.join(self.cfg.output.path, 'ckpt_min')
        else:
            ckpt_path = osp.join(self.cfg.output.path, 'ckpt')
        checkpoint = torch.load(osp.join(ckpt_path, 'checkpoint.pth'), map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])