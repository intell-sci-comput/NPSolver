import logging
import os.path as osp
import swanlab
import hydra
import h5py
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from src.datasets.dataset_3d.dataset import Train3DDataset, \
    my_collate_fn, Test3DDataset
from src.trainers.npsolver_3d_trainer import NPSolver3DTrainer
from src.trainers.utils import RMSELoss
from src.models.ba_transolver.Transolver_Irregular_Mesh import Model as BATransolver
from src.models.ba_transolver.loss import FVMBatchLoss


def train(cfg, logger):
    swanlab.init(
        project=cfg.project.name,
        experiment_name=cfg.output.name,
        config=cfg,
        logdir=cfg.output.path
    )
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info('================= Train ==================')

    # set seed
    random.seed(cfg.project.seed)
    np.random.seed(cfg.project.seed)
    torch.manual_seed(cfg.project.seed)
    torch.cuda.manual_seed(cfg.project.seed)
    # load data
    logger.info('Load data...')
    # dataset
    dataset = Train3DDataset(
        num_samples=cfg.params.num_batches_per_epoch * cfg.params.batch_size * cfg.params.grad_accum,
        mesh_file=osp.join(cfg.data.mesh_dir, cfg.data.file_name),
        norm_f=cfg.data.norm_f
    )
    collate_fn = my_collate_fn
    tr_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    # model
    transolver = BATransolver(
        space_dim=cfg.model.space_dim,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_head,
        Time_Input=False,
        mlp_ratio=cfg.model.mlp_ratio,
        fun_dim=cfg.model.fun_dim,
        bdry_dim=cfg.model.bdry_dim,
        out_dim=cfg.model.out_dim,
        slice_num=cfg.model.slice_num,
        ref=cfg.model.ref,
        unified_pos=cfg.model.unified_pos
    ).to(cfg.project.device)
    logger.info(f'Number of parameters: {sum(p.numel() for p in transolver.parameters())}')
    optimizer = torch.optim.Adam(transolver.parameters(), lr=cfg.params.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.params.lr, epochs=cfg.params.num_epochs,
        steps_per_epoch=cfg.params.num_batches_per_epoch, pct_start=cfg.params.pct_start
    )
    phy_loss_func = FVMBatchLoss(
        max_iter=cfg.model.max_iter,
        scheme=cfg.model.scheme,
        omega=cfg.model.omega
    )
    # trainer
    trainer = NPSolver3DTrainer(
        model=transolver,
        device=torch.device(cfg.project.device),
        cfg=cfg,
        optimizer=optimizer,
        scheduler=scheduler,
        phy_loss_func=phy_loss_func,
        data_loss_func=RMSELoss(),
    )
    trainer.train(
        tr_loader=tr_loader
    )

def test(cfg, logger):
    logger.info('================= Test ==================')
    cfg.model.scheme = 5
    cfg.model.max_iter = 0
    cfg.params.batch_size = 1
    
    # load data
    logger.info('Load data...')
    dataset = Test3DDataset(
        data_dir=cfg.data.sol_dir,
        file_names=cfg.data.test_files,
        apply_bc=cfg.data.apply_bc
    )
    # model
    transolver = BATransolver(
        space_dim=cfg.model.space_dim,
        n_layers=cfg.model.n_layers,
        n_hidden=cfg.model.n_hidden,
        dropout=cfg.model.dropout,
        n_head=cfg.model.n_head,
        Time_Input=False,
        mlp_ratio=cfg.model.mlp_ratio,
        fun_dim=cfg.model.fun_dim,
        bdry_dim=cfg.model.bdry_dim,
        out_dim=cfg.model.out_dim,
        slice_num=cfg.model.slice_num,
        ref=cfg.model.ref,
        unified_pos=cfg.model.unified_pos
    ).to(cfg.project.device)
    logger.info(f'Number of parameters: {sum(p.numel() for p in transolver.parameters())}')
    # trainer
    phy_loss_func = FVMBatchLoss(
        max_iter=cfg.model.max_iter,
        scheme=cfg.model.scheme,
        omega=cfg.model.omega
    )
    trainer = NPSolver3DTrainer(
        model=transolver,
        device=torch.device(cfg.project.device),
        cfg=cfg,
        optimizer=None,
        scheduler=None,
        phy_loss_func=phy_loss_func,
        data_loss_func=None,
    )
    # test
    x_dict, f_dict, p_preds_dict, y_dict, name_dict, r_dict, time_dict, rmse_dict, rel_l2_dict = trainer.test(dataset)  # dict of list
    rmse_list, rel_l2_list, r_list, time_list = [], [], [], []
    for category in p_preds_dict.keys():
        logger.info(f'{category} - rmse {rmse_dict[category]: .2e}, '
                    f'rel l2 {rel_l2_dict[category]:.3e}, r {r_dict[category]:.2e} '
                    f'time {time_dict[category]:.4f} s')
        rmse_list.append(rmse_dict[category])
        rel_l2_list.append(rel_l2_dict[category])
        r_list.append(r_dict[category])
        time_list.append(time_dict[category])
    logger.info(f'Overall - rmse {np.mean(rmse_list):.2e} '
                f'rel l2 {np.mean(rel_l2_list):.2e} '
                f'r {np.mean(r_list):.2e} '
                f'time {np.mean(time_list):.4f} s')

    if cfg.params.save_results:
        # save results
        for category in p_preds_dict.keys():
            with h5py.File(osp.join(cfg.output.path, f'results_{category}'), 'w') as file:
                for i in range(len(x_dict[category])):
                    g = file.create_group(name=f'{i}')
                    g.create_dataset('f_samples', data=f_dict[category][i])
                    g.create_dataset('p_solved', data=y_dict[category][i])
                    g.create_dataset('p_preds', data=p_preds_dict[category][i])
                    g.create_dataset('pos', data=x_dict[category][i])
                    g.create_dataset('case_name', data=name_dict[category][i],
                                    dtype=h5py.string_dtype(encoding='utf-8'))
            logger.info(f'Saved results as {osp.join(cfg.output.path, f'results_{category}')}')


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg):
    """Main"""
    logger = logging.getLogger(__name__)
    if cfg.project.mode == 'train':
        train(cfg, logger)
    else:
        test(cfg, logger)


if __name__ == '__main__':
    main()
