import logging
import os
import os.path as osp
import swanlab
import hydra
import h5py
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

from src.datasets.control_datasets.dataset import TrainDataset, \
    TestDataset, my_collate_fn_random_partition_fixedValue_bc, set_fixed_partition_fixedValue_bc
from src.trainers.control_trainer import ControlTrainer
from src.trainers.utils import RMSELoss
from src.models.ba_transolver.Transolver_Irregular_Mesh import Model as BATransolver
from src.models.ba_transolver.loss import FVMBatchLoss, ControlLoss
from src.fvm_solvers.utils import init_solver


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
    dataset = TrainDataset(
        num_samples=cfg.params.num_batches_per_epoch * cfg.params.batch_size,
        mesh_file=osp.join(cfg.data.mesh_dir, cfg.data.file_name),
        rms_f=cfg.data.rms_f,
        f_scale_mode=cfg.data.f_scale_mode
    )
    collate_fn = my_collate_fn_random_partition_fixedValue_bc
    tr_loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.params.batch_size,
        num_workers=cfg.params.num_workers,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    # model
    ba_transolver = BATransolver(
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
    logger.info(f'Number of parameters: {sum(p.numel() for p in ba_transolver.parameters())}')
    optimizer = torch.optim.Adam(ba_transolver.parameters(), lr=cfg.params.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.params.lr, epochs=cfg.params.num_epochs,
        steps_per_epoch=cfg.params.num_batches_per_epoch, pct_start=cfg.params.pct_start
    )
    phy_loss_func = FVMBatchLoss(
        max_iter=cfg.model.max_iter,
        scheme=cfg.model.scheme,
        omega=cfg.model.omega,
        boundaryP=True
    )
    # trainer
    trainer = ControlTrainer(
        model=ba_transolver,
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
    cfg.model.scheme = 6
    cfg.model.max_iter = 0
    cfg.params.batch_size = 1
    
    # load data
    logger.info('Load data...')
    dataset = TestDataset(
        data_dir=cfg.data.sol_dir,
        file_names=cfg.data.test_files,
        rms_f=cfg.data.rms_f
    )
    # model
    ba_transolver = BATransolver(
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
    logger.info(f'Number of parameters: {sum(p.numel() for p in ba_transolver.parameters())}')
    # trainer
    phy_loss_func = FVMBatchLoss(
        max_iter=cfg.model.max_iter,
        scheme=cfg.model.scheme,
        omega=cfg.model.omega,
        boundaryP=True
    )
    trainer = ControlTrainer(
        model=ba_transolver,
        device=torch.device(cfg.project.device),
        cfg=cfg,
        optimizer=None,
        scheduler=None,
        phy_loss_func=phy_loss_func,
        data_loss_func=None,
    )
    # test
    x_dict, f_dict, p_preds_dict, y_dict, rel_diff_dict, diff_dict, r_dict, time_dict, rmse_dict, rel_l2_dict = trainer.test(dataset)  # dict of list
    rmse_list, rel_l2_list, diff_list, r_list, time_list, rel_diff_list = [], [], [], [], [], []
    for category in p_preds_dict.keys():
        logger.info(f'{category} - rmse {rmse_dict[category]: .2e}, '
                    f'rel l2 {rel_l2_dict[category]:.2e}, r {r_dict[category]:.2e} '
                    f'diff {diff_dict[category]:.2e} rel_diff: {rel_diff_dict[category]:.2e} '
                    f'time {time_dict[category]:.4f} s')
        rel_diff_list.append(rel_diff_dict[category])
        diff_list.append(diff_dict[category])
        rmse_list.append(rmse_dict[category])
        rel_l2_list.append(rel_l2_dict[category])
        r_list.append(r_dict[category])
        time_list.append(time_dict[category])
    logger.info(f'Overall - rmse {np.mean(rmse_list):.2e} '
                f'rel l2 {np.mean(rel_l2_list):.2e} '
                f'r {np.mean(r_list):.2e} '
                f'diff {np.mean(diff_list):.2e} '
                f'rel_diff: {np.mean(rel_diff_list):.2e} '
                f'time {np.mean(time_list):.4f} s')

    # save results
    if cfg.params.save_results:
        for category in p_preds_dict.keys():
            with h5py.File(osp.join(cfg.output.path, f'results_{category}'), 'w') as file:
                for i in range(len(x_dict[category])):
                    g = file.create_group(name=f'{i}')
                    g.create_dataset('f_samples', data=f_dict[category][i])
                    g.create_dataset('p_solved', data=y_dict[category][i])
                    g.create_dataset('p_preds', data=p_preds_dict[category][i])
                    g.create_dataset('pos', data=x_dict[category][i])


def control(cfg, logger):
    logger.info('================= Control ==================')
    cfg.params.batch_size = 1
    weight = 0.001
    threshold = 25
    epochs = 100
    lr = 1.0
    device = torch.device(cfg.project.device)
    
    # load data
    logger.info('Load data...')
    dataset = TestDataset(
        data_dir=cfg.data.sol_dir,
        file_names=cfg.data.test_files,
        rms_f=cfg.data.rms_f,
        apply_bc=False
    )
    sample_indices = np.arange(len(dataset))
    sample_num = len(sample_indices)
    info_list = []
    all_time_list = []
    success_cnt, mismatch_cnt, control_cnt = 0, 0, 0
    ood_fail_cnt, optim_fail_cnt = 0, 0
    u_max_before_list, u_max_after_list = [], []
    u_max_reduced_list, overshot_list = [], []
    cooling_cost_list = []
    for sample_index in sample_indices:
        sample = dataset[sample_index]
        x, f, rms_f, pos_bd, mesh, y, file_name = sample
        part_num = 4
        init_bc_values = 0.1 * torch.randn(part_num, 1, dtype=torch.float32)
        # model
        ba_transolver = BATransolver(
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
        # trainer
        trainer = ControlTrainer(
            model=ba_transolver,
            device=torch.device(cfg.project.device),
            cfg=cfg,
            optimizer=None,
            scheduler=None,
            phy_loss_func=None,
            data_loss_func=None,
        )
        loss_func = ControlLoss(weight=weight, threshold=threshold)
        is_control, p_pred_zero_bc, p_pred, bc_values, mesh, time_list, loss_list = trainer.control(
            sample=sample,
            init_bc_values=init_bc_values,
            loss_func=loss_func,
            epochs=epochs,
            lr=lr,
            threshold=threshold
        )

        cooling_cost_list.append(torch.mean(torch.abs(bc_values)).item())
        solver_dtype = torch.float64
        f_sample = f  # (num_cells, 1)
        f_sample = f_sample.to(solver_dtype).to(device)
        if not is_control:
            success_cnt += 1
            # before optimize
            zero_bc_values = torch.zeros_like(init_bc_values)
            set_fixed_partition_fixedValue_bc(mesh, zero_bc_values)
            solver = init_solver(
                mesh_info=mesh,
                nu=1.0,
                device=device,
                dtype=solver_dtype,
                boundaryP=True
            )
            p_before, _ = solver(f_sample)

            p_gt = p_before
            info = f'Sample {sample_index} initial zero BC already satisfies the threshold constraint: {p_pred.max().item()}. Skip optimization.'
            info_list.append(info)
            print(info)
        else:
            control_cnt += 1
            all_time_list.append(np.sum(time_list))
            # after optimize
            mesh = mesh.to(device)
            solver = init_solver(
                mesh_info=mesh,
                nu=1.0,
                device=device,
                dtype=torch.float64,
                boundaryP=True
            )
            p_gt, _ = solver(f_sample)
            if p_gt.max() <= threshold:
                success_cnt += 1
            elif p_gt.max() > threshold and p_pred.max() <= threshold:  # in-distribution surrogate error
                mismatch_cnt += 1
            elif p_gt.max() > threshold and p_pred.max() > threshold:  # if fail
                # check whether the boundary value -5 satisfies the threshold constraint
                set_fixed_partition_fixedValue_bc(mesh, -5 * torch.ones_like(init_bc_values))
                solver = init_solver(
                    mesh_info=mesh,
                    nu=1.0,
                    device=device,
                    dtype=solver_dtype,
                    boundaryP=True
                )
                p_ood, _ = solver(f_sample)
                if p_ood.max() <= threshold:
                    optim_fail_cnt += 1
                else:
                    ood_fail_cnt += 1

            # before optimize
            zero_bc_values = torch.zeros_like(init_bc_values)
            set_fixed_partition_fixedValue_bc(mesh, zero_bc_values)
            solver = init_solver(
                mesh_info=mesh,
                nu=1.0,
                device=device,
                dtype=solver_dtype,
                boundaryP=True
            )
            p_before, _ = solver(f_sample)
            u_max_before_list.append(p_before.max().item())
            u_max_after_list.append(p_gt.max().item())
            u_max_reduced_list.append(p_before.max().item() - p_gt.max().item())
            overshot_list.append(max(p_gt.max().item() - threshold, 0))

            

        info = f'Sample {sample_index} Threshod: {threshold}, before U_max: {p_before.max()}, '\
               f'after U_max: {p_pred.max()}, ground truth U_max: {p_gt.max()}'
        print(info)
        info_list.append(info)

        if cfg.params.save_results:
            tgt_file = f'control_results_{sample_index}.h5'
            output_path = osp.join(cfg.output.path, 'outputs')
            if not osp.exists(output_path):
                os.makedirs(output_path)
            with h5py.File(osp.join(output_path, tgt_file), 'w') as file:
                file.create_dataset('f_samples', data=f_sample.cpu().numpy())
                file.create_dataset('p_solved', data=p_gt.cpu().numpy())
                file.create_dataset('p_before', data=p_before.detach().cpu().numpy())
                file.create_dataset('p_pred', data=p_pred.detach().cpu().numpy())
                file.create_dataset('pos', data=mesh.pos.cpu().numpy())
                file.create_dataset('time', data=np.array(time_list))
                file.create_dataset('loss', data=np.stack(loss_list))
    print('================= Summary ==================')
    for info in info_list:
        print(info)
    print(f'Success rate {success_cnt/sample_num:.3f}, Mismatch rate {mismatch_cnt/sample_num:.3f}, '
          f'Not control rate {1-control_cnt/sample_num:.3f}, '
          f'U_max before/after {np.mean(u_max_before_list):.3f}/{np.mean(u_max_after_list):.3f}, '
          f'U_max reduced {np.mean(u_max_reduced_list):.3f}, '
          f'Overshot {np.mean(overshot_list):.3f}, '
          f'Cooling cost {np.mean(cooling_cost_list):.3f} Time cost {np.mean(all_time_list):.3f}')
    print(f'OOD fail rate {ood_fail_cnt/sample_num:.3f}, '
          f'Optimization fail rate {optim_fail_cnt/sample_num:.3f}')


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg):
    """Main"""
    logger = logging.getLogger(__name__)
    if cfg.project.mode == 'train':
        train(cfg, logger)
    elif cfg.project.mode == 'test':
        test(cfg, logger)
    elif cfg.project.mode == 'control':
        control(cfg, logger)
    else:
        raise ValueError(f"Unknown mode: {cfg.project.mode}")


if __name__ == '__main__':
    main()
