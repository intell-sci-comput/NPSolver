from .npsolver_trainer import NPSolverTrainer
from .npsolver_nm_trainer import NPSolverNMTrainer


class AutoTrainer:
    """Auto Trainer to select the appropriate trainer based on the configuration."""
    
    @staticmethod
    def get_trainer(cfg, model, device, optimizer=None, scheduler=None, phy_loss_func=None, data_loss_func=None):
        if cfg.data.name == 'dirichlet' or cfg.data.name == 'random_bc':
            return NPSolverTrainer(
                model=model,
                device=device,
                cfg=cfg,
                optimizer=optimizer,
                scheduler=scheduler,
                phy_loss_func=phy_loss_func,
                data_loss_func=data_loss_func
            )
        elif cfg.data.name == 'neumann':
            return NPSolverNMTrainer(
                model=model,
                device=device,
                cfg=cfg,
                optimizer=optimizer,
                scheduler=scheduler,
                phy_loss_func=phy_loss_func,
                data_loss_func=data_loss_func
            )
        else:
            raise ValueError(f'Unknown dataset name {cfg.data.name} for AutoTrainer.')

    