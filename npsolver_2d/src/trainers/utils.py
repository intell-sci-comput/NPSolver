import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class RelativeL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        y_true_norm = torch.linalg.norm(y_true)
        rel_l2 = torch.linalg.norm(y_pred - y_true) / (y_true_norm + 1e-8)
        return rel_l2


class FVMLoss(torch.nn.Module):
    def __init__(self, fvm_residuler):
        super().__init__()
        self.residuler = fvm_residuler
    
    def compute_loss(self, p_output, f_sample):
        """Compute the loss for the FVM model.
        """
        p_output = p_output.reshape(-1, 1)
        f_sample = f_sample.reshape(-1, 1)
        poss_res = self.residuler(p_output, f_sample)
        return poss_res
    
    def forward(self, p_outputs, f_samples):
        """Compute the loss for the FVM model.
        """
        num_steps = p_outputs.shape[0]
        total_loss = 0.0
        for i in range(num_steps):
            p_output = p_outputs[i][0]
            f_sample = f_samples[i][0]
            poss_loss = self.compute_loss(p_output, f_sample)
            total_loss = total_loss + poss_loss
        total_loss = total_loss / num_steps
        return total_loss


class FVMBatchLoss(torch.nn.Module):
    def __init__(self, fvm_residuler, batch_func):
        super().__init__()
        self.residuler = fvm_residuler
        self.batch_func = batch_func
    
    def compute_loss(self, p_output, f_sample):
        """Compute the loss for the FVM model.
        """
        poss_res = self.residuler(p_output, f_sample)
        return poss_res
    
    def forward(self, p_outputs, f_samples):
        """Compute the loss for the FVM model.
        Args:
            p_outputs: (b, 1, h, w)
            f_samples: (b, 1, h, w)
        """
        p_outputs = p_outputs.reshape(p_outputs.shape[0], -1, 1)  # (b, num_cells, 1)
        f_samples = f_samples.reshape(f_samples.shape[0], -1, 1)
        batched_p_outputs = self.batch_func.batch_cells_var(p_outputs)  # (b*num_cells, 1)
        batched_f_samples = self.batch_func.batch_cells_var(f_samples)
        return self.compute_loss(batched_p_outputs, batched_f_samples)


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
