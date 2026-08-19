import torch.nn as nn
import torch

from src.fvm_solvers.space_discretization import FVMSpaceDis


class PoissonResiduler(nn.Module):
    def __init__(self, nu, bouNu, volume, lowerIndex, upperIndex, weights, bouWeights, sf, bouSf, magSf,
                 bouMagSf, deltaCoeff, bouDeltaCoeff, face2cell, patchSize, patchOffset,
                 patchTypeP, patchTypeU, neighborPatchOffset, num_meshes, patch2mesh, cell_offset,
                 max_iter=20, scheme=0, omega=1.0, boundaryP=None):
        super().__init__()
        self.space_dis = FVMSpaceDis(nu, bouNu, volume, lowerIndex, upperIndex, weights, bouWeights, sf, bouSf,
                                     magSf, bouMagSf, deltaCoeff, bouDeltaCoeff, face2cell,
                                     patchSize, patchOffset, patchTypeP, patchTypeU, neighborPatchOffset, num_meshes,
                                     patch2mesh, cell_offset, boundaryP)
        self.max_iter = max_iter
        self.scheme = scheme
        self.omega = omega

    def forward(self, p, f):
        return self.compute_pcg_loss(p, f)
    
    def compute_pcg_loss(self, p, f):
        with torch.no_grad():
            d_p = p.double()
            d_f = f.double()
            Peqn = self.space_dis.construct_Peqn(d_p, d_f, dtype=d_p.dtype)
            d_p_ref, _ = Peqn.solve_pcg(d_p, max_iter=self.max_iter)
            p_ref = d_p_ref.to(p.dtype)
        if self.scheme == 0:
            diff = p_ref - p
            loss = torch.norm(diff)
        elif self.scheme == 1:
            r = Peqn.b - Peqn.Amul(p)
            loss = torch.norm(r)
        elif self.scheme == 2:
            diff = p_ref - p
            r = Peqn.b - Peqn.Amul(p)
            loss = torch.norm(diff) + self.omega * torch.norm(r)
        elif self.scheme == 3:
            diff = p_ref - p
            r = Peqn.b - Peqn.Amul(p)
            loss = diff.pow(2).mean().sqrt() + self.omega * r.pow(2).mean().sqrt()
        elif self.scheme == 4:
            diff = p_ref - p
            loss = diff.pow(2).mean().sqrt()
        elif self.scheme == 5:
            r = Peqn.b - Peqn.Amul(p)
            loss = r.pow(2).mean().sqrt()
        elif self.scheme == 6:
            diff = p_ref - p
            rel_diff = torch.norm(p_ref - p) / torch.norm(p_ref)
            r = Peqn.b - Peqn.Amul(p)
            loss = (rel_diff, diff.pow(2).mean().sqrt(), r.pow(2).mean().sqrt())
        elif self.scheme == 7:
            rel_diff = torch.norm(p_ref - p) / torch.norm(p_ref)
            loss = rel_diff
        else:
            raise ValueError(f'Unknown scheme {self.scheme} for PoissonResiduler.')
        return loss
    
    def compute_pcg_res(self, p, f):
        with torch.no_grad():
            d_p = p.double()
            d_f = f.double()
            Peqn = self.space_dis.construct_Peqn(d_p, d_f, dtype=d_p.dtype)
            d_p_ref, _ = Peqn.solve_pcg(d_p, max_iter=self.max_iter)
            p_ref = d_p_ref.to(p.dtype)
        r = Peqn.b - Peqn.Amul(p)
        return r