from tqdm import tqdm
import torch

from .space_discretization import FVMSpaceDis


class PoissonSolver:
    def __init__(self, nu, bouNu, volume, lowerIndex, upperIndex, weights, bouWeights, sf, bouSf, magSf,
                 bouMagSf, deltaCoeff, bouDeltaCoeff, face2cell, patchSize, patchOffset,
                 patchTypeP, patchTypeU, neighborPatchOffset, num_meshes, patch2mesh, cell_offset,
                 boundaryP=None):
        self.space_dis = FVMSpaceDis(nu, bouNu, volume, lowerIndex, upperIndex, weights, bouWeights, sf, bouSf,
                                     magSf, bouMagSf, deltaCoeff, bouDeltaCoeff, face2cell,
                                     patchSize, patchOffset, patchTypeP, patchTypeU, neighborPatchOffset, num_meshes,
                                     patch2mesh, cell_offset, boundaryP)

    def __call__(self, f):
        p = torch.zeros_like(f)
        return self.solve_poisson(p, f)
    
    def solve_poisson(self, p, f):
        p_new, r_true_list = self.space_dis.solve_poisson_linear_system_ldu(p, f)
        return p_new, r_true_list