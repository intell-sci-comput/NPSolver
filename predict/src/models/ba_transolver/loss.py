import torch

from src.fvm_residulers.poisson_residuler import PoissonResiduler


class FVMBatchLoss(torch.nn.Module):
    def __init__(self, max_iter, scheme, omega, nu=1.0, boundaryP=False):
        super().__init__()
        self.max_iter = max_iter
        self.scheme = scheme
        self.omega = omega
        self.nu = nu
        self.boundaryP = boundaryP

    def construct_residuler(self, mesh_info, device, dtype):
        nu_t = torch.ones((mesh_info.num_surfaces, 1), dtype=dtype, device=device) * self.nu
        bouNu = torch.ones((mesh_info.num_boundary_surfaces, 1), dtype=dtype, device=device) * self.nu
        if not self.boundaryP:
            mesh_info.boundaryP = None
        residuler = PoissonResiduler(
            nu=nu_t,
            bouNu=bouNu,
            volume=mesh_info.volume,
            lowerIndex=mesh_info.lowerIndex,
            upperIndex=mesh_info.upperIndex,
            bouWeights=mesh_info.bouWeights,
            weights=mesh_info.weights,
            sf=mesh_info.sf,
            bouSf=mesh_info.bouSf,
            magSf=mesh_info.magSf,
            bouMagSf=mesh_info.bouMagSf,
            deltaCoeff=mesh_info.deltaCoeff,
            bouDeltaCoeff=mesh_info.bouDeltaCoeff,
            face2cell=mesh_info.face2cell,
            patchSize=mesh_info.patchSize,
            patchOffset=mesh_info.patchOffset,
            patchTypeP=mesh_info.patchTypeP,
            patchTypeU=mesh_info.patchTypeU,
            neighborPatchOffset=mesh_info.neighborPatchOffset,
            num_meshes=mesh_info.num_meshes,
            patch2mesh=mesh_info.patch2mesh,
            cell_offset=mesh_info.cell_offset,
            max_iter=self.max_iter,
            scheme=self.scheme,
            omega=self.omega,
            boundaryP=mesh_info.boundaryP
        )
        return residuler
    
    def forward(self, p_outputs, f_samples, mesh):
        """Compute the loss for the FVM model.
        Args:
            p_outputs: (n_total, 1)
            f_samples: (n_total, 1)
        """
        residuler = self.construct_residuler(mesh, device=p_outputs.device, dtype=p_outputs.dtype)
        return residuler(p_outputs, f_samples)

