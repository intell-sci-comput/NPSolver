import torch

from .possion_solver import PossionSolver


def init_solver(mesh_info, nu, device, dtype, boundaryP=False):
    nu_t = torch.ones((mesh_info.num_surfaces, 1), dtype=dtype, device=device) * nu
    bouNu = torch.ones((mesh_info.num_boundary_surfaces, 1), dtype=dtype, device=device) * nu
    if not boundaryP:
        mesh_info.boundaryP = None
    else:
        boundaryP = mesh_info.boundaryP.to(dtype)
    solver = PossionSolver(
            nu=nu_t,
            bouNu=bouNu,
            volume=mesh_info.volume.to(dtype),
            lowerIndex=mesh_info.lowerIndex,
            upperIndex=mesh_info.upperIndex,
            weights=mesh_info.weights.to(dtype),
            bouWeights=mesh_info.bouWeights.to(dtype),
            sf=mesh_info.sf.to(dtype),
            bouSf=mesh_info.bouSf.to(dtype),
            magSf=mesh_info.magSf.to(dtype),
            bouMagSf=mesh_info.bouMagSf.to(dtype),
            deltaCoeff=mesh_info.deltaCoeff.to(dtype),
            bouDeltaCoeff=mesh_info.bouDeltaCoeff.to(dtype),
            face2cell=mesh_info.face2cell,
            patchSize=mesh_info.patchSize,
            patchOffset=mesh_info.patchOffset,
            patchTypeP=mesh_info.patchTypeP,
            patchTypeU=mesh_info.patchTypeU,
            neighborPatchOffset=mesh_info.neighborPatchOffset,
            num_meshes=mesh_info.num_meshes,
            cell_offset=mesh_info.cell_offset,
            patch2mesh=mesh_info.patch2mesh,
            boundaryP=boundaryP
        )
    return solver