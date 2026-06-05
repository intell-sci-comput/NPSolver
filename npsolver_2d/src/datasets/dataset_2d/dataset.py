import os.path as osp
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..batch_mesh import BatchMesh
from ..data import Mesh, PatchType, BDVertexType


class FourierGenerator:
    def __init__(self, M, dim=2, norm='max'):
        self.dim = dim
        self.M = M
        self.norm = norm

    def get_fourier_init_condition(self, cells_centroid):
        M = self.M
        xx = cells_centroid[:, 0]
        yy = cells_centroid[:, 1]
        W = 0.
        alpha = torch.normal(0., 1., size=(M, M)).to(device=cells_centroid.device)
        beta = torch.normal(0., 1., size=(M, M)).to(device=cells_centroid.device)
        gamma = torch.normal(-1., 1., size=(1,)).to(device=cells_centroid.device)
        for i in range(M):
            for j in range(M):
                phase = (i - M // 2) * xx + (j - M // 2) * yy
                W = W + alpha[i, j] * torch.sin(phase) + beta[i, j] * torch.cos(phase)

        U = W + gamma[0]
        if self.norm == 'max':
            U = U / (U.abs().max() + 1e-8)
        elif self.norm == 'gauss':
            U = (U - U.mean()) / (U.std() + 1e-8)
        else:
            raise NotImplementedError(f'Normalization {self.norm} not implemented.')
        return U.unsqueeze(-1)  # [num_cells, 1]
    
    def __call__(self, pos):
        return self.get_fourier_init_condition(pos)


class TrainDataset(Dataset):
    def __init__(self, num_samples, mesh_file, M=10, dim=2, norm_fourier='max', norm_f=False):
        super().__init__()
        self.dim = dim
        self.f_sampler = FourierGenerator(M=M, dim=dim, norm=norm_fourier)
        self.num_samples = num_samples
        self.mesh_list = self.read_all_mesh_infos(mesh_file)
        self.num_meshs = len(self.mesh_list)
        self.norm_f = norm_f
                                                                                                                                                                                                                                                                                                                                                                                     
    def read_all_mesh_infos(self, mesh_file, dtype=torch.float32):
        mesh_list = []
        with h5py.File(mesh_file, 'r') as f:
            for key in f.keys():
                g = f[key]
                mesh_info = Mesh(
                    pos=torch.tensor(g['pos'][:], dtype=dtype),
                    pos_bd=torch.tensor(g['bou_faces_centroid'][:], dtype=dtype),
                    num_cells=int(g['num_cells'][()]),
                    num_surfaces=int(g['num_surfaces'][()]),
                    num_boundary_surfaces=int(g['num_boundary_surfaces'][()]),
                    num_patches=int(g['num_patches'][()]),
                    volume=torch.tensor(g['volume'][:], dtype=dtype),
                    lowerIndex=torch.tensor(g['lowerIndex'][:], dtype=torch.long),
                    upperIndex=torch.tensor(g['upperIndex'][:], dtype=torch.long),
                    bouWeights=torch.tensor(g['bouWeights'][:], dtype=dtype),
                    weights=torch.tensor(g['weights'][:], dtype=dtype),
                    sf=torch.tensor(g['sf'][:], dtype=dtype),
                    bouSf=torch.tensor(g['bouSf'][:], dtype=dtype),
                    magSf=torch.tensor(g['magSf'][:], dtype=dtype),
                    bouMagSf=torch.tensor(g['bouMagSf'][:], dtype=dtype),
                    deltaCoeff=torch.tensor(g['deltaCoeff'][:], dtype=dtype),
                    bouDeltaCoeff=torch.tensor(g['bouDeltaCoeff'][:], dtype=dtype),
                    face2cell=torch.tensor(g['face2cell'][:], dtype=torch.long),
                    patchSize=torch.tensor(g['patchSize'][:], dtype=torch.long),
                    patchOffset=torch.tensor(g['patchOffset'][:], dtype=torch.long),
                    patchTypeP=torch.tensor(g['patchTypeP'][:], dtype=torch.long),
                    patchTypeU=torch.tensor(g['patchTypeU'][:], dtype=torch.long),
                    neighborPatchOffset=torch.tensor(g['neighborPatchOffset'][:], dtype=torch.long),
                    num_meshes=int(g['num_meshes'][()]),
                    patch2mesh=torch.tensor(g['patch2mesh'][:], dtype=torch.long),
                    cell_offset=torch.tensor(g['cell_offset'][:], dtype=torch.long)
                )
                mesh_list.append(mesh_info)
        return mesh_list

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        mesh_info = np.random.choice(self.mesh_list)
        pos = mesh_info.pos[:, :self.dim]
        pos_bd = mesh_info.pos_bd[:, :self.dim]
        f_sample = self.f_sampler(pos)
        if self.norm_f:
            f_sample = self.normalize_f(f_sample, mesh_info)
        return pos, f_sample, pos_bd, mesh_info

    def normalize_f(self, f, mesh_info):
        mean_f = torch.sum(f * mesh_info.volume) / torch.sum(mesh_info.volume)
        normed_f = f - mean_f
        return normed_f


def my_collate_fn(batch):
    b_size = len(batch)
    n_int_max, n_bd_max = 0, 0
    for pos, _, pos_bd, _ in batch:
        n = pos.shape[0]
        n_bd = pos_bd.shape[0]
        n_int_max = max(n_int_max, n)
        n_bd_max = max(n_bd_max, n_bd)

    pos, f_sample, pos_bd = batch[0][0], batch[0][1], batch[0][2]
    pos_pad = torch.zeros((b_size, n_int_max, pos.shape[1]))
    f_pad = torch.zeros((b_size, n_int_max, f_sample.shape[1]))
    mask_pad = torch.zeros((b_size, n_int_max), dtype=torch.bool)
    pos_bd_pad = torch.zeros((b_size, n_bd_max, pos_bd.shape[1]))
    mask_bd_pad = torch.zeros((b_size, n_bd_max), dtype=torch.bool)
    mesh_list = []
    for b_i, (pos, f_sample, pos_bd, mesh_info) in enumerate(batch):
        n_int = pos.shape[0]
        n_bd = pos_bd.shape[0]
        pos_pad[b_i, :n_int] = pos
        f_pad[b_i, :n_int] = f_sample
        mask_pad[b_i, :n_int] = True
        pos_bd_pad[b_i, :n_bd] = pos_bd
        mask_bd_pad[b_i, :n_bd] = True
        mesh_list.append(mesh_info)
    batch_func = BatchMesh(mesh_list)
    batched_mesh = batch_func.batch_mesh()
    return pos_pad, f_pad, mask_pad, pos_bd_pad, mask_bd_pad, batched_mesh


def my_collate_fn_zeroGradient_bc(batch):
    b_size = len(batch)
    n_int_max, n_bd_max = 0, 0
    for pos, _, pos_bd, _ in batch:
        n = pos.shape[0]
        n_bd = pos_bd.shape[0]
        n_int_max = max(n_int_max, n)
        n_bd_max = max(n_bd_max, n_bd)

    pos, f_sample, pos_bd = batch[0][0], batch[0][1], batch[0][2]
    pos_pad = torch.zeros((b_size, n_int_max, pos.shape[1]))
    f_pad = torch.zeros((b_size, n_int_max, f_sample.shape[1]))
    mask_pad = torch.zeros((b_size, n_int_max), dtype=torch.bool)
    pos_bd_pad = torch.zeros((b_size, n_bd_max, pos_bd.shape[1]))
    mask_bd_pad = torch.zeros((b_size, n_bd_max), dtype=torch.bool)
    mesh_list = []
    for b_i, (pos, f_sample, pos_bd, mesh_info) in enumerate(batch):
        n_int = pos.shape[0]
        n_bd = pos_bd.shape[0]
        pos_pad[b_i, :n_int] = pos
        f_pad[b_i, :n_int] = f_sample
        mask_pad[b_i, :n_int] = True
        pos_bd_pad[b_i, :n_bd] = pos_bd
        mask_bd_pad[b_i, :n_bd] = True
        set_zeroGradient_bc(mesh_info)
        mesh_list.append(mesh_info)
    batch_func = BatchMesh(mesh_list)
    batched_mesh = batch_func.batch_mesh()
    return pos_pad, f_pad, mask_pad, pos_bd_pad, mask_bd_pad, batched_mesh


def my_collate_fn_random_bc(batch):
    b_size = len(batch)
    n_int_max, n_bd_max = 0, 0
    for pos, _, pos_bd, mesh_info in batch:
        n = pos.shape[0]
        n_bd = pos_bd.shape[0]
        n_int_max = max(n_int_max, n)
        n_bd_max = max(n_bd_max, n_bd)

    pos, f_sample, pos_bd = batch[0][0], batch[0][1], batch[0][2]
    pos_pad = torch.zeros((b_size, n_int_max, pos.shape[1]))
    f_pad = torch.zeros((b_size, n_int_max, f_sample.shape[1]))
    mask_pad = torch.zeros((b_size, n_int_max), dtype=torch.bool)
    pos_bd_pad = torch.zeros((b_size, n_bd_max, pos_bd.shape[1]))
    bc_bd_pad = torch.ones((b_size, n_bd_max, BDVertexType.SIZE)) * (-1)
    mask_bd_pad = torch.zeros((b_size, n_bd_max), dtype=torch.bool)
    mesh_list = []
    for b_i, (pos, f_sample, pos_bd, mesh_info) in enumerate(batch):
        n_int = pos.shape[0]
        n_bd = pos_bd.shape[0]
        pos_pad[b_i, :n_int] = pos
        f_pad[b_i, :n_int] = f_sample
        mask_pad[b_i, :n_int] = True
        pos_bd_pad[b_i, :n_bd] = pos_bd
        mask_bd_pad[b_i, :n_bd] = True
        bc_bd = set_random_bc(mesh_info)
        bc_bd_pad[b_i, :n_bd] = bc_bd
        mesh_list.append(mesh_info)
    batch_func = BatchMesh(mesh_list)
    batched_mesh = batch_func.batch_mesh()
    x_bd_pad = torch.cat((pos_bd_pad, bc_bd_pad), dim=-1)  # (b_size, n_bd_max, pos_dim + bd_size)
    return pos_pad, f_pad, mask_pad, x_bd_pad, mask_bd_pad, batched_mesh


def set_random_bc(mesh_info):
    num_patches = mesh_info.num_patches
    patchTypeP = []
    bc_bd = torch.ones((mesh_info.num_boundary_surfaces,), dtype=torch.long) * (-1)
    for i in range(num_patches):
        if mesh_info.patchTypeP[i, 0] == PatchType.EMPTY:
            patchTypeP.append(PatchType.EMPTY)
            continue
        if np.random.rand() < 0.5:
            bc_type = PatchType.FIXED_VALUE
        else:
            bc_type = PatchType.ZERO_GRADIENT
        patchTypeP.append(bc_type)
        iFace = torch.arange(mesh_info.patchOffset[i, 0],
                             mesh_info.patchOffset[i, 0] + mesh_info.patchSize[i, 0])
        bc_bd[iFace] = bc_type
    mesh_info.patchTypeP = torch.tensor(patchTypeP, dtype=torch.long).unsqueeze(dim=-1)
    bc_bd = F.one_hot(bc_bd, num_classes=BDVertexType.SIZE).to(dtype=torch.float32)  # (bd_faces, bd_size)
    return bc_bd


def set_zeroGradient_bc(mesh_info):
    num_patches = mesh_info.num_patches
    patchTypeP = []
    for i in range(num_patches):
        if mesh_info.patchTypeP[i] == PatchType.EMPTY:
            patchTypeP.append(PatchType.EMPTY)
            continue
        patchTypeP.append(PatchType.ZERO_GRADIENT)
    mesh_info.patchTypeP = torch.tensor(patchTypeP, dtype=torch.long,
                                        device=mesh_info.patchTypeP.device).unsqueeze(dim=-1)


class TestDataset(Dataset):
    def __init__(self, data_dir, file_names, dim=2, apply_bc=False):
        self.data_dir = data_dir
        self.dim = 2
        self.sample_list = self.read_all_files(file_names)
        self.apply_bc = apply_bc
        super().__init__()
    
    def read_all_files(self, file_names, dtype=torch.float32):
        sample_list = []
        for file_name in file_names:
            with h5py.File(osp.join(self.data_dir, file_name), 'r') as file:
                length = file.keys()
                for i in range(len(length)):
                    g = file[str(i)]
                    mesh_info = Mesh(
                        pos=torch.tensor(g['pos'][:], dtype=dtype),
                        pos_bd=torch.tensor(g['bou_faces_centroid'][:], dtype=dtype),
                        num_cells=int(g['num_cells'][()]),
                        num_surfaces=int(g['num_surfaces'][()]),
                        num_boundary_surfaces=int(g['num_boundary_surfaces'][()]),
                        num_patches=int(g['num_patches'][()]),
                        volume=torch.tensor(g['volume'][:], dtype=dtype),
                        lowerIndex=torch.tensor(g['lowerIndex'][:], dtype=torch.long),
                        upperIndex=torch.tensor(g['upperIndex'][:], dtype=torch.long),
                        bouWeights=torch.tensor(g['bouWeights'][:], dtype=dtype),
                        weights=torch.tensor(g['weights'][:], dtype=dtype),
                        sf=torch.tensor(g['sf'][:], dtype=dtype),
                        bouSf=torch.tensor(g['bouSf'][:], dtype=dtype),
                        magSf=torch.tensor(g['magSf'][:], dtype=dtype),
                        bouMagSf=torch.tensor(g['bouMagSf'][:], dtype=dtype),
                        deltaCoeff=torch.tensor(g['deltaCoeff'][:], dtype=dtype),
                        bouDeltaCoeff=torch.tensor(g['bouDeltaCoeff'][:], dtype=dtype),
                        face2cell=torch.tensor(g['face2cell'][:], dtype=torch.long),
                        patchSize=torch.tensor(g['patchSize'][:], dtype=torch.long),
                        patchOffset=torch.tensor(g['patchOffset'][:], dtype=torch.long),
                        patchTypeP=torch.tensor(g['patchTypeP'][:], dtype=torch.long),
                        patchTypeU=torch.tensor(g['patchTypeU'][:], dtype=torch.long),
                        neighborPatchOffset=torch.tensor(g['neighborPatchOffset'][:], dtype=torch.long),
                        num_meshes=int(g['num_meshes'][()]),
                        patch2mesh=torch.tensor(g['patch2mesh'][:], dtype=torch.long),
                        cell_offset=torch.tensor(g['cell_offset'][:], dtype=torch.long)
                    )
                    x = torch.tensor(g['pos'][:, :self.dim], dtype=dtype)
                    f = torch.tensor(g['f_samples'][:], dtype=dtype)
                    y = torch.tensor(g['p_solved'][:], dtype=dtype)
                    sample_list.append([x, f, mesh_info, y, file_name])
        return sample_list
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        x, f, mesh_info, y, file_name = self.sample_list[index]
        pos_bd = mesh_info.pos_bd[:, :self.dim]
        if self.apply_bc:
            bc_bd = get_bc(mesh_info)
            x_bd = torch.cat((pos_bd, bc_bd), dim=-1)  # (num_bd_faces, dim+1)
        else:
            x_bd = pos_bd
        return x, f, x_bd, mesh_info, y, file_name


def get_bc(mesh_info):
    num_patches = mesh_info.num_patches
    bc_bd = torch.ones((mesh_info.num_boundary_surfaces,), dtype=torch.long) * (-1)
    for i in range(num_patches):
        iFace = torch.arange(mesh_info.patchOffset[i, 0],
                            mesh_info.patchOffset[i, 0] + mesh_info.patchSize[i, 0])
        bc_bd[iFace] = mesh_info.patchTypeP[i, 0]
    bc_bd = F.one_hot(bc_bd, num_classes=BDVertexType.SIZE).to(dtype=torch.float32)  # (bd_faces, bd_size)
    return bc_bd