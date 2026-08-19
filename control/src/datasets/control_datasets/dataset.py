import os.path as osp
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..batch_mesh import BatchMesh
from ..data import Mesh, PatchType, BDVertexType


class GaussianHotSpotsGenerator:
    def __init__(self, scale_mode='mean_power', dim=2):
        self.scale_mode = scale_mode
        self.dim = dim

    def generate_gaussian_hotspots(
        self,
        pos: torch.Tensor,            # (n,2)
        K_range=(2, 6),
        amp_range=(0.5, 2.0),
        sigma_range=(0.02, 0.08),      # 相对坐标尺度：pos 若在[0,1]，这就合理
        anisotropic=False,
        scale_mode="mean_power",       # "max" or "mean_power"
        target_value=1.0,
        weights: torch.Tensor | None = None,  # (n,1) or (n,), 用于非均匀点云的“面积/体积权重”
        eps=1e-8,
        L=2*torch.pi,
    ):
        """
        返回 f: (n,1) 非负热源。适用于点云/非结构网格采样点。
        """
        n = pos.shape[0]
        x, y = pos[:, 0:1], pos[:, 1:2]
        sigma_range = (sigma_range[0] * L, sigma_range[1] * L)

        # 可选：低频背景（让不同样本有“工况背景差异”）
        f = torch.zeros((n, 1), device=pos.device, dtype=pos.dtype)

        # 热点个数
        K = int(torch.randint(K_range[0], K_range[1] + 1, (1,), device=pos.device).item())

        # 关键：中心从 pos 里抽，天然避开 hole
        center_ids = torch.randint(0, n, (K,), device=pos.device)
        centers = pos[center_ids]  # (K,2)

        for k in range(K):
            cx, cy = centers[k, 0].item(), centers[k, 1].item()
            A = torch.empty(1, device=pos.device).uniform_(amp_range[0], amp_range[1]).item()

            if anisotropic:
                sx = torch.empty(1, device=pos.device).uniform_(sigma_range[0], sigma_range[1]).item()
                sy = torch.empty(1, device=pos.device).uniform_(sigma_range[0], sigma_range[1]).item()
            else:
                s = torch.empty(1, device=pos.device).uniform_(sigma_range[0], sigma_range[1]).item()
                sx, sy = s, s

            gk = torch.exp(-0.5 * ((x - cx) / (sx + eps))**2 - 0.5 * ((y - cy) / (sy + eps))**2)
            f = f + A * gk

        # ---- 规模归一化：让不同样本“总发热功率”或“最大热源强度”可比 ----
        if scale_mode == "max":
            f = f * (target_value / (f.max() + eps))
        elif scale_mode == "mean_power":
            if weights is None:
                mean_val = f.mean()
            else:
                w = weights.reshape(-1, 1).to(f)
                mean_val = (f * w).sum() / (w.sum() + eps)
            f = f * (target_value / (mean_val + eps))
        else:
            raise ValueError(f"Unknown scale_mode: {scale_mode}")

        return f
    
    def __call__(self, pos):
        return self.generate_gaussian_hotspots(pos, scale_mode=self.scale_mode)


class TrainDataset(Dataset):
    def __init__(self, num_samples, mesh_file, dim=2, rms_f=False, f_scale_mode='mean_power'):
        super().__init__()
        self.dim = dim
        self.f_sampler = GaussianHotSpotsGenerator(dim=dim, scale_mode=f_scale_mode)
        self.num_samples = num_samples
        self.mesh_list = self.read_all_mesh_infos(mesh_file)
        self.num_meshs = len(self.mesh_list)
        self.rms_f = rms_f
                                                                                                                                                                                                                                                                                                                                                                                     
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
                    cell_offset=torch.tensor(g['cell_offset'][:], dtype=torch.long),
                    boundaryP=torch.tensor(g['boundaryP'][:], dtype=dtype),
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
        f_sample = -f_sample
        if self.rms_f:
            rms_f = self.compute_rms(f_sample)
        else:
            rms_f = f_sample
        return pos, f_sample, rms_f, pos_bd, mesh_info

    def compute_rms(self, f):
        s = torch.sqrt(torch.mean(f**2))
        rms_f = f / (s + 1e-8)
        return rms_f


def my_collate_fn_random_partition_fixedValue_bc(batch):
    b_size = len(batch)
    n_int_max, n_bd_max = 0, 0
    for pos, _, _, pos_bd, mesh_info in batch:
        n = pos.shape[0]
        n_bd = pos_bd.shape[0]
        n_int_max = max(n_int_max, n)
        n_bd_max = max(n_bd_max, n_bd)

    pos, f_sample, pos_bd = batch[0][0], batch[0][1], batch[0][3]
    pos_pad = torch.zeros((b_size, n_int_max, pos.shape[1]))
    f_pad = torch.zeros((b_size, n_int_max, f_sample.shape[1]))
    rms_f_pad = torch.zeros((b_size, n_int_max, f_sample.shape[1]))
    mask_pad = torch.zeros((b_size, n_int_max), dtype=torch.bool)
    pos_bd_pad = torch.zeros((b_size, n_bd_max, pos_bd.shape[1]))
    bc_type_bd_pad = torch.ones((b_size, n_bd_max, BDVertexType.SIZE)) * (-1)
    bc_value_bd_pad = torch.zeros((b_size, n_bd_max, 1), dtype=pos.dtype)
    mask_bd_pad = torch.zeros((b_size, n_bd_max), dtype=torch.bool)
    mesh_list = []
    for b_i, (pos, f_sample, rms_f, pos_bd, mesh_info) in enumerate(batch):
        n_int = pos.shape[0]
        n_bd = pos_bd.shape[0]
        pos_pad[b_i, :n_int] = pos
        f_pad[b_i, :n_int] = f_sample
        rms_f_pad[b_i, :n_int] = rms_f
        mask_pad[b_i, :n_int] = True
        pos_bd_pad[b_i, :n_bd] = pos_bd
        mask_bd_pad[b_i, :n_bd] = True
        bc_type_bd, bc_value_bd = set_random_partition_fixedValue_bc(mesh_info)
        bc_type_bd_pad[b_i, :n_bd] = bc_type_bd
        bc_value_bd_pad[b_i, :n_bd] = bc_value_bd
        mesh_list.append(mesh_info)
    batch_func = BatchMesh(mesh_list)
    batched_mesh = batch_func.batch_mesh()
    # (b_size, n_bd_max, pos_dim + bc_type_size + bc_value_size)
    x_bd_pad = torch.cat((pos_bd_pad, bc_type_bd_pad, bc_value_bd_pad), dim=-1)
    return pos_pad, f_pad, rms_f_pad, mask_pad, x_bd_pad, mask_bd_pad, batched_mesh


def set_random_partition_fixedValue_bc(mesh_info, bc_scale_value=10, part_num=4):
    bc_type_bd = torch.ones((mesh_info.num_boundary_surfaces,), dtype=torch.long) * (-1)
    bc_value_bd = torch.zeros((mesh_info.num_boundary_surfaces,), dtype=mesh_info.boundaryP.dtype)
    num_patches = mesh_info.num_patches
    for iPatch in range(num_patches):
        if mesh_info.patchTypeP[iPatch] == PatchType.FIXED_VALUE:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            patchSize = mesh_info.patchSize[iPatch, 0].item()
            q, r = divmod(patchSize, part_num)
            idx = [q + 1] * r + [q] * (part_num - r)
            start = 0
            for part_id in range(part_num):
                end = start + idx[part_id]
                part_faces = iFace[start: end]
                fixed_value = (torch.rand(1).to(device=mesh_info.boundaryP.device,
                                                dtype=mesh_info.boundaryP.dtype) - 0.5) * bc_scale_value
                mesh_info.boundaryP[part_faces, 0] = fixed_value
                bc_value_bd[part_faces] = fixed_value
                start = end
            bc_type_bd[iFace] = PatchType.FIXED_VALUE
        elif mesh_info.patchTypeP[iPatch] == PatchType.ZERO_GRADIENT:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            bc_value_bd[iFace] = 0.0
            bc_type_bd[iFace] = PatchType.ZERO_GRADIENT
        elif mesh_info.patchTypeP[iPatch] == PatchType.EMPTY:
            pass
        else:
            raise NotImplementedError(f'Patch type {mesh_info.patchTypeP[iPatch]} not implemented.')
    bc_type_bd = F.one_hot(bc_type_bd, num_classes=BDVertexType.SIZE).to(dtype=torch.float32)  # (bd_faces, bd_size)
    bc_value_bd = bc_value_bd.unsqueeze(-1)  # (bd_faces, 1)
    return bc_type_bd, bc_value_bd


class TestDataset(Dataset):
    def __init__(self, data_dir, file_names, dim=2, rms_f=False, apply_bc=True):
        self.data_dir = data_dir
        self.dim = 2
        self.sample_list = self.read_all_files(file_names)
        self.rms_f = rms_f
        self.apply_bc = apply_bc
        super().__init__()
    
    def read_all_files(self, file_names, dtype=torch.float32):
        sample_list = []
        for file_name in file_names:
            with h5py.File(osp.join(self.data_dir, file_name), 'r') as file:
                for key in file.keys():
                    g = file[key]
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
                        cell_offset=torch.tensor(g['cell_offset'][:], dtype=torch.long),
                        boundaryP=torch.tensor(g['boundaryP'][:], dtype=dtype),
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
        if self.rms_f:
            rms_f = self.compute_rms(f)
        else:
            rms_f = f
        pos_bd = mesh_info.pos_bd[:, :self.dim]
        if self.apply_bc:
            bc_type_bd, bc_value_bd = get_bc(mesh_info)
            # (num_bd_faces, pos_dim + bc_type_size + bc_value_size))
            x_bd = torch.cat((pos_bd, bc_type_bd, bc_value_bd), dim=-1)
        else:
            x_bd = pos_bd
        return x, f, rms_f, x_bd, mesh_info, y, file_name

    def compute_rms(self, f):
        s = torch.sqrt(torch.mean(f**2))
        rms_f = f / (s + 1e-8)
        return rms_f


def get_bc(mesh_info):
    num_patches = mesh_info.num_patches
    bc_type_bd = torch.ones((mesh_info.num_boundary_surfaces,), dtype=torch.long) * (-1)
    bc_value_bd = torch.zeros((mesh_info.num_boundary_surfaces,), dtype=mesh_info.boundaryP.dtype)
    for iPatch in range(num_patches):
        if mesh_info.patchTypeP[iPatch] == PatchType.FIXED_VALUE:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            bc_type_bd[iFace] = PatchType.FIXED_VALUE
            bc_value_bd[iFace] = mesh_info.boundaryP[iFace, 0]
            # bc_value_bd[iFace] = -100
        elif mesh_info.patchTypeP[iPatch] == PatchType.ZERO_GRADIENT:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            bc_value_bd[iFace] = 0.0
            bc_type_bd[iFace] = PatchType.ZERO_GRADIENT
        elif mesh_info.patchTypeP[iPatch] == PatchType.EMPTY:
            pass
        else:
            raise NotImplementedError(f'Patch type {mesh_info.patchTypeP[iPatch]} not implemented.')
    bc_type_bd = F.one_hot(bc_type_bd, num_classes=BDVertexType.SIZE).to(dtype=torch.float32)  # (bd_faces, bd_size)
    bc_value_bd = bc_value_bd.unsqueeze(-1)  # (bd_faces, 1)
    return bc_type_bd, bc_value_bd


def set_fixed_partition_fixedValue_bc(mesh_info, bc_values):
    device = bc_values.device
    part_num = bc_values.shape[0]
    bc_type_bd = torch.ones((mesh_info.num_boundary_surfaces,), dtype=torch.long, device=device) * (-1)
    bc_value_bd = torch.zeros((mesh_info.num_boundary_surfaces,), dtype=mesh_info.boundaryP.dtype, device=device)
    num_patches = mesh_info.num_patches
    for iPatch in range(num_patches):
        if mesh_info.patchTypeP[iPatch] == PatchType.FIXED_VALUE:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            patchSize = mesh_info.patchSize[iPatch, 0].item()
            q, r = divmod(patchSize, part_num)
            idx = [q + 1] * r + [q] * (part_num - r)
            start = 0
            for part_id in range(part_num):
                end = start + idx[part_id]
                part_faces = iFace[start: end]
                mesh_info.boundaryP[part_faces, 0] = bc_values[part_id].detach()
                bc_value_bd[part_faces] = bc_values[part_id]
                start = end
            bc_type_bd[iFace] = PatchType.FIXED_VALUE
        elif mesh_info.patchTypeP[iPatch] == PatchType.ZERO_GRADIENT:
            iFace = torch.arange(mesh_info.patchOffset[iPatch, 0],
                                    mesh_info.patchOffset[iPatch, 0] + mesh_info.patchSize[iPatch, 0])
            bc_value_bd[iFace] = 0.0
            bc_type_bd[iFace] = PatchType.ZERO_GRADIENT
        elif mesh_info.patchTypeP[iPatch] == PatchType.EMPTY:
            pass
        else:
            raise NotImplementedError(f'Patch type {mesh_info.patchTypeP[iPatch]} not implemented.')
    bc_type_bd = F.one_hot(bc_type_bd, num_classes=BDVertexType.SIZE).to(dtype=torch.float32)  # (bd_faces, bd_size)
    bc_value_bd = bc_value_bd.unsqueeze(-1)  # (bd_faces, 1)
    return bc_type_bd, bc_value_bd