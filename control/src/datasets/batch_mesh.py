import torch

from .data import Mesh


class BatchMesh:
    def __init__(self, mesh_list):
        self.mesh_list = mesh_list
        self.num_meshes = len(mesh_list)
        self.cells_count = []
        self.surfaces_count = []
        self.boundary_surfaces_count = []
        self.patches_count = []
        self.cell_offset = []

    def batch_mesh(self):
        batched = Mesh()
        # collect keys from the first mesh
        mesh0 = self.mesh_list[0]
        for key in mesh0.__dict__.keys():
            setattr(batched, key, [])
        # concatenate attributes
        offset_cell = 0
        offset_surface = 0
        offset_bou_surface = 0
        offset_patch = 0
        for mesh_i, mesh in enumerate(self.mesh_list):
            for key, value in mesh.__dict__.items():
                if key == 'lowerIndex':
                    value = value + offset_cell
                elif key == 'upperIndex':
                    value = value + offset_cell
                elif key == 'face2cell':
                    value = value + offset_cell
                elif key == 'patchOffset' or key == 'neighborPatchOffset':
                    value = value + offset_bou_surface
                elif key == 'cell_offset':
                    value = value + offset_cell
                elif key == 'patch2mesh':
                    value = value + mesh_i
                getattr(batched, key).append(value)
            offset_cell += mesh.num_cells
            offset_surface += mesh.num_surfaces
            offset_bou_surface += mesh.num_boundary_surfaces
            offset_patch += mesh.num_patches

        self.cells_count = batched.num_cells
        self.surfaces_count = batched.num_surfaces
        self.boundary_surfaces_count = batched.num_boundary_surfaces
        self.patches_count = batched.num_patches
        for key in batched.__dict__.keys():
            if key.startswith('num_'):
                setattr(batched, key, sum(getattr(batched, key)))
            elif key == 'patchName':
                continue
            else:
                setattr(batched, key, torch.cat(getattr(batched, key), dim=0))
        return batched
    
    def batch_cells_var(self, cells_var_list):
        if isinstance(cells_var_list, list):
            return torch.cat(cells_var_list, dim=0)
        else:
            assert len(cells_var_list.shape) == 3  # (b, num_cells, var_dim)
            return cells_var_list.reshape(-1, cells_var_list.shape[-1])
    
    def unbatch_cells_var(self, batched_var):
        return torch.split(batched_var, self.cells_count, dim=0)