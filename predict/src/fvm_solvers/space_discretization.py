import torch
from torch_scatter import scatter_add

from src.datasets.data import PatchType

class FVMSpaceDis:
    def __init__(self, nu, bouNu, volume, lowerIndex, upperIndex, weights, bouWeights, sf, bouSf, magSf,
                 bouMagSf, deltaCoeff, bouDeltaCoeff, face2cell, patchSize, patchOffset,
                 patchTypeP, patchTypeU, neighborPatchOffset, num_meshes, patch2mesh, cell_offset, boundaryP=None):
        """ Basic Info:
            nu (tensor): [num_surfaces, 3]
            bouNu (tensor): [num_boundary_surfaces, 3]
            volume (tensor): [num_cells, 1]
            lowerIndex (tensor): [num_surfaces, 1]
            upperIndex (tensor): [num_surfaces, 1]
            weights (tensor): [num_surfaces,]
            sf (tensor): [num_surfaces, 3]
            bouSf (tensor): [num_boundary_surfaces, 3]
            magSf (tensor): [num_surfaces, 1]
            bouMagSf (tensor): [num_boundary_surfaces, 1]
            deltaCoeff (tensor): [num_surfaces, 1]
            bouDeltaCoeff (tensor): [num_boundary_surfaces, 1]
            face2cell (tensor): [num_boundary_surfaces, 1]
            patchSize (tensor, int): [num_patches, 1]
            patchOffset (tensor, int): [num_patches, 1]
            patchType (list, int): [num_patches, 1]
            num_meshes (int): number of meshes in the batch
            patch2mesh (tensor, int): [num_patches, 1], mapping from patch to mesh index
            cell_offset (tensor, int): [num_meshes, 1], cell index offset for each mesh in the batch
        """
        self.nu = nu
        self.bouNu = bouNu
        self.volume = volume
        self.lowerIndex = lowerIndex
        self.upperIndex = upperIndex
        self.weights = weights
        self.bouWeights = bouWeights
        self.sf = sf
        self.bouSf = bouSf
        self.magSf = magSf
        self.bouMagSf = bouMagSf
        self.deltaCoeff = deltaCoeff
        self.bouDeltaCoeff = bouDeltaCoeff
        self.face2cell = face2cell

        self.patchSize = patchSize.squeeze()
        self.patchOffset = patchOffset.squeeze()
        self.patchTypeP = patchTypeP.squeeze()
        self.patchTypeU = patchTypeU.squeeze()
        self.neighborPatchOffset = neighborPatchOffset.squeeze()
        self.boundaryP = boundaryP

        # for batch mesh
        self.num_meshes = num_meshes
        self.patch2mesh = patch2mesh.squeeze()
        self.cell_offset = cell_offset.squeeze(dim=-1)
        self.num_surfaces = self.sf.shape[0]
        self.num_boundary_surfaces = self.bouSf.shape[0]
        self.num_cells = self.volume.shape[0]
        self.num_patches = self.patchSize.shape[0]

        self.all_cyclic = False
        self.p_fixedValue_faces, self.p_zeroGradient_faces, self.p_cyclic_faces, \
            self.p_neighbor_faces, self.p_has_fixedValue_patch = \
                self.collect_boundary_faces(self.patchTypeP, self.nu.device)
    
    def collect_boundary_faces(self, patchType, device):
        """ Collect boundary faces for different BC types
        """
        fixedValue_faces = []
        zeroGradient_faces = []
        cyclic_faces, neighbor_faces = [], []
        has_fixedValue_patch = [False] * self.num_meshes
        for iPatch in range(self.num_patches):
            if patchType[iPatch] == PatchType.FIXED_VALUE:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=device)
                fixedValue_faces.append(iFace)
                has_fixedValue_patch[self.patch2mesh[iPatch]] = True
            elif patchType[iPatch] == PatchType.ZERO_GRADIENT:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=device)
                zeroGradient_faces.append(iFace)
            elif patchType[iPatch] == PatchType.CYCLIC:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=device)
                neighborFace = torch.arange(self.neighborPatchOffset[iPatch],
                                            self.neighborPatchOffset[iPatch] + self.patchSize[iPatch],
                                            device=device)
                cyclic_faces.append(iFace)
                neighbor_faces.append(neighborFace)
        if len(fixedValue_faces) > 0:
            fixedValue_faces = torch.cat(fixedValue_faces, dim=0)
        else:
            fixedValue_faces = None
        if len(zeroGradient_faces) > 0:
            zeroGradient_faces = torch.cat(zeroGradient_faces, dim=0)
        else:
            zeroGradient_faces = None
        if len(cyclic_faces) > 0:
            cyclic_faces = torch.cat(cyclic_faces, dim=0)
            neighbor_faces = torch.cat(neighbor_faces, dim=0)
            if fixedValue_faces is not None or zeroGradient_faces is not None:
                raise NotImplementedError('Mixed BC with cyclic BC is not supported yet.')
            self.all_cyclic = True
        else:
            cyclic_faces = None
            neighbor_faces = None
        return fixedValue_faces, zeroGradient_faces, cyclic_faces, neighbor_faces, \
            has_fixedValue_patch

    def correct_boundary_condition_U(self, U):
        """ Ref: correctBoundaryCondition <vector>
            Args:
                U (tensor, vector): [num_cells, 3]
            Returns:
                bouU (tensor, vector): [num_boundary_surfaces, 3]
        """
        bouU = torch.zeros((self.num_boundary_surfaces, U.shape[1]), dtype=U.dtype, device=U.device)
        for iPatch in range(self.num_patches):
            if self.patchTypeU[iPatch] == PatchType.FIXED_VALUE:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=U.device)
                bouU[iFace] = 0. # temp
            elif self.patchTypeU[iPatch] == PatchType.EMPTY:
                pass
            elif self.patchTypeU[iPatch] == PatchType.ZERO_GRADIENT:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=U.device)
                iCell = self.face2cell[iFace]
                bouU[iFace] = U[iCell]
            elif self.patchTypeU[iPatch] == PatchType.CYCLIC:
                iFace = torch.arange(self.patchOffset[iPatch],
                                     self.patchOffset[iPatch] + self.patchSize[iPatch],
                                     device=U.device)
                iCell = self.face2cell[iFace]
                neighborFace = torch.arange(self.neighborPatchOffset[iPatch],
                                            self.neighborPatchOffset[iPatch] + self.patchSize[iPatch],
                                            device=U.device)
                neighborCell = self.face2cell[neighborFace]
                weights = self.bouWeights[iFace]
                bouU[iFace] = weights * U[iCell] + (1 - weights) * U[neighborCell]
                bouU[neighborFace] = bouU[iFace]
            else:
                raise NotImplementedError(f'BCType {self.patchTypeU[iPatch]} - not implemented yet.')
        return bouU
    
    def correct_boundary_condition_P(self, P):
        """ Ref: correctBoundaryCondition <scalar>
            Args:
                P (tensor, scalar): [num_cells, 1]
            Returns:
                bouP (tensor, scalar): [num_boundary_surfaces, 1]
        """
        if self.boundaryP is not None:
            bouP = self.boundaryP.clone().to(P.dtype)
        else:
            bouP = torch.zeros((self.num_boundary_surfaces, P.shape[1]), dtype=P.dtype, device=P.device)
        # fixedValue
        if self.p_fixedValue_faces is not None:
            pass
            # bouP[self.p_fixedValue_faces] = 10.
        # zeroGradient
        if self.p_zeroGradient_faces is not None:
            iCell = self.face2cell[self.p_zeroGradient_faces]
            bouP[self.p_zeroGradient_faces] = P[iCell]
        # cyclic
        if self.p_cyclic_faces is not None:
            iCell = self.face2cell[self.p_cyclic_faces]
            neighborCell = self.face2cell[self.p_neighbor_faces]
            bouP[self.p_cyclic_faces] = self.bouWeights[self.p_cyclic_faces] * P[iCell] + \
                (1 - self.bouWeights[self.p_cyclic_faces]) * P[neighborCell]
        return bouP
    
    def updateBoundaryCoeffs_P(self, P, bouP):
        gradient_internal_coeffs = torch.zeros((self.num_boundary_surfaces, 1), dtype=P.dtype, device=P.device)
        gradient_boundary_coeffs = torch.zeros((self.num_boundary_surfaces, 1), dtype=P.dtype, device=P.device)
        # fixedValue
        if self.p_fixedValue_faces is not None:
            iFace = self.p_fixedValue_faces
            gradient_internal_coeffs[iFace] = -1.0 * self.bouDeltaCoeff[iFace].to(P.dtype)
            gradient_boundary_coeffs[iFace] = self.bouDeltaCoeff[iFace] * bouP[iFace]
        # zeroGradient
        if self.p_zeroGradient_faces is not None:
            iFace = self.p_zeroGradient_faces
            gradient_internal_coeffs[iFace] = 0
            gradient_boundary_coeffs[iFace] = 0
        # cyclic
        if self.p_cyclic_faces is not None:
            iFace = self.p_cyclic_faces
            gradient_internal_coeffs[iFace] = -1.0 * self.bouDeltaCoeff[iFace].to(P.dtype)
            gradient_boundary_coeffs[iFace] = self.bouDeltaCoeff[iFace].to(P.dtype)
        return gradient_internal_coeffs, gradient_boundary_coeffs
    
    def updateMatrixInterfaces_cyclic(self, psi, source, boundary_coeffs):
        """ Ref: updateMatrixInterfaces for cyclic BC
        """
        idx = self.face2cell  # [num_boundary_surfaces]
        expanded_idx = idx.unsqueeze(1).expand(-1, boundary_coeffs.shape[1])
        # cyclic
        if self.p_cyclic_faces is not None:
            idx = expanded_idx[self.p_cyclic_faces]
            bc = boundary_coeffs[self.p_cyclic_faces]
            neighborCell = self.face2cell[self.p_neighbor_faces]
            source = torch.scatter_add(source, 0, idx, -bc * psi[neighborCell])
        return source

    def fvm_laplacian_matrix(self, gradient_internal_coeffs, gradient_boundary_coeffs,
                             eqn, sign=1.):
        # internal source
        owner = self.lowerIndex
        neighbor = self.upperIndex
        
        lower_value = self.deltaCoeff * self.magSf * sign
        upper_value = lower_value   
        
        eqn.lower = eqn.lower + lower_value
        eqn.upper = eqn.upper + upper_value

        eqn.diag = eqn.diag.scatter_add(0, owner.unsqueeze(-1), -lower_value.to(eqn.diag.dtype))
        eqn.diag = eqn.diag.scatter_add(0, neighbor.unsqueeze(-1), -upper_value.to(eqn.diag.dtype))

        # boundary source, exclude processor BCs
        eqn.internal_coeffs = eqn.internal_coeffs + self.bouMagSf * gradient_internal_coeffs * sign
        eqn.boundary_coeffs = eqn.boundary_coeffs - self.bouMagSf * gradient_boundary_coeffs * sign

    def construct_Peqn(self, p, f, dtype):
        Peqn = eqnMat(self, self.num_surfaces, self.num_boundary_surfaces, self.num_cells,
                      self.face2cell, self.lowerIndex, self.upperIndex,
                      device=p.device, dim=1, dtype=dtype)
        Peqn.source = f * self.volume
        bouP = self.correct_boundary_condition_P(p)
        gradient_internal_coeffs, gradient_boundary_coeffs = self.updateBoundaryCoeffs_P(p, bouP)
        self.fvm_laplacian_matrix(gradient_internal_coeffs, gradient_boundary_coeffs, Peqn)
        for iMesh in range(self.num_meshes):
            if not self.p_has_fixedValue_patch[iMesh]:
                refCell = self.cell_offset[iMesh]
                self.set_reference_pressure(Peqn, refCell=refCell, refValue=0.0)
        return Peqn
    
    def set_reference_pressure(self, Peqn, refCell=0, refValue=0.0):
        """Set reference pressure and preserve symmetry (for PCG)."""

        # 清除与 refCell 相连的所有下、上对角耦合（行 + 列对称）
        mask_lower = (Peqn.lowerIndex == refCell)
        mask_upper = (Peqn.upperIndex == refCell)
        Peqn.lower[mask_lower] = 0.0
        Peqn.upper[mask_lower] = 0.0
        Peqn.lower[mask_upper] = 0.0
        Peqn.upper[mask_upper] = 0.0

        # 将 refCell 方程改为 p_ref = refValue
        Peqn.diag[refCell] = 1.0
        Peqn.source[refCell] = refValue

    def solve_possion_linear_system_ldu(self, p, f):
        Peqn = self.construct_Peqn(p, f, dtype=p.dtype)
        p, r_true_list = Peqn.solve_pcg(p, check_interval=1)
        # r = r_true_list[-1]
        # print(r.item())
        return p, r_true_list


class eqnMat:
    """
    Class to store the matrix of the equation.
    """
    def __init__(self, space_dis, num_surfaces, num_boundary_surfaces, num_cells, face2cell, lowerIndex,
                 upperIndex, device, dim=1, dtype=torch.float32):
        self.space_dis = space_dis
        self.num_surfaces = num_surfaces
        self.num_cells = num_cells
        self.num_boundary_surfaces = num_boundary_surfaces
        self.face2cell = face2cell
        self.lowerIndex = lowerIndex
        self.upperIndex = upperIndex
        self.device = device
        self.dim = dim
        
        self.lower = torch.zeros((self.num_surfaces, 1), device=device, dtype=dtype)
        self.upper = torch.zeros((self.num_surfaces, 1), device=device, dtype=dtype)
        self.diag = torch.zeros((self.num_cells, 1), device=device, dtype=dtype)
        self.source = torch.zeros((self.num_cells, dim), device=device, dtype=dtype)
        self.internal_coeffs = torch.zeros((self.num_boundary_surfaces, dim), device=device, dtype=dtype)
        self.boundary_coeffs = torch.zeros((self.num_boundary_surfaces, dim), device=device, dtype=dtype)
    
    def Amul(self, psi):
        saveDiag = self.diag
        idx = self.face2cell                # [num_boundary_surfaces]
        expanded_idx = idx.unsqueeze(1).expand(-1, self.dim)            # [num_boundary_surfaces, self.dim]        
        saveDiag = saveDiag.scatter_add(0, expanded_idx, self.internal_coeffs)    # addBoundaryDiag 
        
        # matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        Apsi = saveDiag * psi # (num_cells, dim)

        l = self.lowerIndex  # [num_surfaces]
        u = self.upperIndex  # [num_surfaces]
        
        contrib_u = self.lower * psi[l]  # [num_surfaces, 3]
        contrib_l = self.upper * psi[u]  # [num_surfaces, 3]
        
        Apsi = Apsi.scatter_add(0, u.unsqueeze(1).expand(-1, self.dim), contrib_u)
        Apsi = Apsi.scatter_add(0, l.unsqueeze(1).expand(-1, self.dim), contrib_l)
        # subtract boundary contributions
        if self.space_dis.all_cyclic:
            Apsi = self.space_dis.updateMatrixInterfaces_cyclic(psi, Apsi, self.boundary_coeffs)

        return Apsi
    
    @property
    def ADiag(self):
        idx = self.face2cell
        expanded_idx = idx.unsqueeze(1).expand(-1, self.dim)   
        diag = self.diag.scatter_add(0, expanded_idx, self.internal_coeffs)
        return diag
    
    @property
    def b(self):
        idx = self.face2cell
        expanded_idx = idx.unsqueeze(1).expand(-1, self.dim)
        source = self.source.clone()
        if not self.space_dis.all_cyclic:
            source = self.source.scatter_add(0, expanded_idx, self.boundary_coeffs)      # addBoundarySource
        return source

    def solve_jacobi(self, psi0, max_iter=10000, tol=3e-5):
        """
        Solve the linear system using the Jacobi method.
        """
        psi = psi0
        for i in range(max_iter):
            Apsi = self.Amul(psi)
            r = self.b - Apsi
            norm = torch.norm(r)
            if norm < tol:
                print(f'Converged in {i} iterations.')
                break
            update = r / self.ADiag
            psi = psi + update  # Jacobi update
        return psi

    def solve_cg(self, psi0, max_iter=3000, tol=1e-8):
        """
        Solve the linear system using the Conjugate Gradient method.
        """
        psi = psi0
        r = self.b - self.Amul(psi)
        p = r
        rsold = torch.sum(r * r)
        
        for i in range(max_iter):
            Ap = self.Amul(p)
            alpha = rsold / torch.sum(p * Ap)
            psi = psi + alpha * p
            r = r - alpha * Ap
            rsnew = torch.sum(r * r)
            if torch.sqrt(rsnew) < tol:
                print(f'Converged in {i} iterations.')
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        return psi

    def solve_pcg(self, psi0, max_iter=3000, tol=1e-8, check_interval=3000):
        """
        Solve the linear system using the Preconditioned Conjugate Gradient method.

        Parameters
        ----------
        psi0 : torch.Tensor
            Initial guess.
        M_inv : callable or tensor
            Preconditioner inverse. Can be a function M_inv(r) or a diagonal tensor for Jacobi preconditioner.
        """
        psi = psi0
        r = self.b - self.Amul(psi)
        A_diag = self.ADiag
        M_inv = 1.0 / A_diag  # Jacobi preconditioner
        
        # Apply preconditioner
        z = M_inv * r
        p = z.clone()

        rzold = torch.sum(r * z)

        r_true_list = []
        for i in range(max_iter):
            Ap = self.Amul(p)
            alpha = rzold / torch.sum(p * Ap)

            psi = psi + alpha * p
            r = r - alpha * Ap

            # Every check_interval steps, compute true residual
            if (i+1) % check_interval == 0 or i == max_iter - 1:
                r_true = self.b - self.Amul(psi)
                r_norm = torch.norm(r_true)
                r_true_list.append(r_norm)

            if torch.norm(r) < tol:
                print(f'Converged in {i} iterations with norm: {torch.norm(r):.3e}.')
                break

            # Apply preconditioner again
            z = M_inv * r
            rznew = torch.sum(r * z)

            beta = rznew / rzold
            p = z + beta * p
            rzold = rznew
        if len(r_true_list) == 0:
            r_true = self.b - self.Amul(psi)
            r_norm = torch.norm(r_true)
            r_true_list.append(r_norm)
        r_true_list = torch.stack(r_true_list)
        return psi, r_true_list
