import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class Physics_Attention_Irregular_Mesh(nn.Module):
    """
    Interior/Boundary 版本 Transolver-style Attention:

    - 输入:
        x_inter:    (B, N_i_max, C)
        x_bdry:     (B, N_b_max, C) or None
        mask_inter: (B, N_i_max) or None, True = 有效
        mask_bdry:  (B, N_b_max) or None, True = 有效
    - 输出:
        out_inter:  (B, N_i_max, C)   # 只更新 interior 节点
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        slice_num=[64, 16],
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 温度：保持和原版一致的广播形状
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        # 节点特征 -> head 维度
        self.in_project_x  = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        # interior / boundary 各自 slice 数
        self.slice_num_int = slice_num[0]
        self.slice_num_bd  = slice_num[1]

        # interior 用的 slice projection
        self.in_project_slice_int = nn.Linear(dim_head, self.slice_num_int)
        torch.nn.init.orthogonal_(self.in_project_slice_int.weight)

        # boundary 用的 slice projection
        self.in_project_slice_bd = nn.Linear(dim_head, self.slice_num_bd)
        torch.nn.init.orthogonal_(self.in_project_slice_bd.weight)

        # 共用一套 q/k/v：既用于 interior self-attn，也用于 inter←bdry cross-attn
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # 自注意力 / 交叉注意力的融合系数（可学习）
        self.alpha_self  = nn.Parameter(torch.tensor(1.0))
        self.alpha_cross = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_inter, x_bdry, mask_inter=None, mask_bdry=None):
        """
        只返回 interior 的输出 (B, N_i_max, C)
        """

        # ========= (1) interior / boundary 各自 slice ========= #
        z_int, w_int, z_bd, _ = self.fast_slice_group(
            x_inter, x_bdry,
            mask_int=mask_inter,
            mask_bd=mask_bdry
        )  # z_int: (B,H,G_i,D), w_int: (B,H,N_i,G_i), z_bd: (B,H,G_b,D)

        # ========= (2) token self-attn ========= #
        # cat interior and boundary token
        z_all = torch.cat([z_int, z_bd], dim=2)  # (B,H,G_i+G_b,D)
        # self-attention
        q_all = self.to_q(z_all)              # (B,H,G_i+G_b,D)
        k_all = self.to_k(z_all)
        v_all = self.to_v(z_all)

        dots_self = torch.matmul(q_all, k_all.transpose(-1, -2)) * self.scale  # (B,H,G_i+G_b,G_i+G_b)
        attn_self = self.softmax(dots_self)
        attn_self = self.dropout(attn_self)
        z_all_out = torch.matmul(attn_self, v_all)    # (B,H,G_i+G_b,D)


        # ========= (3) Deslice：只回到 interior 节点 ========= #
        z_int_out = z_all_out[:, :, :self.slice_num_int, :]  # 取出 interior token 的输出部分
        out_x_int = torch.einsum("bhgc,bhng->bhnc", z_int_out, w_int)  # (B,H,N_i,D)

        if mask_inter is not None:
            node_mask_i = mask_inter[:, None, :, None].float()
            out_x_int = out_x_int * node_mask_i

        out_x_int = rearrange(out_x_int, "b h n d -> b n (h d)")
        out_x_int = self.to_out(out_x_int)   # (B, N_i_max, C)

        return out_x_int
    
    def fast_slice_group(self, x_int, x_bd, mask_int, mask_bd):
        """
        x:    (B, N_max, C)
        mask: (B, N_max) or None
        返回:
          slice_token:   (B, H, G, D)
          slice_weights: (B, H, N_max, G)
        """
        B, N_int, _ = x_int.shape
        _, N_bd, _ = x_bd.shape
        H, D = self.heads, self.dim_head

        if mask_int is not None:
            node_mask_int = mask_int[:, None, :, None]    # B 1 N 1
            node_mask_f_int = node_mask_int.float()
            node_mask_bd = mask_bd[:, None, :, None]      # B 1 N 1
            node_mask_f_bd = node_mask_bd.float()

        x = torch.cat([x_int, x_bd], dim=1)  # (B, N_total, C)
        fx_mid = self.in_project_fx(x).reshape(B, N_int + N_bd, H, D) \
                                       .permute(0, 2, 1, 3).contiguous()   # B H N D
        x_mid  = self.in_project_x(x).reshape(B, N_int + N_bd, H, D) \
                                       .permute(0, 2, 1, 3).contiguous()   # B H N D

        # padding 节点 feature 清零
        fx_mid_int = fx_mid[:, :, :N_int, :]
        x_mid_int  = x_mid[:, :, :N_int, :]
        fx_mid_bd = fx_mid[:, :, N_int:, :]
        x_mid_bd  = x_mid[:, :, N_int:, :]
        if mask_int is not None:
            fx_mid_int = fx_mid_int * node_mask_f_int
            x_mid_int  = x_mid_int * node_mask_f_int
            fx_mid_bd = fx_mid_bd * node_mask_f_bd
            x_mid_bd  = x_mid_bd * node_mask_f_bd

        # B H N G
        slice_weights_int = self.softmax(self.in_project_slice_int(x_mid_int) / self.temperature)  # softmax over G_i
        slice_weights_bd  = self.softmax(self.in_project_slice_bd(x_mid_bd) / self.temperature)  # softmax over G_b
        if mask_int is not None:
            slice_weights_int = slice_weights_int * node_mask_f_int  # padding 权重清零
            slice_weights_bd  = slice_weights_bd * node_mask_f_bd
        slice_norm_int = slice_weights_int.sum(2)  # B H G_i
        slice_norm_bd = slice_weights_bd.sum(2)    # B H G_b
        slice_token_int = torch.einsum("bhnc,bhng->bhgc", fx_mid_int, slice_weights_int)
        slice_token_int = slice_token_int / (slice_norm_int[..., None] + 1e-5)  # B H G_i D
        slice_token_bd = torch.einsum("bhnc,bhng->bhgc", fx_mid_bd, slice_weights_bd)
        slice_token_bd = slice_token_bd / (slice_norm_bd[..., None] + 1e-5)  # B H G_b D

        return slice_token_int, slice_weights_int, slice_token_bd, slice_weights_bd

