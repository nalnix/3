"""
CurveletINR Architecture
Y->FDCT->coarse_y, wedge_y
       ->wedge_idx->wedge physical embedding->w_emb
Z->spectral attn->low pass filtering->coarse_z
                                    ->wedge z attn->wedge_z
coarse_y, coarse_z->coarseINR->coarse_x->
wedge_y, wedge_z, w_emb->wedgeINR->wedge_x->IFDCT->x
"""

# PREVIOUS EDITION (lkw):
# comparatively high SAM (leading to low PSNR, high ERGAS, high RMSE)
# high aggregation weight?(0.9-1.1)
# lissajous-like pattern in fusion image?

# NEW EDITION (pa):
# move spectral self attention before filtering
# type error detail_z=Z-coarse_z proven valid

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.module.fe_block import make_coord, MLP
from model.base_model import BaseModel, register_model

import matplotlib.pyplot as plt

def vis_weight(w:Tensor):
    weight=w.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(weight, aspect='auto', cmap='viridis',vmax=1)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Weight Value')

    # 设置坐标轴标签
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Sample / Neuron Index')
    ax.set_title('Weight Matrix Heatmap (24×5)')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./weight.png', dpi=150)
    plt.close()

def vis_spectrum(x: Tensor,name='x',is_wedge=False):
    if is_wedge:
        x=x.sum(dim=1)
        B,C,H,W = x.shape
        spec = torch.fft.fft2(x[0, 0]).abs().log1p().cpu()
    else:
        spec = torch.fft.fft2(x[0, 0]).abs().log1p().cpu()
    plt.figure()
    plt.imshow(torch.fft.fftshift(spec).numpy(),cmap='hot',vmax=7)
    plt.colorbar()
    plt.savefig(f'./{name}_spec.png', dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FDCT Implementation (Window Cache & Fast Processing)
# ─────────────────────────────────────────────────────────────────────────────

_WINDOW_CACHE: dict = {}

def _Meyer_window(x: Tensor) -> Tensor:
    x = x.clamp(0, 1)
    v = x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)
    return torch.cos(0.5 * math.pi * v)

def _phi(R: Tensor, r_cut: float) -> Tensor:
    return _Meyer_window((R - 0.5 * r_cut) / (0.5 * r_cut))

def _psi(R: Tensor, r_cut: float) -> Tensor:
    return _Meyer_window(1.0 - (R - 0.5 * r_cut) / (0.5 * r_cut))

def _to_sparse(win: Tensor, threshold: float = 1e-6) -> tuple[Tensor, Tensor]:
    idx = torch.nonzero(win.flatten() > threshold, as_tuple=False).flatten()
    val = win.flatten()[idx]
    return idx, val

def _compute_windows(H, W, num_scales, num_angles_coarse, device, dtype) -> list[list[dict]]:
    cache_key = (H, W, num_scales, num_angles_coarse, str(device), str(dtype))
    if cache_key in _WINDOW_CACHE: return _WINDOW_CACHE[cache_key]

    v = torch.fft.fftfreq(H, device=device)
    u = torch.fft.fftfreq(W, device=device)
    V, U = torch.meshgrid(v, u, indexing="ij")
    R = torch.sqrt(U ** 2 + V ** 2).clamp(min=1e-12)
    Theta = torch.atan2(V, U)

    r_bounds = [0.5 ** (num_scales - 1 - s) for s in range(num_scales)]
    sum_sq = torch.zeros((H, W), device=device, dtype=dtype)
    raw_windows = []

    r_cut = r_bounds[0]
    coarse_win = _phi(R, r_cut).to(dtype)
    sum_sq += coarse_win ** 2
    raw_windows.append([coarse_win])

    for s in range(1, num_scales):
        r_inner, r_outer = r_bounds[s - 1], r_bounds[s]
        radial_band = _psi(R, r_inner) * _phi(R, r_outer)

        num_angles = num_angles_coarse * (2 ** (s - 1))
        if num_angles % 2 != 0: num_angles += 1
        angle_step = math.pi / num_angles

        scale_raw = []
        for n in range(num_angles):
            theta_c = (n + 0.5) * angle_step
            d_up = ((Theta - theta_c) + math.pi) % (2 * math.pi) - math.pi
            d_dn = ((Theta - theta_c - math.pi) + math.pi) % (2 * math.pi) - math.pi
            d = torch.where(d_up.abs() < d_dn.abs(), d_up, d_dn)
            ang_win = _Meyer_window(d.abs() / angle_step)
            win = (radial_band * ang_win).to(dtype)

            sum_sq += win ** 2
            scale_raw.append(win)
        raw_windows.append(scale_raw)

    norm_factor = torch.sqrt(sum_sq + 1e-14)
    windows: list[list[dict]] = []

    for s, scale_raw in enumerate(raw_windows):
        scale_wins = []
        for win in scale_raw:
            win_norm = win / norm_factor
            idx, val = _to_sparse(win_norm)
            is_coarse = (s == 0)
            scale_wins.append({'idx': idx, 'val': val, 'decimation': (1, 1), 'is_coarse': is_coarse})
        windows.append(scale_wins)

    _WINDOW_CACHE[cache_key] = windows
    return windows

def fdct2d(x: Tensor, num_scales: int = 4, num_angles_coarse: int = 8) -> list[list[Tensor]]:
    B, C, H, W = x.shape
    X_fft = torch.fft.fft2(x.float(), norm="ortho")
    windows = _compute_windows(H, W, num_scales, num_angles_coarse, x.device, x.dtype)
    coeffs: list[list[Tensor]] = []
    for scale_wins in windows:
        scale_coeffs = []
        for win_info in scale_wins:
            idx, val = win_info['idx'], win_info['val'].to(X_fft.dtype)
            freq_band = torch.zeros_like(X_fft)
            freq_band.reshape(B, C, -1)[:, :, idx] = X_fft.reshape(B, C, -1)[:, :, idx] * val
            spatial = torch.fft.ifft2(freq_band, norm="ortho")
            coeff = spatial.real if win_info['is_coarse'] else spatial
            cdtype = x.dtype if win_info['is_coarse'] else (torch.complex64 if x.dtype == torch.float32 else torch.complex128)
            scale_coeffs.append(coeff.to(cdtype))
        coeffs.append(scale_coeffs)
    return coeffs

def ifdct2d(coeffs: list[list[Tensor]], output_shape: tuple[int, int, int, int]) -> Tensor:
    B, C, H, W = output_shape
    device, dtype = coeffs[0][0].device, coeffs[0][0].dtype
    num_scales = len(coeffs)
    num_angles_coarse = len(coeffs[1]) if num_scales > 1 else 8
    windows = _compute_windows(H, W, num_scales, num_angles_coarse, device, dtype if not dtype.is_complex else torch.float32)
    cdtype = torch.complex64 if (dtype == torch.float32 or dtype.is_complex) else torch.complex128
    X_rec = torch.zeros(B, C, H, W, dtype=cdtype, device=device)
    for scale_wins, scale_coeffs in zip(windows, coeffs):
        for win_info, coeff in zip(scale_wins, scale_coeffs):
            idx, val = win_info['idx'], win_info['val'].to(cdtype)
            up_fft = torch.fft.fft2(coeff.to(cdtype), norm="ortho")
            X_rec.reshape(B, C, -1)[:, :, idx] += up_fft.reshape(B, C, -1)[:, :, idx] * val

    x_rec = torch.fft.ifft2(X_rec, norm="ortho").real
    return x_rec.to(dtype if not dtype.is_complex else torch.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Stacking Utilities
# ─────────────────────────────────────────────────────────────────────────────

def stack_wedges(coeffs: list[list[Tensor]]):
    detail = coeffs[1:]
    wedge_list, wedge_index = [], []
    max_h = max(w.shape[-2] for scale in detail for w in scale)
    max_w = max(w.shape[-1] for scale in detail for w in scale)

    for s, scale_wedges in enumerate(detail):
        for n, wedge in enumerate(scale_wedges):
            B, C, L1, L2 = wedge.shape
            w_real = torch.view_as_real(wedge).permute(0, 4, 1, 2, 3).reshape(B, C * 2, L1, L2)
            if L1 != max_h or L2 != max_w:
                w_real = F.interpolate(w_real, size=(max_h, max_w), mode='bilinear', align_corners=False)
            wedge_list.append(w_real)
            wedge_index.append((s, n))
    
    return torch.stack(wedge_list, dim=2), wedge_index

def unstack_wedges(wedge_tensor: Tensor, coeffs_template: list[list[Tensor]], wedge_index: list):
    B, C2, Wn, H, W = wedge_tensor.shape
    C = C2 // 2
    new_coeffs = [coeffs_template[0]]
    for scale in coeffs_template[1:]:
        new_coeffs.append([None] * len(scale))

    for idx, (s, n) in enumerate(wedge_index):
        orig_size = coeffs_template[s + 1][n].shape[-2:]
        w_real = wedge_tensor[:, :, idx, :, :]
        if w_real.shape[-2:] != orig_size:
            w_real = F.interpolate(w_real, size=orig_size, mode='bilinear', align_corners=False)
        w_real_reshaped = w_real.view(B, 2, C, *orig_size).permute(0, 2, 3, 4, 1).contiguous()
        new_coeffs[s + 1][n] = torch.view_as_complex(w_real_reshaped)
    
    return new_coeffs

# ─────────────────────────────────────────────────────────────────────────────
# Model Modules 
# ─────────────────────────────────────────────────────────────────────────────

# ── Wedge neighbor utilities (shared by WedgeZAttention and WedgeINR) ────────
@staticmethod
def _angles_at_scale(num_angles_coarse: int, s: int) -> int:
    return num_angles_coarse * (2 ** s)

# def _wedge_clamp_s(s, num_scales):
#     return max(0, min(s, num_scales - 1))

def _wrap_n(s, n, num_angles_coarse):
    return n % _angles_at_scale(num_angles_coarse, s)

# def _wedge_resolve(s, n, num_scales, num_angles_coarse):
#     s2 = _wedge_clamp_s(s, num_scales)
#     return s2, _wedge_wrap_n(s2, n, num_angles_coarse)

_ANGULAR_NEIGHBOR_CACHE: dict = {}   # for WedgeZAttention

def build_neighbor_indices(wedge_index, num_scales, num_angles_coarse, device):
    """
    Build and cache both neighbor index matrices for a given wedge_index.
    Returns angular_idx, [Wn, 5].

    Angular neighbor order (for WedgeZAttention):
      0: self          (s,   n)
      1: angular left  (s,   n-1)
      2: angular right (s,   n+1)
      3: child         (s+1, 2n)
      4: parent        (s-1, n//2)
    """
    key = tuple(wedge_index)
    if key not in _ANGULAR_NEIGHBOR_CACHE:
        wi_map = {wn: i for i, wn in enumerate(wedge_index)}
        Wn = len(wedge_index)
        ang = torch.zeros(Wn, 5, dtype=torch.long)
        for i, (s, n) in enumerate(wedge_index):
            if s==0:
                ang_nb = [(s,n),(s,_wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s+1,2*n),(s,n)]
            elif s==num_scales-1:
                ang_nb = [(s,n),(s, _wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s,n),(s-1, n // 2)]
            else:
                ang_nb = [(s,n),(s, _wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s + 1, 2 * n),(s - 1, n // 2)]

            for k, wn in enumerate(ang_nb): ang[i, k] = wi_map.get(wn, i)

        _ANGULAR_NEIGHBOR_CACHE[key] = ang.to(device)


    return _ANGULAR_NEIGHBOR_CACHE[key]

@staticmethod
def weighted_agg(feat, idx, w, scale):
    """
    Neighbor-weighted aggregation.
      feat : [B, Wn, hw, D]
      idx  : [Wn, K]
      w    : [Wn, K]  raw logits -> softmax inside
      scale: [K]      per neighbor scaling factor
    returns: [B, Wn, hw, D]
    """
    B, Wn, hw, D = feat.shape
    K = idx.shape[1]
    nb = feat[:, idx.reshape(-1), :, :]   # [B, Wn*K, hw, D]
    nb = nb.reshape(B, Wn, K, hw, D)
    self_nb = nb[:,:,0]                   # [B, Wn, hw, D]
    # w  = w.softmax(dim=-1)                # [Wn, K]
    w = scale * torch.sigmoid(w)         #range from 0 to scale(initialized as 0.5*0.01)
    # vis_weight(w)

    w = w[None, :, :, None, None]        # [1, Wn, K, 1, 1]
    return (nb * w).sum(dim=2)+self_nb   # [B, Wn, hw, D]

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=2, max_len=4096):
        super().__init__()
        self.d_model = d_model
    def forward(self, x):
        if self.d_model <= 0: return x
        pe_list = [x]
        for k in range(1, self.d_model + 1):
            pe_list.append(torch.sin(math.pi * k * x))
            pe_list.append(torch.cos(math.pi * k * x))
        return torch.cat(pe_list, dim=-1)

class WedgePhysicalEmbedding(nn.Module):
    def __init__(self, num_scales: int, num_angles_coarse: int, embed_dim: int = 16, scale_embed_dim: int = 8):
        super().__init__()
        self.num_angles_coarse = num_angles_coarse
        self.scale_embed = nn.Embedding(num_scales, scale_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(scale_embed_dim + 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    def forward(self, wedge_index: list, device) -> Tensor:
        s_list = [s for s, _ in wedge_index]
        n_list = [n for _, n in wedge_index]
        ns_list = [self.num_angles_coarse * (2 ** s) for s in s_list]
        theta = torch.tensor([2.0 * math.pi * (n + 0.5) / ns for n, ns in zip(n_list, ns_list)],
                             dtype=torch.float32, device=device)
        sin_cos = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        s_ids = torch.tensor(s_list, dtype=torch.long, device=device)
        s_emb = self.scale_embed(s_ids)
        feat = torch.cat([sin_cos, s_emb], dim=-1)
        w_emb = self.proj(feat) # [Wn, embed_dim]
        return w_emb

class WedgeZAttention(nn.Module):
    """
    wedge_z_n = CrossAttn(Q=z, K=K_agg(n), V=V_agg(n))

    wedge_y is HR [B, Cy_ri, Wn, H, W]; z is LR [B, Cz, h, w].
    wedge_y is downsampled to (h, w) before attention so that Q and K/V
    share the same spatial resolution.  Output wedge_z is LR [B, Cz*2, Wn, h, w].

    K_agg / V_agg: learned weighted sum over N(n) ∪ {n}  (weights from w_emb)
    A/phi heads share Q and K; only V is split.
    """

    def __init__(self,
                 Cz: int,
                 Cy_ri: int,
                 embed_dim: int,
                 num_scales: int,
                 num_angles_coarse: int,
                 nb_K: int = 5,
                 d_head: int = 64):
        super().__init__()
        self.Cz = Cz
        self.Cy_ri = Cy_ri
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.num_angles_coarse = num_angles_coarse
        self.nb_K = nb_K
        self.d_head = d_head

        # neighbor aggregation weights (shared for K and V)
        self.base_scale = nn.Parameter(torch.full((5,), 0.01))
        self.scale_mod = nn.Linear(1, 5, bias=False)
        nn.init.constant_(self.scale_mod.weight, 0.0)
        
        # self.nb_w = nn.Linear(embed_dim, nb_K)
        self.nb_w = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)
        )
        nn.init.zeros_(self.nb_w[-1].weight)
        nn.init.zeros_(self.nb_w[-1].bias)

        # Q: from z (LR)
        self.Wq = nn.Linear(Cz, d_head)

        # K: from aggregated [wedge_y_ds, w_emb]
        self.Wk_A = nn.Linear(Cy_ri + embed_dim, d_head)
        # self.Wk_A   = nn.Linear(Cy_ri , d_head)
        self.Wk_phi = nn.Linear(Cy_ri + embed_dim, d_head)

        nn.init.constant_(self.Wk_A.bias,   -2.0)   # sigmoid(-2) ≈ 0.12
        nn.init.constant_(self.Wk_phi.bias, -2.0)

        # V: two heads (A and phi), from aggregated [wedge_y_ds, z]
        self.Wv_A   = nn.Linear(Cy_ri + Cz, Cz)
        self.Wv_phi = nn.Linear(Cy_ri + Cz, Cz)

    def forward(self,
                wedge_y: torch.Tensor,   # [B, Cy_ri, Wn, H, W]  HR
                z:       torch.Tensor,   # [B, Cz,          h, w] LR
                w_emb:   torch.Tensor,   # [Wn, embed_dim]
                wedge_index: list,
                ) -> torch.Tensor:       # [B, Cz*2, Wn, h, w]   LR

        B, Cy_ri, Wn, H, W = wedge_y.shape
        _, Cz, h, w = z.shape
        d = self.d_head
        device = z.device
        ds_ratio = H / h

        nb_idx = build_neighbor_indices(
            wedge_index, self.num_scales, self.num_angles_coarse, device) # [Wn, nb_K]

        # --- downsample wedge_y from (H,W) to (h,w) ---
        # reshape to 4D for interpolate, then restore Wn dim
        wy_ds = F.adaptive_avg_pool2d(
            wedge_y.reshape(B * Wn, Cy_ri, H, W), 
            output_size=(h, w)
            ).reshape(B, Wn, Cy_ri, h, w)

        # wy_ds = F.interpolate(
        #     wedge_y.reshape(B * Wn, Cy_ri, H, W),
        #     size=(h, w), mode='bilinear', align_corners=False
        # ).reshape(B, Wn, Cy_ri, h, w)              # [B, Wn, Cy_ri, h, w]

        # layout: [B, Wn, hw, D]
        wy  = wy_ds.permute(0, 1, 3, 4, 2).reshape(B, Wn, h * w, Cy_ri)
        fe  = w_emb[None, :, None, :].expand(B, Wn, h * w, -1)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, h * w, Cz)          # [B, hw, Cz]
        z_wn   = z_flat[:, None, :, :].expand(B, Wn, h * w, Cz)       # [B, Wn, hw, Cz]

        # ================================================================
        # Step 1: neighbor weighted aggregation -> K_agg, V_agg
        # ================================================================
        # nb_w = self.nb_w(w_emb)                                        # [Wn, nb_K]
        nb_emb   = w_emb[nb_idx.reshape(-1)].reshape(Wn, self.nb_K, -1)   # [Wn, K, emb]
        self_emb = w_emb[:, None, :].expand(Wn, self.nb_K, -1)             # [Wn, K, emb]
        pair     = torch.cat([self_emb - nb_emb, self_emb], dim=-1)         # [Wn, K, emb*2]
        nb_w     = self.nb_w(pair).squeeze(-1)                              # [Wn, K]
        
        # scale=0.01
        ds_tensor = torch.tensor([math.log2(ds_ratio)], device=device, dtype=torch.float32)
        # 用 tanh 限制调制范围在 [-1,1]，然后映射到 [0,1]
        mod = torch.tanh(self.scale_mod(ds_tensor)) * 0.5 + 0.5   # [1, 5]
        scale = self.base_scale * mod.squeeze(0)                  # [5]

        k_in  = torch.cat([wy, fe], dim=-1)                            # [B, Wn, hw, Cy_ri+emb]
        K_agg = weighted_agg(k_in, nb_idx, nb_w, scale)                        # [B, Wn, hw, Cy_ri+emb]

        v_in  = torch.cat([wy, z_wn], dim=-1)                          # [B, Wn, hw, Cy_ri+Cz]
        V_agg = weighted_agg(v_in, nb_idx, nb_w, scale)                       # [B, Wn, hw, Cy_ri+Cz]

        # ================================================================
        # Step 2: project -> Q, K, Va, Vp
        # ================================================================
        Q  = self.Wq(z_flat)        # [B, hw, d]
        # Ka   = self.Wk_A(K_agg[:, :, :, :Cy_ri])     # [B, Wn, hw, d]
        Ka   = self.Wk_A(K_agg)     # [B, Wn, hw, d]
        Kp = self.Wk_phi(K_agg)   # [B, Wn, hw, d]
        Va = self.Wv_A(V_agg)       # [B, Wn, hw, Cz]
        Vp = self.Wv_phi(V_agg)     # [B, Wn, hw, Cz]

        # ================================================================
        # Step 3: attention score (shared Q & K, split only at V)
        # softmax over Wn: "which wedge matters for this LR pixel"
        # ================================================================
        Q_exp = Q[:, None, :, :]                                       # [B, 1,  hw, d]
        score_A = (Q_exp * Ka).sum(-1) / (d ** 0.5)                   # [B, Wn, hw]
        score_p = (Q_exp * Kp).sum(-1) / (d ** 0.5)                   # [B, Wn, hw]
        attn_A  = torch.sigmoid(score_A)                               # [B, Wn, hw]
        attn_p  = torch.sigmoid(score_p)                               # [B, Wn, hw]

        # ================================================================
        # Step 4: amplitude-phase modulation on z
        # ================================================================
        out_a = attn_A[:, :, :, None] * Va                            # [B, Wn, hw, Cz]
        out_p = attn_p[:, :, :, None] * Vp                            # [B, Wn, hw, Cz]                                     # [B, Wn, hw, Cz]

        A   = F.softplus(out_a)                                        # > 0
        # phi = out_p
        phi=torch.pi*torch.tanh(out_p)                             # [-pi, pi]

        re = A * z_wn * torch.cos(phi)                                 # [B, Wn, hw, Cz]
        im = A * z_wn * torch.sin(phi)

        wedge_z = torch.cat([re, im], dim=-1)                          # [B, Wn, hw, Cz*2]
        wedge_z = wedge_z.permute(0, 3, 1, 2)                         # [B, Cz*2, Wn, hw]
        wedge_z = wedge_z.reshape(B, Cz * 2, Wn, h, w)
        return wedge_z

# --- 新增：4D 光谱自注意力机制 (用于 Z 提取 z_guide) ---
class SpectralAttention4D(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1) # [B, C, H, W]
        
        q = q.view(B, C, -1) # [B, C, HW]
        k = k.view(B, C, -1)
        v = v.view(B, C, -1)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 光谱通道间的注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) # [B, C, C]
        
        out = (attn @ v).view(B, C, H, W)
        return x + self.proj(out)

class LearnableLowPassFilter(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.padding = kernel_size // 2
        # 使用 groups=channels 确保只在空间域做滤波，不破坏原本的光谱通道独立性
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                              padding=self.padding, groups=channels, bias=False)
        
        # --- 赋予高斯低通初始先验 ---
        sigma = kernel_size / 4.0  # 经验值，让高斯分布覆盖整个 kernel
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum() # 归一化
        kernel2d = g.view(-1, 1) * g.view(1, -1)
        
        # 扩展到所有通道: [channels, 1, kernel_size, kernel_size]
        kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        
        # 将生成的权重赋给卷积层（它本身 requires_grad=True，因此是可学习的）
        with torch.no_grad():
            self.conv.weight.data.copy_(kernel2d)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class WedgeINR(nn.Module):
    def __init__(self, z_bands: int, y_bands: int, mlp_hidden: list = None,
                 pe_d_model: int = 2, embed_dim: int = 16, nb_K: int = 5,
                 num_scales: int = 4, num_angles_coarse: int = 8):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.num_scales = num_scales
        self.num_angles_coarse = num_angles_coarse
        self.pe = PositionalEmbedding(pe_d_model)
        coord_dim = 2 * (2 * pe_d_model + 1)
        # inputs: wedge_z (LR, Cz*2) + wedge_y_agg (HR, Cy_ri) + rel_coord
        # wedge_y_agg = q_y_self + neighbor-weighted q_y (same dim as q_y)
        in_dim  = z_bands + y_bands + coord_dim
        out_dim = z_bands + 1       # z_bands = Cz*2; +1 for mixture weight
        self.mlp = MLP(in_dim, out_dim, mlp_hidden)

        # # INR neighbor weights: project w_emb -> [Wn, nb_K] logits
        # self.inr_nb_w = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 2, nb_K),
        # )
        # nn.init.zeros_(self.inr_nb_w[-1].weight)
        # nn.init.zeros_(self.inr_nb_w[-1].bias)

    def forward(self, wedge_z: Tensor, wedge_y: Tensor,
                wedge_index: list, w_emb: Tensor) -> Tensor:
        """
        wedge_z     : [B, Cz_ri, Wn, h, w]  LR  (Cz_ri = Cz*2)
        wedge_y     : [B, Cy_ri, Wn, H, W]  HR
        wedge_index : list of (s, n) tuples
        w_emb       : [Wn, embed_dim]        wedge physical embeddings
        returns     : [B, Cz_ri, Wn, H, W]  HR
        """
        B, C_z, Wn, h, w = wedge_z.shape
        _, C_y, _,  H, W = wedge_y.shape
        N = H * W
        device = wedge_z.device

        coord      = make_coord((H, W)).unsqueeze(0).expand(B * Wn, -1, -1).to(device)
        feat_coord = make_coord((h, w), flatten=False).permute(2, 0, 1)\
                         .unsqueeze(0).expand(B * Wn, -1, -1, -1).to(device)
        rx, ry = 1.0 / h, 1.0 / w

        z_flat = wedge_z.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_z, h, w)
        y_flat = wedge_y.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_y, H, W)

        # ── q_y: HR guide feature at each HR pixel ────────────────────────
        grid_y = coord.clone().flip(-1).unsqueeze(1)
        q_y = F.grid_sample(y_flat, grid_y, mode='nearest', align_corners=False)\
                  [:, :, 0, :].permute(0, 2, 1)                        # [B*Wn, N, C_y]

        # # ── neighbor-aggregated q_y_agg (FDCT physical prior) ─────────────
        # # weighted_agg expects [B, Wn, hw, D]; reshape in/out accordingly
        # _, nb_idx = build_neighbor_indices(
        #     wedge_index, self.num_scales, self.num_angles_coarse, device)
        # nb_w    = self.inr_nb_w(w_emb)                                 # [Wn, K]
        # # print(nb_w.softmax(dim=-1))
        # q_y_4d  = q_y.reshape(B, Wn, N, C_y)                          # [B, Wn, N, C_y]
        # q_y_nb  = weighted_agg(q_y_4d, nb_idx, nb_w)                  # [B, Wn, N, C_y]
        # q_y_agg = (q_y_4d + q_y_nb).reshape(B * Wn, N, C_y)          # residual add

        # ── 4-corner LFC mixture (unchanged) ─────────────────────────────
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                grid_z  = coord_.flip(-1).unsqueeze(1)
                q_z     = F.grid_sample(z_flat, grid_z, mode='nearest', align_corners=False)\
                              [:, :, 0, :].permute(0, 2, 1)            # [B*Wn, N, C_z]
                q_coord = F.grid_sample(feat_coord, grid_z, mode='nearest', align_corners=False)\
                              [:, :, 0, :].permute(0, 2, 1)
                rel_coord = self.pe((coord - q_coord) * torch.tensor([h, w], device=device))
                inp  = torch.cat([q_z, q_y, rel_coord], dim=-1) 
                pred = self.mlp(inp.view(B * Wn * N, -1)).view(B * Wn, N, -1)
                preds.append(pred)

        preds  = torch.stack(preds, dim=-1)                            # [B*Wn, N, out, 4]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        out    = (preds[:, :, :-1, :] * weight.unsqueeze(-2)).sum(-1)  # [B*Wn, N, C_z]

        return out.view(B, Wn, H, W, C_z).permute(0, 4, 1, 2, 3)      # [B, C_z, Wn, H, W]

class CoarseINR(nn.Module):
    # 针对 3x3 local ensemble, bilinear 采样以及无 PE 版本的插值和融合
    def __init__(self, z_bands: int, y_bands: int, mlp_hidden: list = None, pe_d_model: int = 0):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        
        # pe_d_model 改为了 0，不再使用 Positional Embedding
        # 此时输入的坐标只有相对 x 和 y，因此维度恒为 2
        coord_dim = 2 
        in_dim  = z_bands + y_bands + coord_dim 
        out_dim = z_bands + 1
        
        self.mlp = MLP(in_dim, out_dim, mlp_hidden)
 
    def forward(self, coarse_y: Tensor, coarse_z: Tensor) -> Tensor:
        B, C_z, h, w = coarse_z.shape
        _, C_y, H, W = coarse_y.shape
        N = H * W
        device = coarse_z.device
 
        coord = make_coord((H, W)).unsqueeze(0).expand(B, -1, -1).to(device)
        feat_coord = make_coord((h, w), flatten=False).permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1).to(device)
        
        # 对于 3x3 邻域，偏移量应该是一整个特征图像素的距离
        # 在 [-1, 1] 坐标系下，一个像素的宽度是 2.0 / size
        rx, ry = 2.0 / h, 2.0 / w
 
        grid_y = coord.clone().flip(-1).unsqueeze(1)
        # grid_sample 模式改为 bilinear
        q_y = F.grid_sample(coarse_y, grid_y, mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
 
        preds = []
        # 改为 3x3 local ensemble (9 个邻域)
        for vx in [-1, 0, 1]:
            for vy in [-1, 0, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                grid_z  = coord_.flip(-1).unsqueeze(1)
                
                # grid_sample 模式改为 bilinear
                q_z     = F.grid_sample(coarse_z, grid_z, mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, grid_z, mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                
                # 取消了 self.pe，直接计算相对坐标并保持特征拼接
                rel_coord = (coord - q_coord) * torch.tensor([h, w], device=device)
                
                inp  = torch.cat([q_z, q_y, rel_coord], dim=-1)
                pred = self.mlp(inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)
 
        # preds 经过 stack 之后，最后一个维度长度为 9
        preds  = torch.stack(preds, dim=-1)
        # 对 9 个预测结果的权重进行 softmax 归一化
        weight = F.softmax(preds[:, :, -1, :], dim=-1)

        # 加权融合
        out    = (preds[:, :, :-1, :] * weight.unsqueeze(-2)).sum(-1)
        return out.view(B, H, W, C_z).permute(0, 3, 1, 2)

# class LowFreqFiLM(nn.Module):
#     def __init__(self, z_bands: int, y_bands: int, hidden: int = 128):
#         super().__init__()
#         self.net = nn.Sequential(nn.Conv2d(z_bands + y_bands, hidden, 3, 1, 1), nn.ReLU(inplace=True),
#                                  nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True))
#         self.gamma_head = nn.Conv2d(hidden, z_bands, 1)
#         self.beta_head = nn.Conv2d(hidden, z_bands, 1)

#     def forward(self, coarse_y: Tensor, coarse_z_up: Tensor):
#         feat = self.net(torch.cat([coarse_y, coarse_z_up], dim=1))
#         gamma = self.gamma_head(feat) + 1.0
#         beta = self.beta_head(feat)
#         return gamma * coarse_z_up + beta

# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
@register_model('b')
class CuINR(BaseModel):
    def __init__(self, hsi_dim: int = 31, msi_dim: int = 4, num_scales: int = 4,
                 num_angles_coarse: int = 8, mlp_hidden: list = None,
                 pe_d_model: int = 2, scale: int = 4, wedge_embed_dim: int = 16,
                 attn_d_head: int = 64,attn_num_heads: int = 4,film_hidden: int = 128):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.hsi_dim, self.msi_dim = hsi_dim, msi_dim
        self.num_scales, self.num_angles_coarse, self.scale = num_scales, num_angles_coarse, scale

        Cz_ri, Cy_ri = hsi_dim * 2, msi_dim * 2

        self.z_lpf=LearnableLowPassFilter(channels=hsi_dim,kernel_size=7)

        # 1. Wedge physical embedding
        self.wedge_embed = WedgePhysicalEmbedding(num_scales, num_angles_coarse,
                                                   embed_dim=wedge_embed_dim)

        # 2. Spectral self-attention on Z (4D, LR)
        self.z_spectral_attn = SpectralAttention4D(hsi_dim)

        # 3. wedge_z attention: cross-attend z (LR) to wedge_y (HR) -> wedge_z (LR)
        self.wedge_z_attn = WedgeZAttention(
            Cz=hsi_dim, Cy_ri=Cy_ri, embed_dim=wedge_embed_dim,
            num_scales=num_scales, num_angles_coarse=num_angles_coarse,
            d_head=attn_d_head,
        )

        # 4. High-frequency Wedge INR: wedge_z (LR) + wedge_y (HR) -> wedge_x (HR)
        self.wedge_inr = WedgeINR(z_bands=Cz_ri, y_bands=Cy_ri, mlp_hidden=mlp_hidden,
                                   pe_d_model=pe_d_model, embed_dim=wedge_embed_dim,
                                   num_scales=num_scales, num_angles_coarse=num_angles_coarse)

        # 5. Coarse INR：coarse_y (LR) + coarse_z (LR) -> coarse_x (LR)
        self.coarse_inr = CoarseINR(z_bands=hsi_dim, y_bands=msi_dim, mlp_hidden=mlp_hidden, pe_d_model=0)
        # # 5. Low-frequency FiLM
        # self.low_freq_film = LowFreqFiLM(hsi_dim, msi_dim, film_hidden)

    def _fdct(self, x): return fdct2d(x, self.num_scales, self.num_angles_coarse)
    def _ifdct(self, coeffs, size): return ifdct2d(coeffs, size)

    def _forward_implem(self, Y: Tensor, lms: Tensor, Z: Tensor) -> Tensor:
        B, Cy, H, W = Y.shape
        _, Cz, h, w = Z.shape

        # 1. MSI decomposition
        CY = self._fdct(Y)
        coarse_y = CY[0][0]                                   # [B, Cy, H, W]
        wedge_y, wedge_index = stack_wedges(CY)               # [B, Cy_ri, Wn, H, W]

        # 2. HSI low pass filtering
        z_guide=self.z_spectral_attn(Z)
        coarse_z=self.z_lpf(z_guide) #coarse[B,Cz,h,w]
        detail_z=Z-coarse_z    #detail[B,Cz,h,w]

        # 3. coarse fusion
        # z_up = F.interpolate(z_guide, size=(H, W), mode='bilinear', align_corners=False)
        coarse_x = self.coarse_inr(coarse_y, coarse_z)         # [B, Cz, H, W]

        # 4. build HSI wedge information through WedgeZAttention
        w_emb   = self.wedge_embed(wedge_index, device=Y.device)  # [Wn, embed_dim]
        wedge_z = self.wedge_z_attn(wedge_y, detail_z, w_emb, wedge_index)
        # wedge_z: [B, Cz_ri, Wn, h, w]

        # 4. wedge fusion
        wedge_x = self.wedge_inr(wedge_z, wedge_y, wedge_index, w_emb)
        
        # 5. fused image inverse transform
        CX_detail = unstack_wedges(wedge_x, CY, wedge_index)
        CX_detail[0] = [coarse_x.to(torch.complex64)]
        X=self._ifdct(CX_detail, (B, Cz, H, W))

        return X

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(pan, lms, ms)
        loss = criterion(sr, gt)
        return sr.clamp(0, 1), loss

    def val_step(self, ms, lms, pan):
        pred = self._forward_implem(pan, lms, ms)
        return pred.clamp(0, 1)

    def debug_step(self, ms, lms, pan, gt):
        pred = self._forward_implem(pan, lms, ms)
        vis_spectrum(gt)
        CP = self._fdct(pred)
        CG = self._fdct(gt)

        loss=torch.nn.L1Loss()
        coarse_loss = loss(CP[0][0], CG[0][0])
        print(f"Coarse loss: {coarse_loss.item():.5f}")
        wedge_pred, wedge_index = stack_wedges(CP)
        wedge_gt, _   = stack_wedges(CG)
        # # for s in range(self.num_scales):
        # #     mask = [i for i, (sc, _) in enumerate(wedge_index) if sc == s]
        # #     loss_s = loss(wedge_pred[:, :, mask], wedge_gt[:, :, mask])
        # #     print(f"scale {s}: {loss_s.item():.5f}")
        wedge_loss = loss(wedge_pred, wedge_gt)
        print(f"Wedge loss: {wedge_loss.item():.5f}")
        B,C,H,W = pred.shape
        loss_re = loss(wedge_pred[:, :C,:,:], wedge_gt[:, :C,:,:])
        loss_im = loss(wedge_pred[:, C:,:,:], wedge_gt[:, C:,:,:])
        print(f"Wedge loss (re/im): {loss_re.item():.5f} / {loss_im.item():.5f}")


        return pred.clamp(0, 1)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, Cz, Cy, H, W, scale = 2, 31, 3, 32, 32, 4
    h, w = H // scale, W // scale
    model = CuINR(hsi_dim=Cz, msi_dim=Cy, num_scales=3, num_angles_coarse=4, scale=scale).to(device)

    Y   = torch.randn(B, Cy, H, W).to(device)
    lms = torch.randn(B, Cz, H, W).to(device)
    Z   = torch.randn(B, Cz, h, w).to(device)
    gt  = torch.randn(B, Cz, H, W).to(device)

    criterion = nn.L1Loss()
    sr, loss  = model.train_step(Z, lms, Y, gt, criterion)

    print(f"Output shape    : {sr.shape}")
    print(f"Train loss      : {loss.item():.4f}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
