"""
CurveletINR Architecture
1. Y  -> FDCT -> coarse_y [HR], wedge_y [B, Cy_ri, Wn, H, W]
2. Z  -> SpectralAttention4D -> z_guide [B, Cz, h, w]
3. coarse_y + z_guide -> CoarseINR -> coarse_x [B, Cz, H, W]
4. wedge_y + z_guide + w_emb -> WedgeZAttention -> wedge_z [B, Cz_ri, Wn, h, w]
5. wedge_z (LR) + aggregated wedge_y (HR) -> WedgeINR -> wedge_x [B, Cz_ri, Wn, H, W]
6. coarse_x + wedge_x -> IFDCT -> X_rec
"""
#edited

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.module.fe_block import make_coord, MLP
from model.base_model import BaseModel, register_model

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
_CHILD_NEIGHBOR_CACHE:   dict = {}   # for WedgeINR

def build_neighbor_indices(wedge_index, num_scales, num_angles_coarse, device):
    """
    Build and cache both neighbor index matrices for a given wedge_index.
    Returns (angular_idx, child_idx), each [Wn, 5].

    Angular neighbor order (for WedgeZAttention):
      0: self          (s,   n)
      1: angular left  (s,   n-1)
      2: angular right (s,   n+1)
      3: child         (s+1, 2n)
      4: parent        (s-1, n//2)

    Child neighbor order (for WedgeINR):
      0: self          (s,   n)
      1: child left    (s+1, wrap(2n-1))
      2: child right   (s+1, wrap(2n+1))
      3: child center  (s+1, 2n)
      4: parent        (s-1, n//2)
    """
    key = tuple(wedge_index)
    if key not in _ANGULAR_NEIGHBOR_CACHE:
        wi_map = {wn: i for i, wn in enumerate(wedge_index)}
        Wn = len(wedge_index)
        ang = torch.zeros(Wn, 5, dtype=torch.long)
        cld = torch.zeros(Wn, 5, dtype=torch.long)
        for i, (s, n) in enumerate(wedge_index):
            if s==0:
                ang_nb = [(s,n),(s,_wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s+1,2*n),(s,n)]
                cld_nb = [(s,n),(s+1, _wrap_n(s + 1, 2 * n - 1, num_angles_coarse)),(s + 1, _wrap_n(s + 1, 2 * n + 1, num_angles_coarse)),(s + 1, 2 * n),(s,n)]
            elif s==num_scales-1:
                ang_nb = [(s,n),(s, _wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s,n),(s-1, n // 2)]
                cld_nb = [(s,n),(s,n),(s,n),(s,n),(s-1, n // 2)]
            else:
                ang_nb = [(s,n),(s, _wrap_n(s, n - 1, num_angles_coarse)),(s, _wrap_n(s, n + 1, num_angles_coarse)),(s + 1, 2 * n),(s - 1, n // 2)]
                cld_nb = [(s,n),(s + 1, _wrap_n(s + 1, 2 * n - 1, num_angles_coarse)),(s + 1, _wrap_n(s + 1, 2 * n + 1, num_angles_coarse)),(s + 1, 2 * n),(s - 1, n // 2)]

            for k, wn in enumerate(ang_nb): ang[i, k] = wi_map.get(wn, i)
            for k, wn in enumerate(cld_nb): cld[i, k] = wi_map.get(wn, i)
        _ANGULAR_NEIGHBOR_CACHE[key] = ang.to(device)
        _CHILD_NEIGHBOR_CACHE[key]   = cld.to(device)

    return _ANGULAR_NEIGHBOR_CACHE[key], _CHILD_NEIGHBOR_CACHE[key]

@staticmethod
def weighted_agg(feat, idx, w):
    """
    Neighbor-weighted aggregation.
      feat : [B, Wn, hw, D]
      idx  : [Wn, K]
      w    : [Wn, K]  raw logits -> softmax inside
    returns: [B, Wn, hw, D]
    """
    B, Wn, hw, D = feat.shape
    K = idx.shape[1]
    nb = feat[:, idx.reshape(-1), :, :]   # [B, Wn*K, hw, D]
    nb = nb.reshape(B, Wn, K, hw, D)
    w  = w.softmax(dim=-1)                # [Wn, K]
    w  = w[None, :, :, None, None]        # [1, Wn, K, 1, 1]
    return (nb * w).sum(dim=2)            # [B, Wn, hw, D]

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
        # self.nb_w = nn.Linear(embed_dim, nb_K)
        self.nb_w = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, nb_K)
        )
        nn.init.zeros_(self.nb_w[-1].weight)
        nn.init.zeros_(self.nb_w[-1].bias)

        # Q: from z (LR)
        self.Wq = nn.Linear(Cz, d_head)

        # K: from aggregated [wedge_y_ds, w_emb]
        self.Wk = nn.Linear(Cy_ri + embed_dim, d_head)

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

        nb_idx, _ = build_neighbor_indices(
            wedge_index, self.num_scales, self.num_angles_coarse, device) # [Wn, nb_K]

        # --- downsample wedge_y from (H,W) to (h,w) ---
        # reshape to 4D for interpolate, then restore Wn dim
        wy_ds = F.interpolate(
            wedge_y.reshape(B * Wn, Cy_ri, H, W),
            size=(h, w), mode='bilinear', align_corners=False
        ).reshape(B, Wn, Cy_ri, h, w)              # [B, Wn, Cy_ri, h, w]

        # layout: [B, Wn, hw, D]
        wy  = wy_ds.permute(0, 1, 3, 4, 2).reshape(B, Wn, h * w, Cy_ri)
        fe  = w_emb[None, :, None, :].expand(B, Wn, h * w, -1)
        z_flat = z.permute(0, 2, 3, 1).reshape(B, h * w, Cz)          # [B, hw, Cz]
        z_wn   = z_flat[:, None, :, :].expand(B, Wn, h * w, Cz)       # [B, Wn, hw, Cz]

        # ================================================================
        # Step 1: neighbor weighted aggregation -> K_agg, V_agg
        # ================================================================
        nb_w = self.nb_w(w_emb)                                        # [Wn, nb_K]

        k_in  = torch.cat([wy, fe], dim=-1)                            # [B, Wn, hw, Cy_ri+emb]
        K_agg = weighted_agg(k_in, nb_idx, nb_w)                        # [B, Wn, hw, Cy_ri+emb]

        v_in  = torch.cat([wy, z_wn], dim=-1)                          # [B, Wn, hw, Cy_ri+Cz]
        V_agg = weighted_agg(v_in, nb_idx, nb_w)                       # [B, Wn, hw, Cy_ri+Cz]

        # ================================================================
        # Step 2: project -> Q, K, Va, Vp
        # ================================================================
        Q  = self.Wq(z_flat)        # [B, hw, d]
        K  = self.Wk(K_agg)         # [B, Wn, hw, d]
        Va = self.Wv_A(V_agg)       # [B, Wn, hw, Cz]
        Vp = self.Wv_phi(V_agg)     # [B, Wn, hw, Cz]

        # ================================================================
        # Step 3: attention score (shared Q & K, split only at V)
        # softmax over Wn: "which wedge matters for this LR pixel"
        # ================================================================
        Q_exp = Q[:, None, :, :]                                       # [B, 1,  hw, d]
        score = (Q_exp * K).sum(-1) / (d ** 0.5)                      # [B, Wn, hw]
        attn  = score.softmax(dim=1)                                   # [B, Wn, hw]

        # ================================================================
        # Step 4: amplitude-phase modulation on z
        # ================================================================
        attn_exp = attn[:, :, :, None]                                 # [B, Wn, hw, 1]
        out_a = attn_exp * Va                                          # [B, Wn, hw, Cz]
        out_p = attn_exp * Vp                                          # [B, Wn, hw, Cz]

        A   = F.softplus(out_a)                                        # > 0
        phi = out_p

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

        # INR neighbor weights: project w_emb -> [Wn, nb_K] logits
        self.inr_nb_w = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, nb_K),
        )
        nn.init.zeros_(self.inr_nb_w[-1].weight)
        nn.init.zeros_(self.inr_nb_w[-1].bias)

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

        # ── neighbor-aggregated q_y_agg (FDCT physical prior) ─────────────
        # weighted_agg expects [B, Wn, hw, D]; reshape in/out accordingly
        _, nb_idx = build_neighbor_indices(
            wedge_index, self.num_scales, self.num_angles_coarse, device)
        nb_w    = self.inr_nb_w(w_emb)                                 # [Wn, K]
        q_y_4d  = q_y.reshape(B, Wn, N, C_y)                          # [B, Wn, N, C_y]
        q_y_nb  = weighted_agg(q_y_4d, nb_idx, nb_w)                  # [B, Wn, N, C_y]
        q_y_agg = (q_y_4d + q_y_nb).reshape(B * Wn, N, C_y)          # residual add

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
                inp  = torch.cat([q_z, q_y_agg, rel_coord], dim=-1)   # q_y -> q_y_agg
                pred = self.mlp(inp.view(B * Wn * N, -1)).view(B * Wn, N, -1)
                preds.append(pred)

        preds  = torch.stack(preds, dim=-1)                            # [B*Wn, N, out, 4]
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        out    = (preds[:, :, :-1, :] * weight.unsqueeze(-2)).sum(-1)  # [B*Wn, N, C_z]

        return out.view(B, Wn, H, W, C_z).permute(0, 4, 1, 2, 3)      # [B, C_z, Wn, H, W]

class CoarseINR(nn.Module):
    # (逻辑与原版基本一致，专门负责低频部分的插值和融合)
    def __init__(self, z_bands: int, y_bands: int, mlp_hidden: list = None, pe_d_model: int = 2):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.pe = PositionalEmbedding(pe_d_model)
        coord_dim = 2 * (2 * pe_d_model + 1)
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
        rx, ry = 1.0 / h, 1.0 / w
 
        grid_y = coord.clone().flip(-1).unsqueeze(1)
        q_y = F.grid_sample(coarse_y, grid_y, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
 
        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                grid_z  = coord_.flip(-1).unsqueeze(1)
                q_z     = F.grid_sample(coarse_z, grid_z, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, grid_z, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = self.pe((coord - q_coord) * torch.tensor([h, w], device=device))
                
                inp  = torch.cat([q_z, q_y, rel_coord], dim=-1)
                pred = self.mlp(inp.view(B * N, -1)).view(B, N, -1)
                preds.append(pred)
 
        preds  = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        out    = (preds[:, :, :-1, :] * weight.unsqueeze(-2)).sum(-1)
        return out.view(B, H, W, C_z).permute(0, 3, 1, 2)


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

        # 5. Low-frequency Coarse INR
        self.coarse_inr = CoarseINR(z_bands=hsi_dim, y_bands=msi_dim,
                                     mlp_hidden=mlp_hidden, pe_d_model=pe_d_model)

    def _fdct(self, x): return fdct2d(x, self.num_scales, self.num_angles_coarse)
    def _ifdct(self, coeffs, size): return ifdct2d(coeffs, size)

    def _forward_implem(self, Y: Tensor, lms: Tensor, Z: Tensor) -> Tensor:
        B, Cy, H, W = Y.shape
        _, Cz, h, w = Z.shape

        # =======================================================
        # Step 1: MSI (Y) frequency decomposition
        # =======================================================
        CY = self._fdct(Y)
        coarse_y = CY[0][0]                                   # [B, Cy, H, W]
        wedge_y, wedge_index = stack_wedges(CY)               # [B, Cy_ri, Wn, H, W]

        # =======================================================
        # Step 2: HSI (Z) spectral feature extraction (stays LR)
        # =======================================================
        z_guide = self.z_spectral_attn(Z)                     # [B, Cz, h, w]

        # =======================================================
        # Step 3: Low-frequency coarse fusion
        # =======================================================
        coarse_x = self.coarse_inr(coarse_y, z_guide)         # [B, Cz, H, W]

        # =======================================================
        # Step 4: Build wedge_z (LR) via cross-attention
        # =======================================================
        w_emb   = self.wedge_embed(wedge_index, device=Y.device)  # [Wn, embed_dim]
        wedge_z = self.wedge_z_attn(wedge_y, z_guide, w_emb, wedge_index)
        # wedge_z: [B, Cz_ri, Wn, h, w]

        # =======================================================
        # Step 5: High-frequency INR super-resolution
        # =======================================================
        wedge_x = self.wedge_inr(wedge_z, wedge_y, wedge_index, w_emb)

        # =======================================================
        # Step 6: Reconstruction via IFDCT
        # =======================================================
        CX_detail = unstack_wedges(wedge_x, CY, wedge_index)
        CX_detail[0] = [coarse_x.to(torch.complex64)]
        return self._ifdct(CX_detail, (B, Cz, H, W))

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(pan, lms, ms)
        loss = criterion(sr, gt)
        return sr.clamp(0, 1), loss

    def val_step(self, ms, lms, pan):
        pred = self._forward_implem(pan, lms, ms)
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