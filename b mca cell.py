"""
CurveletINR: Hyperspectral-Multispectral Image Fusion via
             *Fast Discrete Curvelet Transform + Implicit Neural Representation

Architecture: (Optimized Hybrid Approach)
    1. fast "stack_wedges" batch processing and accompanied wedge index (s,n)
    2. perform FDCT on the upsampled "lms" image
    *3. wedge physical graph neighbouring mask:
                    -> scale mask + spatial cross attention (nn.MultiheadAttention)
                    -> wedge mask + spectral self attention (5D gating & residual attention)
    4. one point q_y and four neighbouring q_z point for INR sampling
    *5. wedge physical embedding (sin/cos theta & s) for INR guidance
"""

# curvelet masked attention, physical aware aggregation (MLP), wedge physical embedding (equal wedge&scale + 2 cell)

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model.module.fe_block import make_coord, MLP
from model.base_model import BaseModel, register_model


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for relative coordinates."""
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


class WedgePhysicalEmbedding(nn.Module):
    """
    Physical Wedge Embedding of (s,n) list

    angles (wedges): theta_n = 2*pi*(n+0.5)/N_s; N_s = num_angles_coarse*2^s
                     [sin(theta_n),cos(theta_n)] -> continuous, periodic, default: 8
    scales         : Embedding(num_scales, scale_embed_dim) -> default: 8
    cell           : Tensor([rx,ry]) -> default: 2
    projection     : Linear(wedge_embed_dim + scale_embed_dim, embed_dim) + LayerNorm -> default: 16
    """
    def __init__(self, num_scales: int, num_angles_coarse: int, embed_dim: int = 16, scale_embed_dim: int = 8):
        super().__init__()
        self.num_angles_coarse = num_angles_coarse
        self.scale_embed = nn.Embedding(num_scales, scale_embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(scale_embed_dim*2+2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
 
    def forward(self, wedge_index: list, B: int, cell: Tensor, device) -> Tensor:
        """
        Args:
            wedge_index : list of (s, n) from stack_wedges, len = Wn
            B           : batch size
            device      : target device
        Returns:
            w_emb_raw : (B*Wn, embed_dim) -> for broadcasting
        """
        s_list = [s for s, _ in wedge_index]
        n_list = [n for _, n in wedge_index]
        Wn = len(wedge_index)
 
        ns_list = [self.num_angles_coarse * (2 ** s) for s in s_list]
 
        theta = torch.tensor(
            [2.0 * math.pi * (n + 0.5) / ns for n, ns in zip(n_list, ns_list)],
            dtype=torch.float32, device=device
        )  # (Wn,)

        sin_cos = torch.stack([torch.sin(theta), torch.cos(theta), torch.sin(theta*2), torch.cos(theta*2), torch.sin(theta*4), torch.cos(theta*4), torch.sin(theta*8), torch.cos(theta*8), ], dim=-1)  # (Wn, 8)
 
        s_ids = torch.tensor(s_list, dtype=torch.long, device=device)        # (Wn,)
        s_emb = self.scale_embed(s_ids)                                       # (Wn, scale_embed_dim=8)

        cell_exp=cell.unsqueeze(0).expand(Wn,-1).to(device)     #(Wn,2)
 
        feat = torch.cat([sin_cos, s_emb, cell_exp], dim=-1)                           # (Wn, 16)
        w_emb = self.proj(feat)                                               # (Wn, embed_dim)
 
        # (Wn,) -> (B*Wn, embed_dim)
        # w_emb = w_emb.unsqueeze(0).expand(B, Wn, -1).reshape(B * Wn, -1)
        return w_emb

class CurveletMaskedAttention(nn.Module):
    
    nb_K=5

    def __init__(
        self,
        z_channels: int,
        y_channels: int,
        num_wedges: int,
        num_scales: int,
        num_angles_coarse: int,
        attn_dim: int = 16,
        num_heads: int = 4,
        wedge_embed_dim: int = 16,
        # dropout: float = 0.0,
        ):
        super().__init__()
        assert attn_dim % num_heads == 0

        self.z_channels = z_channels
        self.y_channels = y_channels
        self.num_wedges = num_wedges
        self.num_scales = num_scales
        self.num_angles_coarse = num_angles_coarse
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.wedge_embed_dim=wedge_embed_dim

        # spatial multihead cross attention
        self.q_proj = nn.Linear(num_wedges * z_channels, attn_dim)
        self.k_proj = nn.Linear(num_wedges * y_channels, attn_dim)
        self.v_proj = nn.Linear(num_wedges * y_channels, attn_dim)

        # self.cross_attn = nn.MultiheadAttention(
        #     embed_dim=attn_dim,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True, # not conducting over-patch interaction
        # )

        # self.cross_nb_weight = nn.Parameter(torch.ones(self.nb_K))   # [K]
        # self.self_nb_weight  = nn.Parameter(torch.ones(self.nb_K))    # [K]
        self.cross_nb_proj = nn.Sequential(
            nn.Linear(wedge_embed_dim, wedge_embed_dim * 2),
            nn.GELU(),
            nn.Linear(wedge_embed_dim * 2, self.nb_K)
        )
        self.self_nb_proj = nn.Sequential(
            nn.Linear(wedge_embed_dim, wedge_embed_dim * 2),
            nn.GELU(),
            nn.Linear(wedge_embed_dim * 2, self.nb_K)
        )

        nn.init.zeros_(self.cross_nb_proj[-1].weight)
        nn.init.zeros_(self.self_nb_proj[-1].bias)
        
        nn.init.zeros_(self.self_nb_proj[-1].weight)
        nn.init.zeros_(self.self_nb_proj[-1].bias)

        # spectral self attention (Conv3d for 5D tensors projection)
        self.spectral_qkv  = nn.Conv3d(z_channels, z_channels * 3, kernel_size=1, bias=False)
        self.spectral_proj = nn.Conv3d(z_channels, z_channels,     kernel_size=1)
        self.spectral_temperature = nn.Parameter(torch.ones(1))

        self.out_proj = nn.Linear(attn_dim, num_wedges * z_channels)

        # self._mask_cache: Dict[Tuple[Tuple[int, int], ...], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._neighbor_cache: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    # create graph neighbouring mask on (s,n) coordinates
    @staticmethod
    def _angles_at_scale(num_angles_coarse: int, s: int) -> int:
        return num_angles_coarse * (2 ** s)

    # def _valid(self, s: int, n: int) -> bool:
    #     if s < 0 or s >= self.num_scales:
    #         return False
    #     return 0 <= n < self._angles_at_scale(self.num_angles_coarse, s)

    def _clamp_s(self, s: int) -> int:
        """Repeat boundary on scale axis."""
        return max(0, min(s, self.num_scales - 1))

    def _wrap_n(self, s: int, n: int) -> int:
        return n % self._angles_at_scale(self.num_angles_coarse, s)

    def _resolve(self, s: int, n: int) -> Tuple[int, int]:
        """Clamp s to boundary, then wrap n to valid angle range."""
        s2 = self._clamp_s(s)
        n2 = self._wrap_n(s2, n)
        return s2, n2

    def _build_neighbor_indices(
        self,
        wedge_index: List[Tuple[int, int]],
        device: torch.device,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cross_idx: [Wn, K]  long  — indices into wedge_index for each cross neighbor
            self_idx:  [Wn, K]   long  — indices into wedge_index for each self neighbor
        All wedges always have exactly K neighbors (repeat-boundary, no padding needed).
        """
        key = tuple(wedge_index)
        if key in self._neighbor_cache:
            cross_idx, self_idx = self._neighbor_cache[key]
            return cross_idx.to(device), self_idx.to(device)

        wi_map = {wn: i for i, wn in enumerate(wedge_index)}
        Wn = len(wedge_index)
        cross_idx = torch.zeros(Wn, self.nb_K, dtype=torch.long)
        self_idx  = torch.zeros(Wn, self.nb_K,  dtype=torch.long)

        for i, (s, n) in enumerate(wedge_index):
            # cross neighbors: self, +scale children x3, -scale parent
            cross_nb = [
                (s, n),
                self._resolve(s + 1, 2 * n),
                self._resolve(s + 1, 2 * n - 1),
                self._resolve(s + 1, 2 * n + 1),
                self._resolve(s - 1, n // 2),
            ]
            for k, wn in enumerate(cross_nb):
                cross_idx[i, k] = wi_map.get(wn, i)  # fallback to self if missing
 
            # self neighbors: self, -scale parent, angle±1, +scale child
            self_nb = [
                (s, n),
                self._resolve(s - 1, n // 2),
                (s, self._wrap_n(s, n - 1)),
                (s, self._wrap_n(s, n + 1)),
                self._resolve(s + 1, 2 * n),
            ]
            for k, wn in enumerate(self_nb):
                self_idx[i, k] = wi_map.get(wn, i)
 
        self._neighbor_cache[key] = (cross_idx.cpu(), self_idx.cpu())
        return cross_idx.to(device), self_idx.to(device)

    @staticmethod
    def _weighted_agg(
        feat: torch.Tensor,       # [B, Wn, *, D]  (* = HW or D)
        idx:  torch.Tensor,       # [Wn, K]
        w:    torch.Tensor,       # [Wn, K]
        gather_dim: int = 1,
        ) -> torch.Tensor:
        """
        Gather K neighbors along gather_dim=1 (Wn axis), then weighted-sum.
        feat layout assumed [B, Wn, ...].
        """
        B, Wn = feat.shape[:2]
        K = idx.shape[1]
        # gather: [B, Wn*K, ...]
        nb = feat[:, idx.reshape(-1)]                      # [B, Wn*K, ...]
        nb = nb.reshape(B, Wn, K, *feat.shape[2:])        # [B, Wn, K, ...]
        w_norm = w.softmax(dim=-1)                          # [K]  sums to 1
        # broadcast weights over all trailing dims
        for _ in feat.shape[2:]:
            w_norm = w_norm.unsqueeze(-1)
        w_norm = w_norm.unsqueeze(0)          # [1, Wn, K, ...]
        return (nb * w_norm).sum(dim=2)                    # [B, Wn, ...]

    # define two inner attention
    def _masked_spatial_attn(
        self,
        z: torch.Tensor,           # [B, Cz, Wn, H, W]
        y: torch.Tensor,           # [B, Cy, Wn, H, W]
        # cross_mask: torch.Tensor,  # [Wn, Wn] bool, True = allowed
        cross_idx: torch.Tensor,   # [Wn, K]
        w_emb: torch.Tensor, # [Wn, wedge_embed_dim]
        # cross_cnt,
        ) -> torch.Tensor:
        """
        Masked spatial cross-attention.

        1. reshape: [B, C, Wn, H, W] -> Q/K/V: [B, Wn, HW, C]
        2. for query token i, aggregate K/V from source token j (cross_mask[i][j]==True)
                -> mean-pool K/V over masked Wn neighbors
        3. spatial attention: Q [B, Wn, HW, C] x K_agg [B, Wn, HW, C]
                -> weight map: [B, Wn, HW, HW]  (spatial)
        4. 5D tensor output
        """
        B, Cz, Wn, H, W = z.shape
        HW = H * W
        # K=cross_idx.shape[1]

        # [B, C, Wn, H, W] -> [B, HW, Wn * C]
        z_seq = z.permute(0, 3, 4, 2, 1).reshape(B, HW, Wn * Cz)
        y_seq = y.permute(0, 2, 3, 4, 1).reshape(B, Wn, HW, self.y_channels)
 
        # Aggregate y neighbors with learnable weights -> [B, Wn, HW, Cy]
        cross_nb_weight=self.cross_nb_proj(w_emb)   # [Wn,K]

        y_agg = self._weighted_agg(y_seq, cross_idx, cross_nb_weight)
        # [B, HW, Wn*Cy]
        ky = y_agg.permute(0, 2, 1, 3).reshape(B, HW, Wn * self.y_channels)
 
        q = self.q_proj(z_seq)   # [B, HW, D]
        k = self.k_proj(ky)
        v = self.v_proj(ky)

        head_dim = self.attn_dim // self.num_heads
        q_in = q.view(B, HW, self.num_heads, head_dim).transpose(1, 2)
        k_in = k.view(B, HW, self.num_heads, head_dim).transpose(1, 2)
        v_in = v.view(B, HW, self.num_heads, head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_in, k_in, v_in)  # [B, num_heads, HW, head_dim]
        out = out.transpose(1, 2).reshape(B, HW, self.attn_dim)  # [B, HW, D]

        # print(out.shape)
        out = self.out_proj(out)  # [B, HW, Cz*Wn]
        # print(out.shape)
        out = out.reshape(B, H, W, Cz, Wn).permute(0, 3, 4, 1, 2).contiguous() # [B, Cz, Wn, H, W]

        return out + z

    def _masked_spectral_attn(
        self,
        x: torch.Tensor,          # [B, C, Wn, H, W]
        # self_mask: torch.Tensor,   # [Wn, Wn] bool, True = allowed
        self_idx: torch.Tensor,   # [Wn, K]
        w_emb: torch.Tensor,      # [Wn, wedge_embed_dim]
        ) -> torch.Tensor:
        """
        Masked spectral self-attention.

        1. reshape: [B, D, Wn, H, W] -> Q/K/V: [B, Wn, D, HW]
        2. for query i, caculate attention on key/value j (self_mask[i][j]==True)
                -> weight map * temperature: [B, Wn, D, D] (channel-wise coviriance)
        3. residual + proj[attention output]

        """
        B, C, Wn, H, W = x.shape
        HW = H * W
        # K=self_idx.shape[-1]

        qkv = self.spectral_qkv(x)              # [B, 3C, Wn, H, W]
        q, k, v = qkv.chunk(3, dim=1)           # [B, C, Wn, H, W]

        q = q.permute(0, 2, 1, 3, 4).reshape(B, Wn, C, HW)
        k = k.permute(0, 2, 1, 3, 4).reshape(B, Wn, C, HW)
        v = v.permute(0, 2, 1, 3, 4).reshape(B, Wn, C, HW)

        # L2 Norm
        q = F.normalize(q, dim=-1)              # [B, Wn, C, HW]
        k = F.normalize(k, dim=-1)

        # Aggregate neighbors with learnable weights -> [B, Wn, C, HW]
        self_nb_weight=self.self_nb_proj(w_emb)   # [Wn,K]
        k_agg = self._weighted_agg(k, self_idx, self_nb_weight)
        v_agg = self._weighted_agg(v, self_idx, self_nb_weight)

        attn = (q @ k_agg.transpose(-2, -1)) * self.spectral_temperature
        attn = attn.softmax(dim=-1)               # [B, Wn, C, C]

        out = attn @ v_agg                        # [B, Wn, C, HW]
        out = out.reshape(B, Wn, C, H, W).permute(0, 2, 1, 3, 4).contiguous()   # [B, C, Wn, H, W]

        # proj + residual
        out = self.spectral_proj(out)

        return out + x

    def forward(
        self,
        z: torch.Tensor,           # [B, Cz, Wn, H, W]
        y: torch.Tensor,           # [B, Cy, Wn, H, W]
        wedge_index: List[Tuple[int, int]],
        w_emb: torch.Tensor,       # [Wn, wedge_embed_dim]
        ) -> torch.Tensor:

        B, Cz, Wn, H, W = z.shape
        device = z.device
        # cross_mask, self_mask = self._build_masks(wedge_index, device)
        cross_idx, self_idx = self._build_neighbor_indices(wedge_index, device)


        # Stage 1: masked spatial cross-attention
        cross_out_5d = self._masked_spatial_attn(z, y, cross_idx,w_emb)
        # cross_out_5d = self._masked_spatial_attn(z, y, cross_mask)

        # Stage 2: masked spectral self-attention
        out = self._masked_spectral_attn(cross_out_5d, self_idx,w_emb)    # [B, H, W, Wn, Cz]
        # spec_5d = self._masked_spectral_attn(cross_out_5d, self_mask)

        # # Stage 3: projection
        # out = (
        #     spec_5d
        #     .permute(0, 3, 4, 2, 1)              # [B, H, W, Wn, D]
        #     .contiguous()
        #     # .reshape(B * H * W, Wn, self.attn_dim)
        # )
        # out = self.out_proj(out)                  # [B*HW, Wn, Cz]
        # out = out.reshape(B, H, W, Wn, Cz).permute(0, 4, 3, 1, 2).contiguous()  # [B, Cz, Wn, H, W]
        return out

class WedgeINR(nn.Module):
    def __init__(self, z_bands: int, y_bands: int, num_wedges: int,
                 num_scales: int, num_angles_coarse: int,
                 mlp_hidden: list = None, pe_d_model: int = 2,
                 wedge_embed_dim: int = 16, scale_embed_dim: int = 8):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.pe = PositionalEmbedding(pe_d_model)
        coord_dim = 2 * (2 * pe_d_model + 1)
        # self.wedge_embed = nn.Embedding(num_wedges, 16)
        in_dim = z_bands + y_bands + coord_dim + 16
        out_dim = z_bands + 1
        self.mlp = MLP(in_dim, out_dim, mlp_hidden)

    def forward(self, cz_wedge_attn: Tensor, cy_wedge: Tensor,
                lr_size: tuple, 
                w_emb_raw: Tensor,  #[Wn, wedge_embed_dim]
                # wedge_index: list,
                ):
        B, C_z, Wn, H, W = cz_wedge_attn.shape
        _, C_y, _, _, _ = cy_wedge.shape
        h, w = lr_size
        N = H * W
        device = cz_wedge_attn.device

        coord = make_coord((H, W)).unsqueeze(0).expand(B * Wn, -1, -1).to(device) 
        feat_coord = make_coord((h, w), flatten=False).permute(2, 0, 1).unsqueeze(0).expand(B * Wn, -1, -1, -1).to(device) 
        rx, ry = 1.0 / h, 1.0 / w
        # rx, ry=h/H, w/W
        # cell=torch.tensor([rx,ry],dtype=torch.float32,device=device)

        z_feat_flat = cz_wedge_attn.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_z, H, W)
        y_feat_flat = cy_wedge.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_y, H, W)

        # wedge_ids = torch.arange(Wn, device=device).unsqueeze(0).expand(B, Wn).reshape(-1)
        # w_emb = self.wedge_embed(wedge_ids).unsqueeze(1).expand(-1, N, -1)

        # w_emb_raw = self.wedge_embed(wedge_index, B, cell, device)           # (Wn, embed_dim)
        w_emb = w_emb_raw.unsqueeze(0).unsqueeze(2).expand(B, -1, N, -1).reshape(B * Wn, N, -1) # (B*Wn, N, embed_dim)

        grid_y = coord.clone().flip(-1).unsqueeze(1)
        q_y = F.grid_sample(y_feat_flat, grid_y, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        preds = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx
                coord_[:, :, 1] += vy * ry
                grid_z = coord_.flip(-1).unsqueeze(1)
                q_z = F.grid_sample(z_feat_flat, grid_z, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                q_coord = F.grid_sample(feat_coord, grid_z, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                rel_coord = self.pe((coord - q_coord) * torch.tensor([h, w], device=device))
                inp = torch.cat([q_z, q_y, rel_coord, w_emb], dim=-1)
                pred = self.mlp(inp.view(B * Wn * N, -1)).view(B * Wn, N, -1)
                preds.append(pred)
        
        preds = torch.stack(preds, dim=-1)
        weight = F.softmax(preds[:, :, -1, :], dim=-1)
        out_flat = (preds[:, :, :-1, :] * weight.unsqueeze(-2)).sum(-1)
        
        return out_flat.view(B, Wn, H, W, C_z).permute(0, 4, 1, 2, 3)


class LowFreqFiLM(nn.Module):
    def __init__(self, z_bands: int, y_bands: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(z_bands + y_bands, hidden, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hidden, hidden, 3, 1, 1), nn.ReLU(inplace=True))
        self.gamma_head = nn.Conv2d(hidden, z_bands, 1)
        self.beta_head = nn.Conv2d(hidden, z_bands, 1)

    def forward(self, coarse_y: Tensor, coarse_z_up: Tensor):
        feat = self.net(torch.cat([coarse_y, coarse_z_up], dim=1))
        gamma = self.gamma_head(feat) + 1.0
        beta = self.beta_head(feat)
        return gamma * coarse_z_up + beta

# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
@register_model('b')
class CuINR(BaseModel):
    def __init__(self, hsi_dim: int=31, msi_dim: int=4, num_scales: int=4, num_angles_coarse: int=8,
                 attn_embed_dim: int=64, attn_num_heads: int=4, mlp_hidden: list=None,
                 film_hidden: int=128, pe_d_model: int=2, scale: int=4, wedge_embed_dim: int = 16,):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.hsi_dim, self.msi_dim = hsi_dim, msi_dim
        self.num_scales, self.num_angles_coarse, self.scale = num_scales, num_angles_coarse, scale
        self.num_wedges = sum(num_angles_coarse * (2 ** s) for s in range(num_scales - 1))
        
        z_bands_ri, y_bands_ri = hsi_dim * 2, msi_dim * 2
        
        self.wedge_embed = WedgePhysicalEmbedding(
            num_scales=num_scales,
            num_angles_coarse=num_angles_coarse,
            embed_dim=wedge_embed_dim,
            # scale_embed_dim=scale_embed_dim,
        )

        self.attention = CurveletMaskedAttention(
            z_channels=z_bands_ri,
            y_channels=y_bands_ri,
            num_wedges=self.num_wedges,
            num_scales=num_scales - 1,
            num_angles_coarse=num_angles_coarse,
            attn_dim=64,
            num_heads=4,
            # dropout=0.0,
        )
        
        self.wedge_inr = WedgeINR(
            z_bands_ri, y_bands_ri, self.num_wedges,
            num_scales=num_scales - 1,          # num_detail = num_scales - 1
            num_angles_coarse=num_angles_coarse,
            mlp_hidden=mlp_hidden,
        )
        
        self.low_freq_film = LowFreqFiLM(hsi_dim, msi_dim, film_hidden)

    def _fdct(self, x): return fdct2d(x, self.num_scales, self.num_angles_coarse)
    def _ifdct(self, coeffs, size): return ifdct2d(coeffs, size)

    def _forward_implem(self, Y: Tensor, lms: Tensor, Z: Tensor) -> Tensor:
        B, C, H, W = lms.shape
        _,_,h,w=Z.shape

        a=3
        def lanczos_kernel_1d(size, a=3):
            x = torch.linspace(-(a-0.5), a-0.5, size)
            kernel = torch.sinc(x) * torch.sinc(x / a)
            kernel = kernel / kernel.sum()
            return kernel

        k_size = 2 * a + 1
        kh = lanczos_kernel_1d(k_size, a).to(lms.device)
        kw = lanczos_kernel_1d(k_size, a).to(lms.device)

        kernel_2d = kh.unsqueeze(1) * kw.unsqueeze(0)  # (k, k)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        kernel_2d = kernel_2d.expand(C, 1, k_size, k_size)  # (C,1,k,k)

        lms = F.conv2d(lms, kernel_2d, padding=a, groups=C)
        
        # Step 1: FDCT
        CY = self._fdct(Y)
        CLZ = self._fdct(lms) 

        # Step 2: Stack wedges for efficient batch processing
        CY_wedge, wedge_index = stack_wedges(CY)
        CLZ_wedge, _ = stack_wedges(CLZ)

        # Step 2.5: physical wedge embedding
        cell=torch.tensor([h/H,w/W],dtype=torch.float32,device=lms.device)
        w_emb=self.wedge_embed(wedge_index,B,cell,lms.device)

        # Step 3: Attention (Masked Spatial Cross-Attention + Masked Spectral Self-Attention)
        CZ_wedge_attn = self.attention(CLZ_wedge, CY_wedge, wedge_index,w_emb)
        
        Hs, Ws = CY_wedge.shape[-2], CY_wedge.shape[-1]
        # lr_size = (Hs // self.scale, Ws // self.scale)
        lr_size=(h,w)
        
        # Step 4: INR Processing
        detail_X_wedge = self.wedge_inr(CZ_wedge_attn, CY_wedge, lr_size=lr_size, w_emb_raw=w_emb)
        CX_detail = unstack_wedges(detail_X_wedge, CLZ, wedge_index) 

        # Step 5: Low-frequency modulation 
        coarse_y = CY[0][0]
        coarse_lz = CLZ[0][0]
        coarse_x = self.low_freq_film(coarse_y, coarse_lz)
        CX_detail[0] = [coarse_x]

        # Step 6: Reconstruction
        X_rec = self._ifdct(CX_detail, (B, C, H, W))
        
        return X_rec

    def train_step(self, ms, lms, pan, gt, criterion):
        sr = self._forward_implem(pan, lms, ms)
        loss = criterion(sr, gt)
        return sr.clamp(0, 1), loss

    def val_step(self, ms, lms, pan):
        pred = self._forward_implem(pan, lms, ms)
        return pred.clamp(0, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, C, H, W, scale = 2, 31, 32, 32, 4
    model = CuINR(hsi_dim=C, msi_dim=3, num_scales=3, num_angles_coarse=4,
                  attn_embed_dim=32, attn_num_heads=2, mlp_hidden=[64, 32],
                  scale=scale).to(device)

    Y   = torch.randn(B, 3, H, W).to(device)
    lms = torch.randn(B, C, H, W).to(device)
    Z   = torch.randn(B, C, H // scale, W // scale).to(device)
    gt  = torch.randn(B, C, H, W).to(device)

    criterion = nn.L1Loss()
    sr, loss  = model.train_step(Z, lms, Y, gt, criterion)

    print(f"Output shape    : {sr.shape}")
    print(f"Train loss      : {loss.item():.4f}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
