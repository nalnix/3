"""
CurveletINR: Hyperspectral-Multispectral Image Fusion via
             Fast Discrete Curvelet Transform + Implicit Neural Representation

Architecture: (Optimized Hybrid Approach)
  - Retains the fast "stack_wedges" batch processing from the original version.
  - Fix 1: Corrected FDCT windowing to prevent corner-frequency redundancy.
  - Fix 2: Performs FDCT on the upsampled 'lms' image.
  - NEW Fix 3: Added lightweight Spectral Self-Attention to enhance inter-band local correlation.
  - NEW Fix 4: Replaced global FiLM with SPADE for explicit spatial (B, C, H, W) modulation.
"""

import math
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

class CrossBandWedgeAttention(nn.Module):
    def __init__(self, z_bands: int, y_bands: int, num_wedges: int, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.embed_dim, self.num_heads, self.num_wedges = embed_dim, num_heads, num_wedges
        self.q_proj = nn.Linear(z_bands * num_wedges, embed_dim)
        self.k_proj = nn.Linear(y_bands * num_wedges, embed_dim)
        self.v_proj = nn.Linear(y_bands * num_wedges, embed_dim)
        self.out_proj = nn.Linear(embed_dim, z_bands * num_wedges)

    def forward(self, cz_wedge: Tensor, cy_wedge: Tensor):
        B, C_z, Wn, H, W = cz_wedge.shape
        _, C_y, _, _, _ = cy_wedge.shape
        N = H * W

        q = cz_wedge.permute(0, 3, 4, 1, 2).reshape(B, N, C_z * Wn)
        k = cy_wedge.permute(0, 3, 4, 1, 2).reshape(B, N, C_y * Wn)
        
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(k)
        
        head_dim = self.embed_dim // self.num_heads
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.embed_dim)
        attn_out = self.out_proj(attn_out)
        
        return attn_out.view(B, H, W, C_z, Wn).permute(0, 3, 4, 1, 2)


# ==========================================
# 新增点 1：轻量级光谱自注意力 (Transposed Channel Attention)
# ==========================================
class SpectralSelfAttention(nn.Module):
    """
    作用于通道 (Channel) 维度的自注意力机制。
    使用转置注意力矩阵计算 K * Q^T 生成 C x C 注意力图，
    不仅极大降低空间大分辨率下的内存消耗 (避免 O(N^2) )，而且能完美捕捉波段间的协方差关系。
    """
    def __init__(self, channels: int):
        super().__init__()
        # nn.Conv3d 配合 kernel_size=1 等效于在 Wn, H, W 三个空间/方向维度上的 point-wise Linear
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        B, C, Wn, H, W = x.shape
        
        # 1. 生成 Q, K, V
        qkv = self.qkv(x) # [B, 3C, Wn, H, W]
        q, k, v = qkv.chunk(3, dim=1)
        
        # 2. 展平为二维序列，计算通道维度的注意力
        N = Wn * H * W
        q = q.view(B, C, N)
        k = k.view(B, C, N)
        v = v.view(B, C, N)
        
        # 对 Q, K 进行 L2 归一化 (稳定 Transposed Attention 计算)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # 计算 C x C 注意力图
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        # 3. 将通道特征重构作用于 V
        out = (attn @ v).view(B, C, Wn, H, W)
        
        # 4. 映射输出 + 残差
        out = self.proj(out)
        return out + x


class WedgeINR(nn.Module):
    def __init__(self, z_bands: int, y_bands: int, num_wedges: int, mlp_hidden: list = None, pe_d_model: int = 2):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.pe = PositionalEmbedding(pe_d_model)
        coord_dim = 2 * (2 * pe_d_model + 1)
        self.wedge_embed = nn.Embedding(num_wedges, 16)
        in_dim = z_bands + y_bands + coord_dim + 16
        out_dim = z_bands + 1
        self.mlp = MLP(in_dim, out_dim, mlp_hidden)

    def forward(self, cz_wedge_attn: Tensor, cy_wedge: Tensor, lr_size: tuple):
        B, C_z, Wn, H, W = cz_wedge_attn.shape
        _, C_y, _, _, _ = cy_wedge.shape
        h, w = lr_size
        N = H * W
        device = cz_wedge_attn.device

        coord = make_coord((H, W)).unsqueeze(0).expand(B * Wn, -1, -1).to(device) 
        feat_coord = make_coord((h, w), flatten=False).permute(2, 0, 1).unsqueeze(0).expand(B * Wn, -1, -1, -1).to(device) 
        rx, ry = 1.0 / h, 1.0 / w

        z_feat_flat = cz_wedge_attn.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_z, H, W)
        y_feat_flat = cy_wedge.permute(0, 2, 1, 3, 4).reshape(B * Wn, C_y, H, W)

        wedge_ids = torch.arange(Wn, device=device).unsqueeze(0).expand(B, Wn).reshape(-1)
        w_emb = self.wedge_embed(wedge_ids).unsqueeze(1).expand(-1, N, -1)

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


# ==========================================
# 新增点 2：替换为 SPADE 架构的低频空间自适应调制
# ==========================================
class LowFreqSPADE(nn.Module):
    """
    Spatially-Adaptive Normalization (SPADE)。
    不再使用全局池化/广播，而是显式通过独立卷积层生成尺寸为 [B, C, H, W] 的空间矩阵进行 out = x * scale + shift
    """
    def __init__(self, z_bands: int, y_bands: int, hidden: int = 128):
        super().__init__()
        # 1. 对被调制的低频主体 (HSI) 做实例归一化
        self.norm = nn.InstanceNorm2d(z_bands, affine=False)
        
        # 2. 共享条件提取网络 (融合 HSI + MSI 的低频语义上下文)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(z_bands + y_bands, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 3. 使用两个并行的普通卷积显式生成 H x W 尺寸的空间尺度和偏置矩阵
        self.conv_scale = nn.Conv2d(hidden, z_bands, kernel_size=3, padding=1)
        self.conv_shift = nn.Conv2d(hidden, z_bands, kernel_size=3, padding=1)

    def forward(self, coarse_y: Tensor, coarse_z: Tensor):
        # 归一化输入 x
        z_norm = self.norm(coarse_z)
        
        # 提取条件特征 (使用 cat 保留双模态共同引导信息)
        cond_feat = self.shared_conv(torch.cat([coarse_y, coarse_z], dim=1))
        
        # 独立生成 [B, C, H, W] 的 scale 和 shift 矩阵
        scale = self.conv_scale(cond_feat)
        shift = self.conv_shift(cond_feat)
        
        # SPADE 乘加运算 (1.0 + scale 用于初始化接近恒等映射，提升稳定)
        out = z_norm * (1.0 + scale) + shift
        return out

# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────
@register_model('b')
class CuINR(BaseModel):
    def __init__(self, hsi_dim: int=31, msi_dim: int=4, num_scales: int=4, num_angles_coarse: int=8,
                 attn_embed_dim: int=64, attn_num_heads: int=4, mlp_hidden: list=None,
                 film_hidden: int=128, pe_d_model: int=2, scale: int=4):
        super().__init__()
        if mlp_hidden is None: mlp_hidden = [256, 128]
        self.hsi_dim, self.msi_dim = hsi_dim, msi_dim
        self.num_scales, self.num_angles_coarse, self.scale = num_scales, num_angles_coarse, scale
        self.num_wedges = sum(num_angles_coarse * (2 ** s) for s in range(num_scales - 1))
        
        z_bands_ri, y_bands_ri = hsi_dim * 2, msi_dim * 2
        
        self.cross_attn = CrossBandWedgeAttention(z_bands_ri, y_bands_ri, self.num_wedges, attn_embed_dim, attn_num_heads)
        # -> 接入轻量光谱自注意力模块
        self.spectral_attn = SpectralSelfAttention(z_bands_ri) 
        
        self.wedge_inr = WedgeINR(z_bands_ri, y_bands_ri, self.num_wedges, mlp_hidden, pe_d_model)
        
        # -> 替换为 SPADE 调制模块
        self.low_freq_spade = LowFreqSPADE(hsi_dim, msi_dim, film_hidden)

    def _fdct(self, x): return fdct2d(x, self.num_scales, self.num_angles_coarse)
    def _ifdct(self, coeffs, size): return ifdct2d(coeffs, size)

    def _forward_implem(self, Y: Tensor, lms: Tensor, Z: Tensor) -> Tensor:
        B, C, H, W = lms.shape
        
        # Step 1: FDCT
        CY = self._fdct(Y)
        CLZ = self._fdct(lms) 

        # Step 2: Stack wedges for efficient batch processing
        CY_wedge, wedge_index = stack_wedges(CY)
        CLZ_wedge, _ = stack_wedges(CLZ)

        # Step 3: Attention (Spatial Cross-Attention + Spectral Self-Attention)
        CZ_wedge_attn = self.cross_attn(CLZ_wedge, CY_wedge)
        
        # --- 增加的光谱通道自注意力处理 ---
        CZ_wedge_attn = self.spectral_attn(CZ_wedge_attn)
        
        Hs, Ws = CY_wedge.shape[-2], CY_wedge.shape[-1]
        lr_size = (Hs // self.scale, Ws // self.scale)
        
        # Step 4: INR Processing
        detail_X_wedge = self.wedge_inr(CZ_wedge_attn, CY_wedge, lr_size=lr_size)
        CX_detail = unstack_wedges(detail_X_wedge, CLZ, wedge_index) 

        # Step 5: SPADE Low-frequency modulation 
        coarse_y = CY[0][0]
        coarse_lz = CLZ[0][0]
        coarse_x = self.low_freq_spade(coarse_y, coarse_lz)  # --- 调用 SPADE ---
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