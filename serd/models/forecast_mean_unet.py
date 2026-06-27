# -*- coding: utf-8 -*-
"""
ForecastMeanUNet3D2D
====================
均值/回归模型版本：
- 无 sigma / 无 sde 相关条件
- 保留 label / valid_time / init_time 条件（可选）
- 保留 base_emb（可学习向量）作为全局条件底座
- 条件融合改为“联合编码”：
    base / lead / valid / init 分别编码 -> concat -> MLP + LayerNorm
    trunk 和 variable head 各有自己的 joint encoder
- 每个变量 head 改成：
    2 个 conditioned ResBlock + 最后 3x3 conv
- 输入输出接口保持不变

输出:
    mu_err = model(cond_2d=..., forecast_3d=..., y=..., obs_time=..., init_time=...)
    mu_err.shape == (B, out_channels, H, W)
"""

import math
from typing import Iterable, Tuple, Optional, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 基础组件
# =========================

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int):
    g = min(32, max(1, channels // 4))
    while g > 1 and (channels % g != 0):
        g -= 1
    return GroupNorm32(g, channels, eps=1e-6)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def zero_module(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors, ctx.input_params, output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag: bool):
    if not flag:
        return func(*inputs)
    args = tuple(inputs) + tuple(params)
    return CheckpointFunction.apply(func, len(inputs), *args)


# =========================
# 连续 embedding（用于 z_coord）
# =========================

def continuous_sincos_embedding(x: torch.Tensor, dim: int, max_period: int = 10000):
    """
    x: 任意 shape 的连续标量张量，建议归一化到 [0,1]
    输出: x.shape + (dim,)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=x.device)
        / max(half, 1)
    )
    args = x.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


# =========================
# UNet 通用组件
# =========================

class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# -------------------------
# 注意力模块（空间注意力）
# -------------------------

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads if num_head_channels == -1 else max(1, channels // num_head_channels)
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads) if use_new_attention_order else QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x_flat))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# =========================
# 大核/空洞卷积 ResBlock + FiLM
# =========================

class ResBlockLarge(TimestepBlock):
    def __init__(
        self,
        in_channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        *,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        ksize: int = 5,
        dilations: Tuple[int, int] = (1, 2),
        use_conv_skip: bool = False,
        resample_with_conv: bool = True,
    ):
        super().__init__()
        assert ksize in (3, 5, 7), "ksize should be 3/5/7"
        self.in_ch = in_channels
        self.out_ch = out_channels or in_channels
        self.use_checkpoint = use_checkpoint
        self.updown = up or down
        d1, d2 = dilations
        p1 = (ksize // 2) * d1
        p2 = (ksize // 2) * d2

        if up:
            self.h_upd = Upsample(in_channels, resample_with_conv, dims)
            self.x_upd = Upsample(in_channels, resample_with_conv, dims)
        elif down:
            self.h_upd = Downsample(in_channels, resample_with_conv, dims)
            self.x_upd = Downsample(in_channels, resample_with_conv, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.norm1 = normalization(in_channels)
        self.act1 = nn.SiLU()

        self.film1 = nn.Linear(emb_channels, 2 * in_channels)
        with torch.no_grad():
            nn.init.zeros_(self.film1.weight)
            nn.init.zeros_(self.film1.bias)

        self.conv1 = conv_nd(dims, in_channels, self.out_ch, ksize, padding=p1, dilation=d1)

        self.norm2 = normalization(self.out_ch)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = zero_module(conv_nd(dims, self.out_ch, self.out_ch, ksize, padding=p2, dilation=d2))

        if self.out_ch == in_channels:
            self.skip = nn.Identity()
        elif use_conv_skip:
            self.skip = conv_nd(dims, in_channels, self.out_ch, 3, padding=1)
        else:
            self.skip = conv_nd(dims, in_channels, self.out_ch, 1)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        h = self.norm1(x)
        h = self.act1(h)

        if self.updown:
            h = self.h_upd(h)
            x = self.x_upd(x)

        scale, shift = self.film1(emb).chunk(2, dim=1)
        while scale.dim() < h.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        h = h * (1.0 + scale) + shift

        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.drop(h)
        h = self.conv2(h)
        return self.skip(x) + h


# =========================
# 3D level-wise Transformer（沿 Z 层 self-attention） + 连续 pressure embedding
# =========================

class LevelTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int = 2,
        nhead: int = 4,
        dim_feedforward: Optional[int] = None,
        max_z_levels: int = 64,
        use_film: bool = True,
        *,
        p_min_hpa: float = 0.1,
        p_max_hpa: float = 1100.0,
        z_fourier_dim: Optional[int] = None,
        max_period: int = 10000,
    ):
        super().__init__()
        self.channels = channels
        self.max_z_levels = max_z_levels
        self.use_film = use_film
        self.max_period = max_period

        self.register_buffer("log_p_min", torch.tensor(math.log(p_min_hpa), dtype=torch.float32), persistent=False)
        self.register_buffer("log_p_max", torch.tensor(math.log(p_max_hpa), dtype=torch.float32), persistent=False)

        d_model = channels
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.z_fourier_dim = int(z_fourier_dim) if z_fourier_dim is not None else d_model
        self.z_proj = nn.Linear(self.z_fourier_dim, d_model)

        if self.use_film:
            self.film = nn.Linear(d_model, 2 * channels)
            with torch.no_grad():
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, z_coord_hpa: torch.Tensor) -> torch.Tensor:
        B, C, Z, H, W = h.shape
        if Z > self.max_z_levels:
            raise ValueError(f"Z={Z} exceeds LevelTransformer.max_z_levels={self.max_z_levels}")
        assert z_coord_hpa.shape == (B, Z), f"z_coord_hpa must be (B,Z), got {z_coord_hpa.shape}"

        tokens = h.mean(dim=(-1, -2)).permute(0, 2, 1)  # (B,Z,C)

        logp = torch.log(z_coord_hpa.to(dtype=torch.float32).clamp(min=1e-6))
        denom = (self.log_p_max - self.log_p_min).clamp(min=1e-6)
        z_norm = ((logp - self.log_p_min) / denom).clamp(0.0, 1.0)

        z_fourier = continuous_sincos_embedding(z_norm, self.z_fourier_dim, max_period=self.max_period)
        z_pos = self.z_proj(z_fourier)  # (B,Z,C)

        tokens = tokens + z_pos
        tokens = self.encoder(tokens)

        if self.use_film:
            scale_shift = self.film(tokens)
            scale, shift = scale_shift.chunk(2, dim=-1)
            scale = scale.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            h = h * (1.0 + scale) + shift
        else:
            tokens_back = tokens.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            h = h + tokens_back

        return h


# =========================
# Z 维可学习加权池化（支持变长 Z）
# =========================

class ZSpatialSoftmaxPool(nn.Module):
    def __init__(self, channels: int, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.to_logits = conv_nd(3, channels, 1, 1)
        with torch.no_grad():
            nn.init.zeros_(self.to_logits.weight)
            if self.to_logits.bias is not None:
                nn.init.zeros_(self.to_logits.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.to_logits(x)  # (B,1,Z,H,W)
        w = torch.softmax(logits.float(), dim=2).type(logits.dtype)
        return (w * x).sum(dim=2)   # (B,C,H,W)


# =========================
# 主模型：均值/回归版（无 sigma）
# =========================

class ForecastMeanUNet3D2D(nn.Module):
    """
    均值/回归网络（无 sigma / 无噪声时间步）
    - cond_2d: (B, in_channels_2d, H, W)
    - forecast_3d: (B, in_channels_3d, Z, H, W)
    - 输出: (B, out_channels, H, W)

    改动：
    1) 条件融合从加法改成联合编码：
       base / lead / valid / init 分别编码后 concat -> MLP + LayerNorm
       trunk 和 variable head 各有自己的 joint encoder
    2) 每个变量 head:
       2 个 conditioned ResBlock + 最后 3x3 conv
    3) 修复 res3d_up1 / res3d_up2 在 num_res_blocks_3d > 1 时的通道错误
    """

    def __init__(
        self,
        in_channels_2d: int,
        in_channels_3d: int,
        out_channels: int,
        num_classes: Optional[int],
        *,
        model_channels_2d: int = 128,
        base_channels_3d: int = 32,
        num_res_blocks_2d: int = 1,
        num_res_blocks_3d: int = 1,
        channel_mult_2d: Tuple[int, ...] = (1, 2, 4, 4),
        dropout: float = 0.20,
        ksize: int = 5,
        dilations_2d: Tuple[int, int] = (1, 2),
        dilations_3d: Tuple[int, int] = (1, 1),
        attn_ds_2d: Iterable[int] = (4, 8, 16),
        attn_3d: bool = True,
        use_label_cond: bool = True,
        use_obs_time: bool = True,
        pad_to_mult_of_32: bool = True,
        max_z_levels: int = 64,
        fixed_z_coord_hpa: Optional[torch.Tensor] = None,
        num_var_head_res_blocks: int = 2,
    ):
        super().__init__()

        self.in_channels_2d = in_channels_2d
        self.in_channels_3d = in_channels_3d
        self.out_channels = out_channels
        self.model_channels_2d = model_channels_2d
        self.base_channels_3d = base_channels_3d
        self.num_classes = num_classes
        
        self.channel_mult_2d = channel_mult_2d
        self.dropout = dropout
        self.ksize = ksize
        self.dilations_2d = dilations_2d
        self.dilations_3d = dilations_3d
        self.use_label_cond = bool(use_label_cond) and (num_classes is not None)
        self.use_obs_time = bool(use_obs_time)
        self.pad_to_mult_of_32 = pad_to_mult_of_32
        self.num_var_head_res_blocks = int(num_var_head_res_blocks)

        # fixed z buffer
        self.register_buffer("fixed_z_coord_hpa", None, persistent=True)
        if fixed_z_coord_hpa is not None:
            z = torch.as_tensor(fixed_z_coord_hpa, dtype=torch.float32)
            if z.dim() == 1:
                z = z.view(1, -1)
            elif z.dim() == 2:
                if z.shape[0] != 1:
                    raise ValueError(f"fixed_z_coord_hpa as 2D must be (1,Z), got {z.shape}")
            else:
                raise ValueError(f"fixed_z_coord_hpa must be (Z,) or (1,Z), got {z.shape}")
            self.fixed_z_coord_hpa = z

        # -------------------------
        # 条件 embedding（无 sigma）
        # -------------------------
        time_embed_dim = model_channels_2d * 4
        self.time_embed_dim = time_embed_dim

        # base_emb：全局可学习“默认工作模式”
        self.base_emb = nn.Parameter(torch.zeros(time_embed_dim))
        self.base_ln = nn.LayerNorm(time_embed_dim)

        if self.use_obs_time:
            self.obs_time_mlp = nn.Sequential(
                nn.Linear(4, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.obs_time_ln = nn.LayerNorm(time_embed_dim)
        else:
            self.obs_time_mlp = None
            self.obs_time_ln = None

        self.use_init_time = True
        self.init_time_mlp = nn.Sequential(
            nn.Linear(4, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.init_time_ln = nn.LayerNorm(time_embed_dim)

        if self.use_label_cond:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.label_emb = None

        # -------------------------
        # joint encoder：条件联合编码
        # 槽位固定为 [base, lead, valid, init]
        # trunk 与 head 各自独立
        # -------------------------
        joint_in_dim = time_embed_dim * 4

        self.trunk_joint_mlp = nn.Sequential(
            nn.Linear(joint_in_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.trunk_joint_ln = nn.LayerNorm(time_embed_dim)

        self.head_joint_mlp = nn.Sequential(
            nn.Linear(joint_in_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        self.head_joint_ln = nn.LayerNorm(time_embed_dim)

        # -------------------------
        # 3D Encoder
        # -------------------------
        self.in3d = conv_nd(3, in_channels_3d, base_channels_3d, 3, padding=1)
        ch3 = base_channels_3d

        self.res3d_down1 = nn.ModuleList([
            ResBlockLarge(
                ch3, time_embed_dim, dropout,
                out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
            )
            for _ in range(num_res_blocks_3d)
        ])
        self.down3d_1 = Downsample(ch3, use_conv=True, dims=3, out_channels=ch3 * 2)
        ch3 *= 2

        self.res3d_down2 = nn.ModuleList([
            ResBlockLarge(
                ch3, time_embed_dim, dropout,
                out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
            )
            for _ in range(num_res_blocks_3d)
        ])
        self.down3d_2 = Downsample(ch3, use_conv=True, dims=3, out_channels=ch3 * 2)
        ch3 *= 2

        self.res3d_mid = nn.ModuleList([
            ResBlockLarge(
                ch3, time_embed_dim, dropout,
                out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
            )
            for _ in range(num_res_blocks_3d)
        ])

        self.attn3d = LevelTransformer(
            channels=ch3,
            num_layers=2,
            nhead=4,
            dim_feedforward=4 * ch3,
            max_z_levels=max_z_levels,
            use_film=True,
            p_min_hpa=0.1,
            p_max_hpa=1100.0,
            z_fourier_dim=ch3,
        ) if attn_3d else None

        self.up3d_1 = Upsample(ch3, use_conv=True, dims=3, out_channels=ch3 // 2)
        ch3 //= 2

        # 关键修复：
        # 第一个 block 输入是 concat 后的 2*ch3
        # 后续 block 输入是 ch3
        self.res3d_up1 = nn.ModuleList(
            [
                ResBlockLarge(
                    ch3 * 2, time_embed_dim, dropout,
                    out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
                )
            ] + [
                ResBlockLarge(
                    ch3, time_embed_dim, dropout,
                    out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
                )
                for _ in range(num_res_blocks_3d - 1)
            ]
        )

        self.up3d_2 = Upsample(ch3, use_conv=True, dims=3, out_channels=ch3 // 2)
        ch3 //= 2

        # 关键修复：
        # 第一个 block 输入是 concat 后的 2*ch3
        # 后续 block 输入是 ch3
        self.res3d_up2 = nn.ModuleList(
            [
                ResBlockLarge(
                    ch3 * 2, time_embed_dim, dropout,
                    out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
                )
            ] + [
                ResBlockLarge(
                    ch3, time_embed_dim, dropout,
                    out_channels=ch3, dims=3, ksize=ksize, dilations=dilations_3d
                )
                for _ in range(num_res_blocks_3d - 1)
            ]
        )

        self.out3d_proj = conv_nd(3, ch3, model_channels_2d, 1)
        self.z_pool = ZSpatialSoftmaxPool(model_channels_2d, use_checkpoint=False)

        # -------------------------
        # 2D U-Net trunk
        # -------------------------
        in_channels_total_2d = in_channels_2d + model_channels_2d
        ch = model_channels_2d * channel_mult_2d[0]

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(2, in_channels_total_2d, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        ds = 1
        self.attn_ds_2d: Set[int] = set(int(v) for v in attn_ds_2d)
        num_heads = 4
        num_head_channels = 32
        use_checkpoint = False

        for level, mult in enumerate(channel_mult_2d):
            for _ in range(num_res_blocks_2d):
                layers = [
                    ResBlockLarge(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels_2d),
                        dims=2,
                        ksize=ksize,
                        dilations=dilations_2d,
                    )
                ]
                ch = int(mult * model_channels_2d)
                if ds in self.attn_ds_2d:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                            use_new_attention_order=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult_2d) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, use_conv=True, dims=2, out_channels=ch))
                )
                ds *= 2
                input_block_chans.append(ch)

        self.middle_block = TimestepEmbedSequential(
            ResBlockLarge(ch, time_embed_dim, dropout, dims=2, ksize=ksize, dilations=dilations_2d),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_checkpoint=use_checkpoint,
                use_new_attention_order=False,
            ),
            ResBlockLarge(ch, time_embed_dim, dropout, dims=2, ksize=ksize, dilations=dilations_2d),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult_2d))[::-1]:
            for i in range(num_res_blocks_2d + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockLarge(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels_2d * mult),
                        dims=2,
                        ksize=ksize,
                        dilations=dilations_2d,
                    )
                ]
                ch = int(model_channels_2d * mult)
                if ds in self.attn_ds_2d:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_checkpoint=use_checkpoint,
                            use_new_attention_order=False,
                        )
                    )
                if level and i == num_res_blocks_2d:
                    layers.append(Upsample(ch, use_conv=True, dims=2, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out_norm = normalization(ch)
        self.out_act = nn.SiLU()
        hidden_out = ch

        # -------------------------
        # per-variable head：
        # 2 个 conditioned ResBlock + 最后 3x3 conv
        # -------------------------
        self.var_head_blocks = nn.ModuleList()
        self.var_heads = nn.ModuleList()

        for _ in range(out_channels):
            blocks = nn.ModuleList([
                ResBlockLarge(
                    hidden_out,
                    time_embed_dim,
                    dropout,
                    out_channels=hidden_out,
                    dims=2,
                    ksize=ksize,
                    dilations=dilations_2d,
                )
                for _ in range(self.num_var_head_res_blocks)
            ])
            self.var_head_blocks.append(blocks)
            self.var_heads.append(zero_module(nn.Conv2d(hidden_out, 1, 3, padding=1)))

    def _build_cond_components(self, B: int, device, y, obs_time, init_time=None):
        # base：始终存在
        base = self.base_ln(self.base_emb).to(device=device).unsqueeze(0).expand(B, -1)
        zero_like_base = torch.zeros_like(base)

        if self.use_label_cond and (self.label_emb is not None) and (y is not None):
            lead_emb = self.label_emb(y.to(device=device))
        else:
            lead_emb = zero_like_base

        if self.use_obs_time and (obs_time is not None) and (self.obs_time_mlp is not None):
            valid_feat = self.obs_time_ln(self.obs_time_mlp(obs_time.to(dtype=torch.float32, device=device)))
        else:
            valid_feat = zero_like_base

        if self.use_init_time and (init_time is not None) and (self.init_time_mlp is not None):
            init_feat = self.init_time_ln(self.init_time_mlp(init_time.to(dtype=torch.float32, device=device)))
        else:
            init_feat = zero_like_base

        return base, lead_emb, valid_feat, init_feat

    def _encode_all_cond(self, B: int, device, y, obs_time, init_time=None):
        """
        联合编码：
        trunk_cond = joint_encoder_trunk([base, lead, valid, init])
        head_cond  = joint_encoder_head([base, lead, valid, init])
        """
        base, lead_emb, valid_feat, init_feat = self._build_cond_components(
            B=B, device=device, y=y, obs_time=obs_time, init_time=init_time
        )

        trunk_cat = torch.cat([base, lead_emb, valid_feat, init_feat], dim=1)
        emb_trunk = self.trunk_joint_ln(self.trunk_joint_mlp(trunk_cat))

        head_cat = torch.cat([base, lead_emb, valid_feat, init_feat], dim=1)
        emb_head = self.head_joint_ln(self.head_joint_mlp(head_cat))

        return emb_trunk, emb_head

    def _apply_3d_encoder(self, forecast_3d, z_coord_hpa, emb_trunk):
        B, C3, Z, H, W = forecast_3d.shape
        assert z_coord_hpa.shape == (B, Z), f"z_coord_hpa must be (B,Z), got {z_coord_hpa.shape}"

        h = self.in3d(forecast_3d)

        h1 = h
        for blk in self.res3d_down1:
            h1 = blk(h1, emb_trunk)

        h2 = self.down3d_1(h1)
        for blk in self.res3d_down2:
            h2 = blk(h2, emb_trunk)

        h3 = self.down3d_2(h2)
        for blk in self.res3d_mid:
            h3 = blk(h3, emb_trunk)

        if self.attn3d is not None:
            h3 = self.attn3d(h3, z_coord_hpa=z_coord_hpa)

        hu1 = self.up3d_1(h3)
        hu1 = torch.cat([hu1, h2], dim=1)
        for blk in self.res3d_up1:
            hu1 = blk(hu1, emb_trunk)

        hu2 = self.up3d_2(hu1)
        hu2 = torch.cat([hu2, h1], dim=1)
        for blk in self.res3d_up2:
            hu2 = blk(hu2, emb_trunk)

        h_out = self.out3d_proj(hu2)  # (B, C2d, Z, H, W)
        feat_2d = self.z_pool(h_out)  # (B, C2d, H, W)
        return feat_2d

    def forward(
        self,
        cond_2d: torch.Tensor,
        forecast_3d: torch.Tensor,
        z_coord_hpa: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        obs_time: Optional[torch.Tensor] = None,
        init_time: Optional[torch.Tensor] = None,
        # 兼容旧接口：允许传 sigma，但会被忽略
        sigma: Optional[torch.Tensor] = None,
    ):
        assert cond_2d.dim() == 4, f"cond_2d must be BCHW, got {cond_2d.shape}"
        assert forecast_3d.dim() == 5, f"forecast_3d must be BCZHW, got {forecast_3d.shape}"

        B, _, H, W = cond_2d.shape
        Z = forecast_3d.shape[2]
        assert forecast_3d.shape[0] == B and forecast_3d.shape[-2:] == (H, W)

        # z_coord：优先用输入，否则用 fixed buffer
        if z_coord_hpa is None:
            if self.fixed_z_coord_hpa is None:
                raise ValueError("z_coord_hpa is None but model has no fixed_z_coord_hpa.")
            if self.fixed_z_coord_hpa.shape[1] != Z:
                raise ValueError(f"forecast_3d Z={Z} != fixed_z_coord_hpa Z={self.fixed_z_coord_hpa.shape[1]}")
            z_coord_hpa = self.fixed_z_coord_hpa.to(device=forecast_3d.device).expand(B, -1)
        else:
            assert z_coord_hpa.shape == (B, Z), f"z_coord_hpa must be (B,Z), got {z_coord_hpa.shape}"
            z_coord_hpa = z_coord_hpa.to(device=forecast_3d.device, dtype=torch.float32)

        emb_trunk, emb_head = self._encode_all_cond(
            B=B,
            device=forecast_3d.device,
            y=y,
            obs_time=obs_time,
            init_time=init_time,
        )

        feat3d_2d = self._apply_3d_encoder(forecast_3d, z_coord_hpa, emb_trunk)
        x = torch.cat([cond_2d, feat3d_2d], dim=1)

        # pad 到 32 倍数
        if self.pad_to_mult_of_32:
            pad_h = (-H) % 32
            pad_w = (-W) % 32
            if pad_h or pad_w:
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                x = F.pad(x, (left, right, top, bottom), mode="reflect")
                pad_info = (top, left, H, W)
            else:
                pad_info = (0, 0, H, W)
        else:
            pad_info = (0, 0, H, W)

        top, left, H0, W0 = pad_info

        hs: List[torch.Tensor] = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb_trunk)
            hs.append(h)

        h = self.middle_block(h, emb_trunk)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb_trunk)

        h = self.out_act(self.out_norm(h))

        if top or left or (h.size(-2) != H0) or (h.size(-1) != W0):
            h = h[..., top: top + H0, left: left + W0]

        outs = []
        for head_blocks, head_out in zip(self.var_head_blocks, self.var_heads):
            hh = h
            for blk in head_blocks:
                hh = blk(hh, emb_head)
            outs.append(head_out(hh))

        return torch.cat(outs, dim=1)