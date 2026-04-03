import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoding(nn.Module):
    """Maps scalar t ∈ [0,1] to sinusoidal features so the network can represent high-frequency detail."""
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        freqs = 2.0 ** torch.arange(num_frequencies).float()
        self.register_buffer('freqs', freqs)
        self.out_dim = 2 * num_frequencies + 1  # sin + cos + raw t

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        t_exp = t.unsqueeze(-1) * self.freqs * math.pi  # (B, F)
        return torch.cat([t.unsqueeze(-1), torch.sin(t_exp), torch.cos(t_exp)], dim=-1)


class NeRVBlock(nn.Module):
    """Conv → PixelShuffle (sub-pixel upsample by `scale`) → GELU."""
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ps(self.conv(x)))


# Decoder channel progression (index 0 = base, each step halves channels after upsampling)
_DECODER_CHANNELS = [256, 128, 64, 64, 32, 16, 8, 4]


class NeRV(nn.Module):
    """
    Neural Representations for Videos.

    Architecture:
      t → Fourier encoding → MLP stem → reshape to (C, 3, 4) spatial grid
      → 7× NeRVBlock (each 2× upsample) → (3, 384, 512) → bicubic to target size

    width multiplier controls model capacity:
      width=1 → ~2.6M params, ~5MB fp16
      width=2 → ~10M params, ~20MB fp16
      width=4 → ~41M params, ~82MB fp16
    """
    BASE_H = 3
    BASE_W = 4

    def __init__(self, hidden: int = 256, num_frequencies: int = 16, width: int = 1):
        super().__init__()
        self.enc = FourierEncoding(num_frequencies)
        enc_dim = self.enc.out_dim

        ch = [c * width for c in _DECODER_CHANNELS]  # scaled channel sizes
        self.base_c = ch[0]
        base_flat = ch[0] * self.BASE_H * self.BASE_W

        self.stem = nn.Sequential(
            nn.Linear(enc_dim, hidden * width),
            nn.GELU(),
            nn.Linear(hidden * width, hidden * width),
            nn.GELU(),
            nn.Linear(hidden * width, base_flat),
        )

        # (ch[0], 3, 4) → upsample × 7 → (3, 384, 512)
        self.decoder = nn.Sequential(
            NeRVBlock(ch[0], ch[1], 2),
            NeRVBlock(ch[1], ch[2], 2),
            NeRVBlock(ch[2], ch[3], 2),
            NeRVBlock(ch[3], ch[4], 2),
            NeRVBlock(ch[4], ch[5], 2),
            NeRVBlock(ch[5], ch[6], 2),
            NeRVBlock(ch[6], ch[7], 2),
            nn.Conv2d(ch[7], 3, kernel_size=1),
        )

    def forward(self, t: torch.Tensor, target_size: tuple[int, int] = (874, 1164)) -> torch.Tensor:
        """
        Args:
            t: (B,) frame indices normalized to [0, 1]
            target_size: (H, W) output resolution
        Returns:
            (B, 3, H, W) float32 in [0, 1]
        """
        enc = self.enc(t)
        feat = self.stem(enc).view(-1, self.base_c, self.BASE_H, self.BASE_W)
        rgb = torch.sigmoid(self.decoder(feat))  # (B, 3, 384, 512)
        if rgb.shape[-2:] != target_size:
            rgb = F.interpolate(rgb, size=target_size, mode='bicubic', align_corners=False).clamp(0, 1)
        return rgb
