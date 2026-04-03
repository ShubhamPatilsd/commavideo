#!/usr/bin/env python
"""
Load a saved NeRV checkpoint and reconstruct all frames to a .raw file.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size
from submissions.nerv.model import NeRV


def inflate(src: str, dst: str, device: torch.device = torch.device('cpu'), batch_size: int = 16):
    ckpt = torch.load(src, map_location='cpu', weights_only=True)
    N = ckpt['num_frames']
    width = ckpt.get('width', 1)

    model = NeRV(hidden=256, num_frequencies=16, width=width)
    # fp16 → fp32 for inference (avoids precision issues on CPU)
    sd = {k: v.float() for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(sd)
    model.eval().to(device)

    W, H = camera_size  # (1164, 874)
    t_all = torch.linspace(0, 1, N, device=device)

    with open(dst, 'wb') as f:
        with torch.no_grad():
            for i in range(0, N, batch_size):
                t_batch = t_all[i:i + batch_size]
                rgb = model(t_batch, target_size=(H, W))   # (B, 3, H, W) in [0,1]
                rgb = (rgb * 255).round().clamp(0, 255).to(torch.uint8)
                rgb = rgb.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 3)
                f.write(rgb.cpu().numpy().tobytes())

    print(f"saved {N} frames → {dst}")
    return N


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    device_str = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    inflate(src, dst, device=torch.device(device_str))
