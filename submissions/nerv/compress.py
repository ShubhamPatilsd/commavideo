#!/usr/bin/env python
"""
Train a NeRV model on a video and save quantized weights to archive/.
"""
import sys, argparse
from pathlib import Path

import av
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import yuv420_to_rgb
from submissions.nerv.model import NeRV

WIDTH_PRESETS = {'small': 1, 'medium': 2, 'large': 4}

# Train at the model's native output resolution — eliminates the resolution gap.
# Stored as fp16 → ~1.4 GB RAM for 1200 frames, which is fine.
TRAIN_H, TRAIN_W = 384, 512


def load_frames(video_path: str) -> torch.Tensor:
    """
    Decode all frames, downsample to (TRAIN_H, TRAIN_W), store as fp16.
    Returns (N, 3, H, W) float16 in [0, 1].
    """
    frames = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame).float() / 255.0       # (H, W, 3) fp32
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)           # (1, 3, H, W)
        rgb = F.interpolate(rgb, size=(TRAIN_H, TRAIN_W), mode='bilinear', align_corners=False)
        frames.append(rgb.squeeze(0).half())               # fp16 to save RAM
    container.close()
    return torch.stack(frames)  # (N, 3, TRAIN_H, TRAIN_W) fp16


def train(frames: torch.Tensor, epochs: int, batch_size: int, lr: float, device: torch.device, width: int = 1) -> NeRV:
    N = len(frames)
    frames = frames.to(device)  # stays fp16 on device

    model = NeRV(hidden=256, num_frequencies=16, width=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    t_all = torch.linspace(0, 1, N, device=device)
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params  fp16 weights: {n_params * 2 / 1e6:.1f} MB  amp={use_amp}")

    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        total_loss, steps = 0.0, 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            t_batch = t_all[idx]
            gt = frames[idx].float()  # (B, 3, TRAIN_H, TRAIN_W) — fp32 for loss

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(t_batch, target_size=(TRAIN_H, TRAIN_W))  # (B, 3, TRAIN_H, TRAIN_W)
                pixel_loss = F.l1_loss(pred, gt)

                # Temporal loss: penalise difference between consecutive predicted frames.
                # Forces the model to learn motion, not just per-frame averages.
                if len(idx) >= 2:
                    t_sorted = t_all[torch.sort(idx).values]
                    pred_seq = model(t_sorted, target_size=(TRAIN_H, TRAIN_W))
                    gt_seq = frames[torch.sort(idx).values].float()
                    delta_pred = pred_seq[1:] - pred_seq[:-1]
                    delta_gt   = gt_seq[1:]   - gt_seq[:-1]
                    temporal_loss = F.l1_loss(delta_pred, delta_gt)
                else:
                    temporal_loss = torch.zeros([], device=device)

                loss = pixel_loss + 0.1 * temporal_loss

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += pixel_loss.item()
            steps += 1

        scheduler.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  pixel_loss={total_loss/steps:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default=str(ROOT / 'videos'))
    parser.add_argument('--video-names-file', default=str(ROOT / 'public_test_video_names.txt'))
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-size', default='small', choices=list(WIDTH_PRESETS))
    args = parser.parse_args()

    device = torch.device(args.device)
    archive_dir = HERE / 'archive'
    archive_dir.mkdir(parents=True, exist_ok=True)

    with open(args.video_names_file) as f:
        video_names = [l.strip() for l in f if l.strip()]

    for rel in video_names:
        video_path = Path(args.in_dir) / rel
        base = Path(rel).stem
        out_path = archive_dir / f"{base}.pt"

        print(f"\n=== {rel} ===")
        print(f"  Loading and downsampling frames to {TRAIN_H}×{TRAIN_W} (fp16)...")
        frames = load_frames(str(video_path))
        N = len(frames)
        print(f"  {N} frames  RAM: {frames.numel() * 2 / 1e6:.0f} MB")

        width = WIDTH_PRESETS[args.model_size]
        model = train(frames, args.epochs, args.batch_size, args.lr, device, width=width)

        sd_fp16 = {k: v.half().cpu() for k, v in model.state_dict().items()}
        torch.save({'state_dict': sd_fp16, 'num_frames': N, 'width': width}, str(out_path))

        sz = out_path.stat().st_size
        print(f"  Saved {out_path.name}: {sz / 1e6:.2f} MB")


if __name__ == '__main__':
    main()
