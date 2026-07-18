"""Train custom AI for document shadow removal and restoration.

v2 changes:
- Perceptual loss (VGG-16 features) for better visual quality
- SSIM loss for structural similarity
- Better loss weighting
"""

import argparse
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import models
from torchvision.utils import save_image

from backend.datasets.restoration_dataset import build_restoration_dataset, sd7k_split_paths
from backend.models.document_restorer import DocumentRestorerNet


# ============================================================
# Perceptual Loss (VGG-16)
# ============================================================
class PerceptualLoss(nn.Module):
    """Compares high-level VGG features instead of raw pixels."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # relu1_2=3, relu2_2=8, relu3_3=15
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9]).eval()
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _normalize(self, x):
        return (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._normalize(pred)
        target = self._normalize(target)
        f1 = self.slice1(pred);  t1 = self.slice1(target)
        f2 = self.slice2(f1);    t2 = self.slice2(t1)
        f3 = self.slice3(f2);    t3 = self.slice3(t2)
        return F.l1_loss(f1, t1) + F.l1_loss(f2, t2) + F.l1_loss(f3, t3)


# ============================================================
# SSIM (structural similarity)
# ============================================================
def _ssim_channel(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, so lower is better."""
    channels = pred.shape[1]
    if channels == 1:
        return 1.0 - _ssim_channel(pred, target)
    parts = []
    for c in range(channels):
        parts.append(1.0 - _ssim_channel(pred[:, c:c+1], target[:, c:c+1]))
    return sum(parts) / channels


def psnr_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return 10 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))


def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim_loss(pred.clamp(0, 1), target.clamp(0, 1))


def save_validation_preview(output_dir: Path, epoch: int, source: torch.Tensor, restored: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    preview_dir = output_dir / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(4, source.shape[0])
    mask_rgb = mask[:count].repeat(1, 3, 1, 1)
    rows = torch.cat((source[:count].cpu(), restored[:count].cpu(), target[:count].cpu(), mask_rgb.cpu()), dim=0)
    save_image(rows.clamp(0, 1), preview_dir / f'epoch_{epoch:04d}.png', nrow=count)


# ============================================================
# Training utilities
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-data', nargs='+', default=['datasets/Document_Enhancement/train'])
    parser.add_argument('--paired-data', nargs='*', default=['datasets/ShadowDocument7K'])
    parser.add_argument('--validation-paired-data', nargs='*', default=[])
    parser.add_argument('--output', default='checkpoints/document_restorer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-validation-samples', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--perceptual-weight', type=float, default=0.05,
                        help='Weight for VGG perceptual loss')
    parser.add_argument('--ssim-weight', type=float, default=0.1,
                        help='Weight for SSIM loss')
    return parser.parse_args()


def total_variation(image):
    return (image[:, :, 1:] - image[:, :, :-1]).abs().mean() + \
           (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()


class LossComputer:
    """Computes all loss terms and returns a weighted sum."""

    def __init__(self, perceptual_weight=0.05, ssim_weight=0.1, device='cpu'):
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.perceptual = PerceptualLoss().to(device)

    def __call__(self, restored, predicted_mask, target, mask, include_tv):
        # 1. Pixel-level reconstruction
        reconstruction = F.smooth_l1_loss(restored, target)

        # 2. Shadow-weighted L1 (penalize shadow regions harder)
        shadow_loss = (torch.abs(restored - target) * (1 + mask * 2)).mean()

        # 3. Mask prediction loss
        with autocast(device_type=predicted_mask.device.type, enabled=False):
            mask_loss = F.binary_cross_entropy(predicted_mask.float(), mask.float())

        # 4. Perceptual (VGG) loss — visual quality
        perceptual = self.perceptual(restored, target) if self.perceptual_weight > 0 else torch.tensor(0.0, device=restored.device)

        # 5. SSIM loss — structural similarity
        ssim = ssim_loss(restored, target) if self.ssim_weight > 0 else torch.tensor(0.0, device=restored.device)

        # 6. Total variation — smoothness
        tv = total_variation(restored) if include_tv else torch.tensor(0.0, device=restored.device)

        total = (reconstruction
                 + 0.5 * shadow_loss
                 + 0.1 * mask_loss
                 + self.perceptual_weight * perceptual
                 + self.ssim_weight * ssim
                 + 0.02 * tv)

        return total, reconstruction, shadow_loss


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')

    device = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto'
        else 'cpu'
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    clean_roots = [root for root in args.clean_data if Path(root).exists()]
    paired_roots = [Path(root) for root in args.paired_data if Path(root).exists()]
    if not clean_roots and not paired_roots:
        raise RuntimeError('No training dataset found. Provide --paired-data or --clean-data with existing image folders.')

    train_paired_roots = []
    official_validation_roots = []
    for root in paired_roots:
        splits = sd7k_split_paths(root)
        if splits:
            train_root, test_root = splits
            train_paired_roots.append(train_root)
            official_validation_roots.append(test_root)
        else:
            train_paired_roots.append(root)

    requested_validation_roots = [Path(root) for root in args.validation_paired_data if Path(root).exists()]
    validation_paired_roots = requested_validation_roots or official_validation_roots

    full_train = build_restoration_dataset(clean_roots, train_paired_roots, args.size, augment=True)
    if validation_paired_roots:
        validation_set = build_restoration_dataset([], validation_paired_roots, args.size, augment=False)
        train_set = full_train
    else:
        if len(full_train) < 2:
            raise RuntimeError('Need at least two training pairs.')
        full_validation = build_restoration_dataset(clean_roots, train_paired_roots, args.size, augment=False)
        indices = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(args.seed)).tolist()
        validation_size = max(1, round(len(indices) * 0.1))
        train_set = Subset(full_train, indices[validation_size:])
        validation_set = Subset(full_validation, indices[:validation_size])

    if args.max_train_samples is not None:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    if args.max_validation_samples is not None:
        validation_set = Subset(validation_set, range(min(args.max_validation_samples, len(validation_set))))

    if len(train_set) == 0 or len(validation_set) == 0:
        raise RuntimeError('Training and validation sets must both contain at least one sample.')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == 'cuda',
                              persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == 'cuda',
                                   persistent_workers=args.workers > 0)

    # ---- Model ----
    model = DocumentRestorerNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = LossComputer(args.perceptual_weight, args.ssim_weight, device)

    start_epoch = 0
    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f'Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}')

    print(f'train_pairs={len(train_set)} validation_pairs={len(validation_set)} device={device}')
    print(f'perceptual_weight={args.perceptual_weight} ssim_weight={args.ssim_weight}')

    epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            source = batch['input'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == 'cuda'):
                restored, predicted_mask = model(source)
                loss, _, _ = loss_fn(restored, predicted_mask, target, mask, include_tv=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if batch_index % args.log_interval == 0 or batch_index == len(train_loader):
                print(
                    f'epoch={epoch + 1}/{args.epochs} batch={batch_index}/{len(train_loader)} '
                    f'train_loss={train_loss / batch_index:.4f}',
                    flush=True,
                )

        # ---- Validation ----
        model.eval()
        validation_loss = 0.0
        validation_psnr = 0.0
        validation_ssim = 0.0
        preview_saved = False
        with torch.inference_mode():
            for batch in validation_loader:
                source = batch['input'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                restored, predicted_mask = model(source)
                validation_loss += loss_fn(restored, predicted_mask, target, mask, include_tv=False)[0].item()
                validation_psnr += psnr_score(restored, target).item()
                validation_ssim += ssim_score(restored, target).item()
                if not preview_saved:
                    save_validation_preview(output_dir, epoch + 1, source, restored, target, predicted_mask)
                    preview_saved = True

        validation_loss /= len(validation_loader)
        validation_psnr /= len(validation_loader)
        validation_ssim /= len(validation_loader)
        scheduler.step()

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(validation_loss)
        history['val_psnr'].append(validation_psnr)
        history['val_ssim'].append(validation_ssim)
        (output_dir / 'training_history.json').write_text(json.dumps(history), encoding='utf-8')

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'args': vars(args),
        }
        torch.save(checkpoint, output_dir / 'last.pth')

        if validation_loss < best_loss:
            best_loss = validation_loss
            checkpoint['best_loss'] = best_loss
            torch.save(checkpoint, output_dir / 'best.pth')

        epoch_elapsed = time.time() - epoch_start_time
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - epoch - 1
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_h = int(eta_seconds // 3600)
        eta_m = int((eta_seconds % 3600) // 60)
        eta_s = int(eta_seconds % 60)
        eta_str = f'{eta_h}h {eta_m}m {eta_s}s'

        print(
            f'epoch={epoch + 1}/{args.epochs} '
            f'train_loss={train_loss / len(train_loader):.4f} '
            f'val_loss={validation_loss:.4f} val_psnr={validation_psnr:.2f} val_ssim={validation_ssim:.4f} best={best_loss:.4f} '
            f'epoch_time={epoch_elapsed:.1f}s eta={eta_str}'
        )


if __name__ == '__main__':
    main()
