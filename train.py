"""Train custom AI for document shadow removal and restoration.

v2 changes:
- Perceptual loss (VGG-16 features) for better visual quality
- SSIM loss for structural similarity
- Better loss weighting
"""

import argparse
import os
import shutil
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
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

def dataset_sampling_weights(dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        parent_weights = dataset_sampling_weights(dataset.dataset)
        return parent_weights[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if isinstance(dataset, ConcatDataset):
        weights = []
        for child in dataset.datasets:
            child_weights = dataset_sampling_weights(child)
            weights.append(child_weights / child_weights.sum().clamp_min(1e-12))
        return torch.cat(weights)
    return torch.ones(len(dataset), dtype=torch.double)


def save_validation_preview(output_dir: Path, epoch: int, source: torch.Tensor, restored: torch.Tensor, target: torch.Tensor,
                            target_mask: torch.Tensor, predicted_mask: torch.Tensor):
    preview_dir = output_dir / 'previews'
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(4, source.shape[0])
    target_mask_rgb = target_mask[:count].repeat(1, 3, 1, 1)
    predicted_mask_rgb = predicted_mask[:count].repeat(1, 3, 1, 1)
    rows = torch.cat((
        source[:count].cpu(),
        restored[:count].cpu(),
        target[:count].cpu(),
        target_mask_rgb.cpu(),
        predicted_mask_rgb.cpu(),
    ), dim=0)
    save_image(rows.clamp(0, 1), preview_dir / f'epoch_{epoch:04d}.png', nrow=count)
    stats = {
        'epoch': epoch,
        'target_mask_mean': float(target_mask[:count].mean().detach().cpu()),
        'predicted_mask_mean': float(predicted_mask[:count].mean().detach().cpu()),
        'shadow_pixel_ratio_target_gt_0_1': float((target_mask[:count] > 0.1).float().mean().detach().cpu()),
        'shadow_pixel_ratio_pred_gt_0_1': float((predicted_mask[:count] > 0.1).float().mean().detach().cpu()),
        'restored_mean': float(restored[:count].mean().detach().cpu()),
        'target_mean': float(target[:count].mean().detach().cpu()),
    }
    (preview_dir / f'epoch_{epoch:04d}.json').write_text(json.dumps(stats, indent=2))

def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def publish_checkpoint(source: Path, publish_dir: Path, name: str):
    publish_dir.mkdir(parents=True, exist_ok=True)
    temporary = publish_dir / f'.{name}.{os.getpid()}.tmp'
    shutil.copyfile(source, temporary)
    os.replace(temporary, publish_dir / name)


# ============================================================
# Training utilities
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-data', nargs='+', default=['datasets/Document_Enhancement/train'])
    parser.add_argument('--paired-data', nargs='*', default=[
        'datasets/paired/cvpr-2023-rdd',
        'datasets/paired/jungs-dataset',
        'datasets/paired/kliglers-dataset',
        'datasets/ShadowDocument7K',
    ])
    parser.add_argument('--validation-paired-data', nargs='*', default=[])
    parser.add_argument('--identity-data', nargs='*', default=['data/identity', 'datasets/identity'])
    parser.add_argument('--output', default='checkpoints/document_restorer')
    parser.add_argument('--publish-output', help='Optional stable directory receiving atomic live checkpoint aliases')
    parser.add_argument('--run-id', default='manual')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume')
    parser.add_argument('--resume-weights-only', action='store_true',
                        help='Load model weights from --resume but start a fresh optimizer/scheduler run')
    parser.add_argument('--early-stop-patience', type=int, default=7,
                        help='Stop after this many epochs without validation improvement. Use 0 to disable')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                        help='Minimum validation-loss improvement for best checkpoint and early stopping')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                        help='Gradient clipping max norm. Use 0 to disable')
    parser.add_argument('--max-train-samples', type=int)
    parser.add_argument('--max-validation-samples', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--perceptual-weight', type=float, default=0.05,
                        help='Weight for VGG perceptual loss')
    parser.add_argument('--ssim-weight', type=float, default=0.1,
                        help='Weight for SSIM loss')
    parser.add_argument('--shadow-loss-weight', type=float, default=1.8,
                        help='Weight for shadow-region reconstruction loss')
    parser.add_argument('--illumination-weight', type=float, default=0.55,
                        help='Weight for shadow-region luminance recovery')
    parser.add_argument('--mask-loss-weight', type=float, default=0.35,
                        help='Weight for shadow mask supervision')
    parser.add_argument('--gradient-weight', type=float, default=0.15,
                        help='Weight for edge/gradient consistency loss')
    parser.add_argument('--color-weight', type=float, default=0.08,
                        help='Weight for global color consistency loss')
    parser.add_argument('--identity-weight', type=float, default=0.12,
                        help='Weight for non-shadow identity preservation')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Linear LR warmup epochs before cosine decay')
    return parser.parse_args()


def total_variation(image):
    return (image[:, :, 1:] - image[:, :, :-1]).abs().mean() + \
           (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - target) * weight).sum() / (weight.sum() * pred.shape[1]).clamp_min(1.0)

def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


class LossComputer:
    """Computes all loss terms and returns a weighted sum."""

    def __init__(self, perceptual_weight=0.05, ssim_weight=0.1, shadow_loss_weight=1.25,
                 illumination_weight=0.35, mask_loss_weight=0.2, gradient_weight=0.15,
                 color_weight=0.08, identity_weight=0.25, device='cpu'):
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.shadow_loss_weight = shadow_loss_weight
        self.illumination_weight = illumination_weight
        self.mask_loss_weight = mask_loss_weight
        self.gradient_weight = gradient_weight
        self.color_weight = color_weight
        self.identity_weight = identity_weight
        self.perceptual = PerceptualLoss().to(device)

    def __call__(self, restored, predicted_mask, target, mask, include_tv, source=None):
        # 1. Pixel-level reconstruction
        reconstruction = F.l1_loss(restored, target)

        # 2. Shadow-weighted L1 (penalize shadow regions harder)
        shadow_weight = 1.0 + mask * 8.0
        shadow_loss = weighted_l1(restored, target, shadow_weight)

        # 3. Luminance recovery in shadow regions
        luminance_weights = restored.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        restored_luma = (restored * luminance_weights).sum(dim=1, keepdim=True)
        target_luma = (target * luminance_weights).sum(dim=1, keepdim=True)
        shadow_pixels = mask.clamp(0, 1).sum().clamp_min(1.0)
        illumination_loss = (torch.abs(restored_luma - target_luma) * mask).sum() / shadow_pixels

        # 4. Mask prediction loss
        with autocast(device_type=predicted_mask.device.type, enabled=False):
            target_mask = mask.float().clamp(0, 1)
            positive_ratio = target_mask.mean().clamp(0.01, 0.75)
            pos_weight = ((1.0 - positive_ratio) / positive_ratio).clamp(1.0, 12.0)
            mask_loss = F.binary_cross_entropy(
                predicted_mask.float().clamp(1e-4, 1 - 1e-4),
                target_mask,
                weight=(1.0 - target_mask) + target_mask * pos_weight,
            ) + dice_loss(predicted_mask.float(), target_mask)

        mask_tv = total_variation(predicted_mask) if include_tv else torch.tensor(0.0, device=restored.device)

        # 5. Perceptual (VGG) loss — visual quality
        perceptual = self.perceptual(restored, target) if self.perceptual_weight > 0 else torch.tensor(0.0, device=restored.device)

        # 6. SSIM loss — structural similarity
        ssim = ssim_loss(restored, target) if self.ssim_weight > 0 else torch.tensor(0.0, device=restored.device)

        # 7. Total variation — smoothness
        tv = total_variation(restored) if include_tv else torch.tensor(0.0, device=restored.device)

        gradients = gradient_loss(restored, target) if self.gradient_weight > 0 else torch.tensor(0.0, device=restored.device)
        color_consistency = F.l1_loss(restored.mean(dim=(2, 3)), target.mean(dim=(2, 3))) if self.color_weight > 0 else torch.tensor(0.0, device=restored.device)
        if source is not None and self.identity_weight > 0:
            non_shadow = (1.0 - mask).clamp(0, 1)
            identity = weighted_l1(restored, source, non_shadow)
        else:
            identity = torch.tensor(0.0, device=restored.device)

        total = (reconstruction
                 + self.shadow_loss_weight * shadow_loss
                 + self.illumination_weight * illumination_loss
                 + self.mask_loss_weight * mask_loss
                 + self.perceptual_weight * perceptual
                 + self.ssim_weight * ssim
                 + self.gradient_weight * gradients
                 + self.color_weight * color_consistency
                 + self.identity_weight * identity
                 + 0.02 * tv
                 + 0.01 * mask_tv)

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
    publish_dir = Path(args.publish_output) if args.publish_output else None
    run_manifest = {
        'run_id': args.run_id,
        'status': 'running',
        'started_at': time.time(),
        'output': str(output_dir),
        'publish_output': str(publish_dir) if publish_dir else None,
        'args': vars(args),
    }
    (output_dir / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')

    # ---- Data ----
    clean_roots = [root for root in args.clean_data if Path(root).exists()]
    paired_roots = [Path(root) for root in args.paired_data if Path(root).exists()]
    identity_roots = [root for root in args.identity_data if Path(root).exists()]
    if not clean_roots and not paired_roots and not identity_roots:
        raise RuntimeError('No training dataset found. Provide --paired-data, --clean-data, or --identity-data with existing image folders.')

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

    full_train = build_restoration_dataset(clean_roots, train_paired_roots, args.size, augment=True, identity_roots=identity_roots)
    if validation_paired_roots:
        validation_set = build_restoration_dataset([], validation_paired_roots, args.size, augment=False)
        train_set = full_train
    else:
        if len(full_train) < 2:
            raise RuntimeError('Need at least two training pairs.')
        full_validation = build_restoration_dataset(clean_roots, train_paired_roots, args.size, augment=False, identity_roots=identity_roots)
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

    sampling_weights = dataset_sampling_weights(train_set)
    train_sampler = WeightedRandomSampler(
        sampling_weights,
        num_samples=len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=device.type == 'cuda',
                              persistent_workers=args.workers > 0)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == 'cuda',
                                   persistent_workers=args.workers > 0)

    # ---- Model ----
    model = DocumentRestorerNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')
    loss_fn = LossComputer(
        perceptual_weight=args.perceptual_weight,
        ssim_weight=args.ssim_weight,
        shadow_loss_weight=args.shadow_loss_weight,
        illumination_weight=args.illumination_weight,
        mask_loss_weight=args.mask_loss_weight,
        gradient_weight=args.gradient_weight,
        color_weight=args.color_weight,
        identity_weight=args.identity_weight,
        device=device,
    )

    start_epoch = 0
    best_loss = float('inf')
    best_psnr = float('-inf')
    best_ssim = float('-inf')
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': [], 'val_ssim': []}

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model_state = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
        model.load_state_dict(model_state, strict=True)
        if args.resume_weights_only:
            print(f'Loaded weights from {args.resume}; starting fresh fine-tune run')
        else:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                except (KeyError, ValueError, TypeError) as error:
                    print(f'Warning: scheduler state incompatible with current epoch configuration; using fresh scheduler ({error})')
            elif 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except (KeyError, ValueError, TypeError) as error:
                    print(f'Warning: scheduler state incompatible with current epoch configuration; using fresh scheduler ({error})')
            start_epoch = checkpoint.get('epoch', -1) + 1
            best_loss = checkpoint.get('best_loss', checkpoint.get('best_val_loss', best_loss))
            best_psnr = checkpoint.get('best_psnr', best_psnr)
            best_ssim = checkpoint.get('best_ssim', best_ssim)
            history = checkpoint.get('history', history)
            print(f'Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}')

    print(f'train_pairs={len(train_set)} validation_pairs={len(validation_set)} device={device}')
    print(f'train_paired_roots={[str(root) for root in train_paired_roots]}')
    print(f'validation_paired_roots={[str(root) for root in validation_paired_roots]}')
    print(f'paired_roots={len(train_paired_roots)} clean_roots={len(clean_roots)} identity_roots={len(identity_roots)}')
    print('sampling=balanced_by_dataset_root replacement=true')
    print(f'perceptual_weight={args.perceptual_weight} ssim_weight={args.ssim_weight} lr={args.lr}')

    epoch_times = []
    epochs_without_improvement = 0
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
                loss, _, _ = loss_fn(restored, predicted_mask, target, mask, include_tv=True, source=source)

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
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
        validation_samples = 0
        preview_saved = False
        with torch.inference_mode():
            for batch in validation_loader:
                source = batch['input'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                restored, predicted_mask = model(source)
                batch_size = source.shape[0]
                validation_loss += loss_fn(restored, predicted_mask, target, mask, include_tv=False, source=source)[0].item() * batch_size
                for sample_index in range(batch_size):
                    validation_psnr += psnr_score(
                        restored[sample_index:sample_index + 1], target[sample_index:sample_index + 1]
                    ).item()
                    validation_ssim += ssim_score(
                        restored[sample_index:sample_index + 1], target[sample_index:sample_index + 1]
                    ).item()
                validation_samples += batch_size
                if not preview_saved:
                    save_validation_preview(output_dir, epoch + 1, source, restored, target, mask, predicted_mask)
                    preview_saved = True

        validation_loss /= validation_samples
        validation_psnr /= validation_samples
        validation_ssim /= validation_samples
        scheduler.step()

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(validation_loss)
        history['val_psnr'].append(validation_psnr)
        history['val_ssim'].append(validation_ssim)
        (output_dir / 'training_history.json').write_text(json.dumps(history), encoding='utf-8')

        improved = validation_loss < best_loss - args.min_delta
        psnr_improved = validation_psnr > best_psnr
        ssim_improved = validation_ssim > best_ssim
        if improved:
            best_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        best_psnr = max(best_psnr, validation_psnr)
        best_ssim = max(best_ssim, validation_ssim)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'history': history,
            'args': vars(args),
        }
        atomic_torch_save(checkpoint, output_dir / 'last.pth')
        if publish_dir:
            publish_checkpoint(output_dir / 'last.pth', publish_dir, 'last.pth')

        if improved:
            atomic_torch_save(checkpoint, output_dir / 'best.pth')
            atomic_torch_save(checkpoint, output_dir / 'best_loss.pth')
            if publish_dir:
                publish_checkpoint(output_dir / 'best.pth', publish_dir, 'best.pth')
                publish_checkpoint(output_dir / 'best_loss.pth', publish_dir, 'best_loss.pth')
        if psnr_improved:
            atomic_torch_save(checkpoint, output_dir / 'best_psnr.pth')
            if publish_dir:
                publish_checkpoint(output_dir / 'best_psnr.pth', publish_dir, 'best_psnr.pth')
        if ssim_improved:
            atomic_torch_save(checkpoint, output_dir / 'best_ssim.pth')
            if publish_dir:
                publish_checkpoint(output_dir / 'best_ssim.pth', publish_dir, 'best_ssim.pth')

        run_manifest.update({
            'status': 'running',
            'updated_at': time.time(),
            'epoch': epoch + 1,
            'best_loss': best_loss,
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'latest_preview': f'previews/epoch_{epoch + 1:04d}.png',
        })
        (output_dir / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')

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
            f'no_improve={epochs_without_improvement} epoch_time={epoch_elapsed:.1f}s eta={eta_str}'
        )

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f'early_stop epoch={epoch + 1} patience={args.early_stop_patience} best_loss={best_loss:.4f}')
            break

    run_manifest.update({'status': 'completed', 'finished_at': time.time()})
    (output_dir / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
