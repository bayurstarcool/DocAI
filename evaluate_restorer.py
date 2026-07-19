"""Evaluate DocumentRestorerNet on paired input/target folders.

Expected dataset layout:
  root/input/*
  root/target/*
Matched by relative path without extension.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from backend.datasets.restoration_dataset import PairedDocumentRestorationDataset
from backend.models.document_restorer import DocumentRestorerNet
from backend.utils.image_utils import run_document_restoration_pipeline


def ssim_channel(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    mu_x = F.avg_pool2d(x, 11, 1, 5)
    mu_y = F.avg_pool2d(y, 11, 1, 5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    return (((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))).mean()


def ssim_score(pred, target):
    scores = [ssim_channel(pred[:, channel:channel + 1], target[:, channel:channel + 1]) for channel in range(pred.shape[1])]
    return (sum(scores) / len(scores)).item()


def psnr_score(pred, target):
    mse = F.mse_loss(pred.clamp(0, 1), target.clamp(0, 1))
    return (10 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))).item()

def shadow_l1(pred, target, mask):
    weights = mask.clamp(0, 1)
    return (torch.abs(pred - target) * weights).sum().div((weights.sum() * pred.shape[1]).clamp_min(1.0)).item()

def non_shadow_l1(pred, source, mask):
    weights = (1.0 - mask).clamp(0, 1)
    return (torch.abs(pred - source) * weights).sum().div((weights.sum() * pred.shape[1]).clamp_min(1.0)).item()

def mask_scores(predicted_mask, target_mask, threshold=0.5):
    predicted = predicted_mask >= threshold
    target = target_mask >= threshold
    intersection = (predicted & target).sum().float()
    union = (predicted | target).sum().float()
    total = predicted.sum().float() + target.sum().float()
    dice = ((2 * intersection + 1) / (total + 1)).item()
    iou = ((intersection + 1) / (union + 1)).item()
    return dice, iou


def load_model(checkpoint_path, device, base_channels):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
    model = DocumentRestorerNet(base_channels=base_channels).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired-data', required=True, help='Folder with input/ and target/')
    parser.add_argument('--checkpoint', default='checkpoints/document_restorer/best.pth')
    parser.add_argument('--output', default='evaluation/document_restorer')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--max-samples', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--pipeline', action='store_true', help='Evaluate production postprocess pipeline previews in addition to raw model metrics')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device if args.device != 'auto' else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairedDocumentRestorationDataset(args.paired_data, size=args.size, augment=False)
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')
    model = load_model(args.checkpoint, device, args.base_channels)

    rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_l1 = 0.0
    total_shadow_l1 = 0.0
    total_identity_l1 = 0.0
    total_mask_mean = 0.0
    total_mask_dice = 0.0
    total_mask_iou = 0.0
    sample_count = 0
    preview_tensors = []
    pipeline_preview_count = 0

    with torch.inference_mode():
        for batch in loader:
            source = batch['input'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            restored, predicted_mask = model(source)
            batch_size = source.shape[0]
            for index in range(batch_size):
                pred_item = restored[index:index + 1]
                target_item = target[index:index + 1]
                source_item = source[index:index + 1]
                mask_item = mask[index:index + 1]
                psnr = psnr_score(pred_item, target_item)
                ssim = ssim_score(pred_item, target_item)
                l1 = F.l1_loss(pred_item, target_item).item()
                shadow = shadow_l1(pred_item, target_item, mask_item)
                identity = non_shadow_l1(pred_item, source_item, mask_item)
                predicted_mask_mean = predicted_mask[index].mean().item()
                mask_dice, mask_iou = mask_scores(predicted_mask[index:index + 1], mask_item)
                path = batch['path'][index] if isinstance(batch['path'], list) else str(batch['path'])
                rows.append({'path': path, 'psnr': psnr, 'ssim': ssim, 'l1': l1, 'shadow_l1': shadow, 'non_shadow_l1': identity, 'predicted_mask_mean': predicted_mask_mean, 'mask_dice': mask_dice, 'mask_iou': mask_iou})
                total_psnr += psnr
                total_ssim += ssim
                total_l1 += l1
                total_shadow_l1 += shadow
                total_identity_l1 += identity
                total_mask_mean += predicted_mask_mean
                total_mask_dice += mask_dice
                total_mask_iou += mask_iou
                sample_count += 1
                if len(preview_tensors) < 8:
                    preview_tensors.extend([
                        source[index].detach().cpu(),
                        restored[index].detach().cpu(),
                        target[index].detach().cpu(),
                        predicted_mask[index].repeat(3, 1, 1).detach().cpu(),
                    ])
                if args.pipeline and pipeline_preview_count < 8:
                    import torchvision.transforms.functional as TF
                    source_pil = TF.to_pil_image(source[index].detach().cpu().clamp(0, 1))
                    final, _, info = run_document_restoration_pipeline(model, source_pil, device)
                    final_tensor = TF.to_tensor(final)
                    save_image(final_tensor, output_dir / f'pipeline_{pipeline_preview_count:03d}.png')
                    (output_dir / f'pipeline_{pipeline_preview_count:03d}.json').write_text(json.dumps({'path': path, **info}, indent=2), encoding='utf-8')
                    pipeline_preview_count += 1

    summary = {
        'samples': sample_count,
        'psnr': total_psnr / max(sample_count, 1),
        'ssim': total_ssim / max(sample_count, 1),
        'l1': total_l1 / max(sample_count, 1),
        'shadow_l1': total_shadow_l1 / max(sample_count, 1),
        'non_shadow_l1': total_identity_l1 / max(sample_count, 1),
        'predicted_mask_mean': total_mask_mean / max(sample_count, 1),
        'mask_dice': total_mask_dice / max(sample_count, 1),
        'mask_iou': total_mask_iou / max(sample_count, 1),
        'checkpoint': str(args.checkpoint),
        'paired_data': str(args.paired_data),
        'size': args.size,
        'pipeline_previews': pipeline_preview_count,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'psnr', 'ssim', 'l1', 'shadow_l1', 'non_shadow_l1', 'predicted_mask_mean', 'mask_dice', 'mask_iou'])
        writer.writeheader()
        writer.writerows(rows)
    if preview_tensors:
        save_image(torch.stack(preview_tensors).clamp(0, 1), output_dir / 'preview_grid.png', nrow=4)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
