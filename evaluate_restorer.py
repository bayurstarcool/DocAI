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


def load_model(checkpoint_path, device, base_channels):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get('model') or checkpoint.get('model_state_dict') or checkpoint
    model = DocumentRestorerNet(base_channels=base_channels).to(device)
    model.load_state_dict(model_state, strict=False)
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
    sample_count = 0
    preview_tensors = []

    with torch.inference_mode():
        for batch in loader:
            source = batch['input'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            restored, predicted_mask = model(source)
            batch_size = source.shape[0]
            for index in range(batch_size):
                pred_item = restored[index:index + 1]
                target_item = target[index:index + 1]
                psnr = psnr_score(pred_item, target_item)
                ssim = ssim_score(pred_item, target_item)
                l1 = F.l1_loss(pred_item, target_item).item()
                path = batch['path'][index] if isinstance(batch['path'], list) else str(batch['path'])
                rows.append({'path': path, 'psnr': psnr, 'ssim': ssim, 'l1': l1})
                total_psnr += psnr
                total_ssim += ssim
                total_l1 += l1
                sample_count += 1
                if len(preview_tensors) < 8:
                    preview_tensors.extend([
                        source[index].detach().cpu(),
                        restored[index].detach().cpu(),
                        target[index].detach().cpu(),
                        predicted_mask[index].repeat(3, 1, 1).detach().cpu(),
                    ])

    summary = {
        'samples': sample_count,
        'psnr': total_psnr / max(sample_count, 1),
        'ssim': total_ssim / max(sample_count, 1),
        'l1': total_l1 / max(sample_count, 1),
        'checkpoint': str(args.checkpoint),
        'paired_data': str(args.paired_data),
        'size': args.size,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'psnr', 'ssim', 'l1'])
        writer.writeheader()
        writer.writerows(rows)
    if preview_tensors:
        save_image(torch.stack(preview_tensors).clamp(0, 1), output_dir / 'preview_grid.png', nrow=4)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
