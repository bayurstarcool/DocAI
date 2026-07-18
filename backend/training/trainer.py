"""
Model Trainer with support for multiple losses and metrics
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            self.features = nn.Sequential(*list(vgg.children())).eval()
            for p in self.features.parameters():
                p.requires_grad = False
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        except Exception:
            self.features = None
            self.normalize = None

    def forward(self, x, y):
        if self.features is None:
            return nn.functional.mse_loss(x, y)
        x = self.normalize(x)
        y = self.normalize(y)
        xf = self.features(x)
        yf = self.features(y)
        return nn.functional.mse_loss(xf, yf)


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window(window_size)

    def _create_window(self, window_size):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
                              for x in range(window_size)])
        return gauss / gauss.sum()

    def forward(self, img1, img2):
        if not hasattr(self, 'window') or self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        mu1 = torch.nn.functional.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = torch.nn.functional.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined L1 + Perceptual + SSIM loss"""
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, ssim_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        per_loss = self.perceptual(pred, target)
        ssim_loss = self.ssim(pred, target)
        total = (self.l1_weight * l1_loss +
                self.perceptual_weight * per_loss +
                self.ssim_weight * ssim_loss)
        return total, {'l1': l1_loss.item(), 'perceptual': per_loss.item(), 'ssim': ssim_loss.item()}


class ModelTrainer:
    """Training manager with logging and checkpointing"""

    def __init__(self, model, device='cuda', lr=1e-4, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = CombinedLoss()
        self.scaler = GradScaler()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_metrics': [], 'val_metrics': [],
            'learning_rates': [],
        }
        self.best_val_loss = float('inf')

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            input_img = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                output = self.model(input_img)
                if isinstance(output, dict):
                    output = output.get('enhanced', output)
                loss, metrics = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            input_img = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            output = self.model(input_img)
            if isinstance(output, dict):
                output = output.get('enhanced', output)
            loss, _ = self.criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, train_dataset, val_dataset=None, epochs=50, batch_size=8, num_workers=4):
        """Full training loop"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f"Starting training for {epochs} epochs")
        print(f"Train batches: {len(train_loader)}, Device: {self.device}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, epochs + 1):
            start = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader) if val_loader else train_loss

            self.scheduler.step()

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(lr)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pth')

            # Save latest
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                self.save_samples(epoch, train_loader)

            print(f"Epoch {epoch}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | Time: {elapsed:.1f}s")

        self.save_history()
        return self.history

    def save_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, filename):
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.history = checkpoint.get('history', self.history)
            print(f"Loaded checkpoint: {path}")
            return True
        return False

    @torch.no_grad()
    def save_samples(self, epoch, dataloader):
        """Save sample outputs during training"""
        self.model.eval()
        samples_dir = self.checkpoint_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)

        batch = next(iter(dataloader))
        input_img = batch['input'][:4].to(self.device)
        target = batch['target'][:4]

        output = self.model(input_img)
        if isinstance(output, dict):
            output = output.get('enhanced', output)

        transform = transforms.ToPILImage()
        for i in range(min(4, len(input_img))):
            pred = transform(output[i].cpu().clamp(0, 1))
            orig = transform(target[i])
            input_pil = transform(batch['input'][i])

            # Create comparison grid
            w, h = pred.size()
            grid = Image.new('RGB', (w * 3, h))
            grid.paste(input_pil, (0, 0))
            grid.paste(orig, (w, 0))
            grid.paste(pred, (w * 2, 0))
            grid.save(samples_dir / f'epoch_{epoch:04d}_{i}.png')

    def save_history(self):
        path = self.checkpoint_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_status(self):
        return {
            'device': str(self.device),
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.history['train_loss']),
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
        }
