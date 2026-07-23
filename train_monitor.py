#!/usr/bin/env python3
"""DocAI Training Monitor — comprehensive real-time status."""

import json
import os
import re
import subprocess
import time
from pathlib import Path

BASE_DIR = Path('/home/wahyu/docai')
CHECKPOINT_DIR = BASE_DIR / 'checkpoints' / 'document_restorer'
HISTORY_FILE = CHECKPOINT_DIR / 'training_history.json'
LOG_FILE = BASE_DIR / 'data' / 'train_live.log'


def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(',')]
        if len(parts) < 7:
            return None
        return {
            'name': parts[0],
            'vram_used_mb': float(parts[1]),
            'vram_total_mb': float(parts[2]),
            'utilization_pct': float(parts[3]),
            'temperature_c': float(parts[4]),
            'power_draw_w': float(parts[5]),
            'power_limit_w': float(parts[6]),
        }
    except Exception:
        return None


def parse_log(log_text):
    """Parse live training log for current status."""
    lines = log_text.strip().split('\n')
    status = {
        'current_epoch': None,
        'total_epochs': None,
        'current_batch': None,
        'total_batches': None,
        'train_loss': None,
        'val_loss': None,
        'val_psnr': None,
        'val_ssim': None,
        'eta': None,
        'best_loss': None,
        'epoch_time': None,
        'config': {},
    }

    for line in reversed(lines):
        # Parse epoch line: epoch=1/60 batch=25/1872 train_loss=0.7031
        m = re.search(r'epoch=(\d+)/(\d+)\s+batch=(\d+)/(\d+)\s+train_loss=(\S+)', line)
        if m:
            status['current_epoch'] = int(m.group(1))
            status['total_epochs'] = int(m.group(2))
            status['current_batch'] = int(m.group(3))
            status['total_batches'] = int(m.group(4))
            status['train_loss'] = float(m.group(5))
            break

    # Parse validation line: epoch=1/60 train_loss=0.5021 val_loss=0.3658 val_psnr=18.81 val_ssim=0.8556 best=0.3658
    for line in reversed(lines):
        m = re.search(r'epoch=(\d+)/(\d+)\s+train_loss=(\S+)\s+val_loss=(\S+)\s+val_psnr=(\S+)\s+val_ssim=(\S+)\s+best=(\S+)', line)
        if m:
            status['val_loss'] = float(m.group(3))
            status['val_psnr'] = float(m.group(4))
            status['val_ssim'] = float(m.group(5))
            status['best_loss'] = float(m.group(6))
            break

    # Parse ETA: eta=1h 23m 45s
    for line in reversed(lines):
        m = re.search(r'eta=(\S+\s+\S+\s+\S+)', line)
        if m:
            status['eta'] = m.group(1)
            break

    # Parse epoch time: epoch_time=123.4s
    for line in reversed(lines):
        m = re.search(r'epoch_time=(\S+)', line)
        if m:
            status['epoch_time'] = m.group(1)
            break

    # Parse config from CMD line
    for line in lines:
        if line.startswith('[CMD]'):
            args = line.split()
            for i, arg in enumerate(args):
                if arg == '--epochs' and i + 1 < len(args):
                    status['config']['epochs'] = args[i + 1]
                elif arg == '--batch-size' and i + 1 < len(args):
                    status['config']['batch_size'] = args[i + 1]
                elif arg == '--size' and i + 1 < len(args):
                    status['config']['size'] = args[i + 1]
                elif arg == '--lr' and i + 1 < len(args):
                    status['config']['lr'] = args[i + 1]
                elif arg == '--identity-weight' and i + 1 < len(args):
                    status['config']['identity_weight'] = args[i + 1]
            break

    return status


def format_eta(eta_str):
    if not eta_str:
        return 'N/A'
    return eta_str


def print_status():
    print('=' * 60)
    print('DocAI Training Monitor')
    print('=' * 60)

    # Check if training is running
    is_running = False
    if LOG_FILE.exists():
        log_text = LOG_FILE.read_text()
        if 'Training process exited' not in log_text[-200:]:
            is_running = True

    if is_running:
        print('\n[STATUS] Training RUNNING ✅')
    else:
        print('\n[STATUS] Training NOT RUNNING ❌')

    # GPU status
    gpu = get_gpu_stats()
    if gpu:
        vram_pct = (gpu['vram_used_mb'] / gpu['vram_total_mb']) * 100
        vram_color = '✅' if vram_pct < 90 else '⚠️' if vram_pct < 95 else '❌'
        temp_color = '✅' if gpu['temperature_c'] < 70 else '⚠️' if gpu['temperature_c'] < 80 else '❌'
        power_pct = (gpu['power_draw_w'] / gpu['power_limit_w']) * 100

        print(f'\n[GPU] {gpu["name"]}')
        print(f'  VRAM: {gpu["vram_used_mb"]:.0f}/{gpu["vram_total_mb"]:.0f} MB ({vram_pct:.1f}%) {vram_color}')
        print(f'  Utilization: {gpu["utilization_pct"]:.0f}%')
        print(f'  Temperature: {gpu["temperature_c"]:.0f}°C {temp_color}')
        print(f'  Power: {gpu["power_draw_w"]:.0f}/{gpu["power_limit_w"]:.0f} W ({power_pct:.0f}%)')

    # Parse live log
    if LOG_FILE.exists():
        log_text = LOG_FILE.read_text()
        status = parse_log(log_text)

        if status['current_epoch']:
            progress = ((status['current_epoch'] - 1) + (status['current_batch'] / status['total_batches'])) / status['total_epochs'] * 100
            print(f'\n[PROGRESS] Epoch {status["current_epoch"]}/{status["total_epochs"]} '
                  f'Batch {status["current_batch"]}/{status["total_batches"]} '
                  f'({progress:.1f}%)')

            if status['train_loss']:
                print(f'  Train Loss: {status["train_loss"]:.4f}')

            if status['val_loss']:
                print(f'\n[VALIDATION]')
                print(f'  Val Loss: {status["val_loss"]:.4f}')
                print(f'  PSNR: {status["val_psnr"]:.2f} dB')
                print(f'  SSIM: {status["val_ssim"]:.4f}')
                if status['best_loss']:
                    print(f'  Best: {status["best_loss"]:.4f}')

            if status['eta']:
                print(f'\n[ETA] {status["eta"]}')

            if status['epoch_time']:
                print(f'[SPEED] {status["epoch_time"]}/epoch')

            # Config
            if status['config']:
                print(f'\n[CONFIG]')
                for k, v in status['config'].items():
                    print(f'  {k}: {v}')

        # Quality assessment
        if status['val_psnr'] is not None:
            print(f'\n[QUALITY]')
            psnr = status['val_psnr']
            if psnr >= 25:
                print(f'  PSNR: ✅ Excellent ({psnr:.2f} dB)')
            elif psnr >= 22:
                print(f'  PSNR: ⚠️  Good ({psnr:.2f} dB)')
            elif psnr >= 20:
                print(f'  PSNR: ⚠️  Fair ({psnr:.2f} dB)')
            else:
                print(f'  PSNR: ❌ Poor ({psnr:.2f} dB)')

            ssim = status['val_ssim']
            if ssim >= 0.90:
                print(f'  SSIM: ✅ Excellent ({ssim:.4f})')
            elif ssim >= 0.85:
                print(f'  SSIM: ⚠️  Good ({ssim:.4f})')
            elif ssim >= 0.80:
                print(f'  SSIM: ⚠️  Fair ({ssim:.4f})')
            else:
                print(f'  SSIM: ❌ Poor ({ssim:.4f})')

    # Recent errors
    if LOG_FILE.exists():
        errors = [l for l in LOG_FILE.read_text().split('\n') if 'Error' in l or 'Traceback' in l or 'OOM' in l]
        if errors:
            print(f'\n[ERRORS] {len(errors)} error(s) found')
            for e in errors[-3:]:
                print(f'  {e[:100]}')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    print_status()
