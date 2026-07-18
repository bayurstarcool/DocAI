#!/usr/bin/env python3
"""
Prepare ShadowDocument7K dataset for training
"""

import os
import shutil
from pathlib import Path

def prepare_dataset():
    """Prepare the dataset directory structure"""
    
    base_dir = Path('/home/wahyu/docai/datasets/ShadowDocument7K')
    
    if not base_dir.exists():
        print(f"Dataset not found at {base_dir}")
        print("Please download the dataset first:")
        print("  1. Go to: https://1drv.ms/u/s!Avp0JjwC1wv5jhx6TxHfjlL4BMuA?e=GFTcze")
        print("  2. Download the compressed file (9 GB)")
        print("  3. Extract to: datasets/ShadowDocument7K/")
        return False
    
    # Check structure
    train_input = base_dir / 'train' / 'input'
    train_target = base_dir / 'train' / 'target'
    test_input = base_dir / 'test' / 'input'
    test_target = base_dir / 'test' / 'target'
    
    print("Checking dataset structure...")
    
    if train_input.exists():
        train_count = len(list(train_input.glob('*.png'))) + len(list(train_input.glob('*.jpg')))
        print(f"  Train input: {train_count} images")
    else:
        print("  Train input: NOT FOUND")
    
    if train_target.exists():
        train_count = len(list(train_target.glob('*.png'))) + len(list(train_target.glob('*.jpg')))
        print(f"  Train target: {train_count} images")
    else:
        print("  Train target: NOT FOUND")
    
    if test_input.exists():
        test_count = len(list(test_input.glob('*.png'))) + len(list(test_input.glob('*.jpg')))
        print(f"  Test input: {test_count} images")
    else:
        print("  Test input: NOT FOUND")
    
    if test_target.exists():
        test_count = len(list(test_target.glob('*.png'))) + len(list(test_target.glob('*.jpg')))
        print(f"  Test target: {test_count} images")
    else:
        print("  Test target: NOT FOUND")
    
    # Update config.yml
    config_path = Path('/home/wahyu/docai/docshadow_sd7k/config.yml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = f.read()
        
        # Update paths
        config = config.replace(
            "TRAIN_DIR: '../data/Jung/train/'",
            f"TRAIN_DIR: '../datasets/ShadowDocument7K/train/'"
        )
        config = config.replace(
            "VAL_DIR: '../data/Jung/test/'",
            f"VAL_DIR: '../datasets/ShadowDocument7K/test/'"
        )
        
        with open(config_path, 'w') as f:
            f.write(config)
        
        print("\nUpdated config.yml with dataset paths")
    
    print("\nDataset preparation complete!")
    print("\nTo start training:")
    print("  cd docshadow_sd7k")
    print("  python train.py")
    
    return True


if __name__ == '__main__':
    prepare_dataset()
