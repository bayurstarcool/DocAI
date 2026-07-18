#!/usr/bin/env python3
"""
Fast download script for ShadowDocument7K dataset
Uses openxlab Python API with progress tracking
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add openxlab to path
sys.path.insert(0, '/home/wahyu/miniconda3/lib/python3.14/site-packages')

from openxlab.xlab.handler.user_config import get_config
from openxlab.xlab.clients.auth_client import AuthClient
from openxlab.xlab.clients.dataset_client import DatasetClient

def get_client():
    """Get authenticated client"""
    config = get_config()
    if not config:
        raise Exception("Please configure openxlab first")
    
    auth_client = AuthClient("https://openapi.openxlab.org.cn/api/v1/sso-be/api/v1/open/")
    token = auth_client.get_token(config.ak, config.sk)
    
    dataset_client = DatasetClient("https://openxlab.org.cn/gw/openxlab-xlab/")
    dataset_client.set_token(token.jwt)
    
    return dataset_client

def list_files(client, repo, path="/"):
    """List all files in dataset"""
    files = []
    cursor = None
    
    while True:
        result = client.list_files(repo, path, cursor=cursor)
        if not result or 'files' not in result:
            break
        
        for f in result['files']:
            files.append(f)
        
        cursor = result.get('cursor')
        if not cursor:
            break
    
    return files

def download_file(client, repo, file_path, local_path):
    """Download a single file"""
    try:
        # Get download URL
        url = client.get_download_url(repo, file_path)
        
        # Download with requests
        import requests
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")
        return False

def main():
    repo = "lkljty/ShadowDocument7K"
    target_dir = Path("datasets/ShadowDocument7K")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Fast Download for ShadowDocument7K ===")
    print(f"Repository: {repo}")
    print(f"Target: {target_dir}")
    print()
    
    try:
        client = get_client()
        print("✓ Connected to OpenXLab")
        
        # List all files
        print("\nFetching file list...")
        files = list_files(client, repo)
        print(f"✓ Found {len(files)} files")
        
        # Download files
        downloaded = 0
        failed = 0
        total_size = 0
        
        for file_info in files:
            file_path = file_info.get('path', '')
            file_size = file_info.get('size', 0)
            
            # Create local path
            local_path = target_dir / file_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already exists
            if local_path.exists() and local_path.stat().st_size == file_size:
                downloaded += 1
                continue
            
            # Download
            print(f"Downloading: {file_path} ({file_size/1024:.1f} KB)")
            if download_file(client, repo, file_path, local_path):
                downloaded += 1
                total_size += file_size
                print(f"  ✓ Downloaded ({downloaded}/{len(files)})")
            else:
                failed += 1
                print(f"  ✗ Failed")
            
            # Progress update
            if downloaded % 10 == 0:
                print(f"\n--- Progress: {downloaded}/{len(files)} files ({total_size/1e6:.1f} MB) ---\n")
        
        print(f"\n=== Download Complete ===")
        print(f"Downloaded: {downloaded} files")
        print(f"Failed: {failed} files")
        print(f"Total size: {total_size/1e6:.1f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
