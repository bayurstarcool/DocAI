#!/usr/bin/env python3
"""
Parallel download script for ShadowDocument7K dataset
Downloads multiple files simultaneously for faster speed
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Get token
def get_token():
    result = subprocess.run(['openxlab', 'token'], capture_output=True, text=True)
    token = result.stdout.strip().replace('Bearer ', '')
    return token

# Get file list
def get_file_list(token, repo="lkljty/ShadowDocument7K"):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Use openxlab to get file list
    result = subprocess.run(
        ['openxlab', 'dataset', 'ls', '--dataset-repo', repo],
        capture_output=True, text=True
    )
    
    # Parse file list from output
    files = []
    for line in result.stdout.split('\n'):
        if '/' in line and ('.png' in line or '.jpg' in line or '.yaml' in line or '.md' in line):
            files.append(line.strip())
    
    return files

# Download single file
def download_file(token, repo, file_path, local_path):
    headers = {
        'Authorization': f'Bearer {token}',
    }
    
    # Get download URL
    # This is a simplified version - actual implementation may vary
    try:
        # Try to download directly
        url = f"https://openxlab.org.cn/gw/openxlab-xlab/api/v1/dataset/repo/{repo}/file/{file_path}"
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        pass
    
    return False

def main():
    repo = "lkljty/ShadowDocument7K"
    target_dir = Path("datasets/ShadowDocument7K")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Parallel Download for ShadowDocument7K ===")
    print(f"Repository: {repo}")
    print(f"Target: {target_dir}")
    print()
    
    # Get token
    token = get_token()
    print(f"✓ Token obtained")
    
    # Get file list
    print("\nFetching file list...")
    files = get_file_list(token, repo)
    print(f"✓ Found {len(files)} files")
    
    if not files:
        print("No files found. Trying alternative method...")
        return
    
    # Download files in parallel
    downloaded = 0
    failed = 0
    total_size = 0
    
    def download_with_progress(file_path):
        nonlocal downloaded, failed, total_size
        
        local_path = target_dir / file_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        if download_file(token, repo, file_path, local_path):
            downloaded += 1
            size = local_path.stat().st_size if local_path.exists() else 0
            total_size += size
            return True
        else:
            failed += 1
            return False
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_with_progress, f): f for f in files[:100]}  # Limit to 100 files for test
        
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"✓ Downloaded: {file_path}")
                else:
                    print(f"✗ Failed: {file_path}")
            except Exception as e:
                print(f"✗ Error: {file_path} - {e}")
    
    print(f"\n=== Download Complete ===")
    print(f"Downloaded: {downloaded} files")
    print(f"Failed: {failed} files")
    print(f"Total size: {total_size/1e6:.1f} MB")

if __name__ == '__main__':
    main()
