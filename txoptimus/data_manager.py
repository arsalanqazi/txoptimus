"""
Data Manager — S3 Download & Extraction
========================================
Handles downloading the data zip archives from S3, extracting them into the
local data directory, and verifying integrity.
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


# Placeholder — will be updated once zips are uploaded to S3
S3_URLS = {
    "prime": "https://txoptimus.s3.us-east-1.amazonaws.com/txoptimus_prime_data.zip",
    "optimus": "https://txoptimus.s3.us-east-1.amazonaws.com/txoptimus_optimus_data.zip",
}

DEFAULT_DATA_DIR = os.path.join(str(Path.home()), ".txoptimus")


def get_data_dir(kg_label: str, base_dir: str = None) -> str:
    """Return the resolved data directory for a given KG label."""
    base = base_dir or DEFAULT_DATA_DIR
    return os.path.join(base, kg_label)


def is_data_ready(kg_label: str, base_dir: str = None) -> bool:
    """Check if required data files exist for the given KG."""
    data_dir = get_data_dir(kg_label, base_dir)
    required = [
        os.path.join(data_dir, "data", "kg.csv"),
        os.path.join(data_dir, "model_ckpt", "model.pt"),
        os.path.join(data_dir, "embeddings", f"disease_embeddings_{kg_label}.pt"),
    ]
    return all(os.path.exists(f) for f in required)


def download_file(url: str, dest: str):
    """Download a file from URL with progress bar."""
    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(dest, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        desc="  Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            f.write(chunk)
            pbar.update(len(chunk))

    actual_size = os.path.getsize(dest)
    print(f"  Downloaded: {actual_size / (1024*1024):.0f} MB")


def extract_zip(zip_path: str, extract_to: str):
    """Extract a zip archive with progress."""
    print(f"  Extracting to: {extract_to}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="  Extracting"):
            zf.extract(member, extract_to)

    print(f"  Extracted {len(members)} files.")


def setup_data(kg_label: str = None, base_dir: str = None):
    """
    Download and extract data for the specified KG(s).

    Parameters
    ----------
    kg_label : str or None
        'prime', 'optimus', or None (= both)
    base_dir : str or None
        Override the default ~/.txoptimus directory
    """
    base = base_dir or DEFAULT_DATA_DIR
    os.makedirs(base, exist_ok=True)

    labels = [kg_label] if kg_label else ["prime", "optimus"]

    for label in labels:
        print(f"\n{'='*60}")
        print(f"  Setting up TxOptimus {label.upper()} data")
        print(f"{'='*60}")

        data_dir = get_data_dir(label, base)

        if is_data_ready(label, base):
            print(f"  Data already exists at {data_dir}. Skipping download.")
            print(f"  (Delete the directory to force re-download.)")
            continue

        os.makedirs(data_dir, exist_ok=True)

        url = S3_URLS.get(label)
        if not url or url.startswith("S3_BUCKET_URL"):
            print(f"  ERROR: S3 URL not configured for '{label}'.")
            print(f"  Please update S3_URLS in data_manager.py with your bucket URL.")
            continue

        zip_path = os.path.join(base, f"txoptimus_{label}_data.zip")

        try:
            # Download
            download_file(url, zip_path)

            # Extract
            extract_zip(zip_path, data_dir)

            # Cleanup zip
            os.remove(zip_path)
            print(f"  Cleaned up zip archive.")

            # Verify
            if is_data_ready(label, base):
                print(f"  ✓ {label.upper()} data setup complete!")
            else:
                print(f"  ⚠ Warning: Some expected files may be missing. Check {data_dir}.")

        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Download failed: {e}")
            print(f"  Please check your internet connection and the S3 URL.")
        except zipfile.BadZipFile:
            print(f"  ERROR: Downloaded file is not a valid zip archive.")
        except Exception as e:
            print(f"  ERROR: {e}")


def print_requirements_banner():
    """Print system requirements after pip install."""
    banner = """
============================================================
 TxOptimus v0.1.2 installed successfully!

 REQUIREMENTS:
   • RAM:     ≥32 GB (OptimusKG loads a 21.8M-edge graph)
   • Storage: ≥8 GB free disk space for data files
   • Python:  3.8+ (tested on 3.8.20)
   • GPU:     Optional (CPU inference supported)

 QUICK START:
   1. Download data:  txoptimus --setup
   2. Run inference:  txoptimus --engine optimus --diseases "oral cavity cancer"
   3. Benchmark:      txoptimus --engine benchmark --diseases "oral cavity cancer"

 For more details:  txoptimus --help
============================================================
"""
    print(banner)
