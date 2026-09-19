"""
Create Data Zips for TxOptimus S3 Distribution
===============================================
Packages the correct data files for PrimeKG and OptimusKG into two separate
zip archives ready for upload to S3. Only includes the files needed for the
cell_proliferation pipeline.

Usage:
    python scripts/create_data_zips.py \
        --prime_root    /path/to/TxGNN_Prime \
        --optimus_root  /path/to/TxGNN_Optimus \
        --embeddings_dir ./embeddings_output \
        --output_dir    ./zips_output

Output:
    txoptimus_prime_data.zip
    txoptimus_optimus_data.zip
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# File manifests — ONLY include what's needed for cell_proliferation
# ---------------------------------------------------------------------------

PRIME_DATA_FILES = [
    # Core KG
    "data/kg.csv",
    "data/node.csv",
    "data/edges.csv",
    # Cell proliferation split
    "data/cell_proliferation_42/train.csv",
    "data/cell_proliferation_42/valid.csv",
    "data/cell_proliferation_42/test.csv",
    # Directed subgraph
    "data/cell_proliferation_kg/kg_directed.csv",
    # Disease list
    "data/disease_files/cell_proliferation.csv",
    # Model checkpoint
    "model_ckpt/config.pkl",
    "model_ckpt/model.pt",
]

OPTIMUS_DATA_FILES = [
    # Core KG
    "data/kg.csv",
    "data/node.csv",
    "data/edges.csv",
    # Cell proliferation split
    "data/cell_proliferation_42/train.csv",
    "data/cell_proliferation_42/valid.csv",
    "data/cell_proliferation_42/test.csv",
    # Directed subgraph
    "data/cell_proliferation_kg/kg_directed.csv",
    # Disease list
    "data/disease_files/cell_proliferation.csv",
    # Model checkpoint
    "model_ckpt/config.pkl",
    "model_ckpt/model.pt",
]


def create_zip(root_dir: str, file_list: list, embeddings_dir: str,
               kg_label: str, output_path: str):
    """
    Create a zip archive containing:
      1. Data and model files from the KG directory
      2. Pre-computed PubMedBERT embeddings
    """
    missing = []
    for f in file_list:
        full = os.path.join(root_dir, f)
        if not os.path.exists(full):
            missing.append(f)

    if missing:
        print(f"\n  WARNING: The following files are missing from {root_dir}:")
        for m in missing:
            print(f"    - {m}")
        print("  These files will be SKIPPED.\n")

    # Check embeddings exist
    emb_file = os.path.join(embeddings_dir, f"disease_embeddings_{kg_label}.pt")
    names_file = os.path.join(embeddings_dir, f"disease_names_{kg_label}.json")

    if not os.path.exists(emb_file):
        print(f"  ERROR: Embedding file not found: {emb_file}")
        print("  Run build_disease_embeddings.py first!")
        sys.exit(1)
    if not os.path.exists(names_file):
        print(f"  ERROR: Names file not found: {names_file}")
        print("  Run build_disease_embeddings.py first!")
        sys.exit(1)

    print(f"  Creating {output_path} ...")
    total_size = 0

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Add data and model files
        for f in file_list:
            full = os.path.join(root_dir, f)
            if os.path.exists(full):
                fsize = os.path.getsize(full) / (1024 * 1024)
                total_size += fsize
                print(f"    Adding: {f} ({fsize:.1f} MB)")
                zf.write(full, arcname=f)

        # Add embeddings
        for src, arcname in [
            (emb_file, f"embeddings/disease_embeddings_{kg_label}.pt"),
            (names_file, f"embeddings/disease_names_{kg_label}.json"),
        ]:
            fsize = os.path.getsize(src) / (1024 * 1024)
            total_size += fsize
            print(f"    Adding: {arcname} ({fsize:.1f} MB)")
            zf.write(src, arcname=arcname)

    zip_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Done! {output_path}")
    print(f"  Uncompressed total: {total_size:.0f} MB")
    print(f"  Compressed zip:     {zip_size:.0f} MB")
    return zip_size


def main():
    parser = argparse.ArgumentParser(description="Create TxOptimus data zip archives for S3")
    parser.add_argument("--prime_root", type=str, required=True,
                        help="Root directory of TxGNN_Prime (contains data/ and model_ckpt/)")
    parser.add_argument("--optimus_root", type=str, required=True,
                        help="Root directory of TxGNN_Optimus (contains data/ and model_ckpt/)")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory containing pre-computed embedding files")
    parser.add_argument("--output_dir", type=str, default="./zips_output",
                        help="Directory to write zip archives")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" TxOptimus Data Zip Builder")
    print("=" * 60)

    # --- Prime ---
    print(f"\n--- PrimeKG ({args.prime_root}) ---")
    prime_zip = os.path.join(args.output_dir, "txoptimus_prime_data.zip")
    prime_size = create_zip(
        args.prime_root, PRIME_DATA_FILES,
        args.embeddings_dir, "prime", prime_zip
    )

    # --- Optimus ---
    print(f"\n--- OptimusKG ({args.optimus_root}) ---")
    optimus_zip = os.path.join(args.output_dir, "txoptimus_optimus_data.zip")
    optimus_size = create_zip(
        args.optimus_root, OPTIMUS_DATA_FILES,
        args.embeddings_dir, "optimus", optimus_zip
    )

    print(f"\n{'=' * 60}")
    print(f" Summary")
    print(f"{'=' * 60}")
    print(f"  txoptimus_prime_data.zip:   {prime_size:.0f} MB")
    print(f"  txoptimus_optimus_data.zip: {optimus_size:.0f} MB")
    print(f"  Output directory: {args.output_dir}")
    print(f"\n  Next step: Upload these zips to your S3 bucket.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
