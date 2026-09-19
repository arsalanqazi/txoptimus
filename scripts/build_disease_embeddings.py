"""
Build Disease Embeddings for TxOptimus
======================================
Pre-computes PubMedBERT (BiomedBERT) embeddings for all disease names in both
PrimeKG and OptimusKG knowledge graphs. The resulting .pt and .json files are
shipped inside the data zip archives so that fuzzy disease matching works
at runtime without requiring GPU or model downloads.

Usage (on workstation with GPU or high-RAM CPU):
    python scripts/build_disease_embeddings.py \
        --prime_data  /path/to/TxGNN_Prime/data \
        --optimus_data /path/to/TxGNN_Optimus/data \
        --output_dir  ./embeddings_output

Output per KG:
    disease_embeddings_{prime|optimus}.pt   — dict {disease_id: tensor(768,)}
    disease_names_{prime|optimus}.json      — dict {disease_id: "disease name string"}

Requirements:
    pip install transformers torch pandas
"""

import os
import sys
import json
import argparse
from collections import OrderedDict

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def convert2str(x):
    """Standardise mixed-type IDs (float 5515.0 → '5515') to strings."""
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return str(x)
        if x == int(x):
            return str(int(x))
        return str(x)
    return str(x).strip()


def extract_disease_names(data_folder: str) -> dict:
    """
    Read kg.csv and build a {disease_id -> disease_name} mapping.
    Only reads the columns we need to keep memory usage sane.
    """
    print(f"  Reading kg.csv from {data_folder} ...")
    df = pd.read_csv(
        os.path.join(data_folder, 'kg.csv'),
        usecols=['x_id', 'x_type', 'x_name', 'y_id', 'y_type', 'y_name']
    )

    df_diseases = df[df.x_type == 'disease'][['x_id', 'x_name']].drop_duplicates()
    df_diseases_y = df[df.y_type == 'disease'][['y_id', 'y_name']].drop_duplicates()
    df_diseases_y.columns = ['x_id', 'x_name']
    df_diseases = pd.concat([df_diseases, df_diseases_y]).drop_duplicates()

    id2name = {}
    for _, row in df_diseases.iterrows():
        did = convert2str(row['x_id'])
        name = str(row['x_name']).strip()
        if name and name.lower() != 'nan':
            id2name[did] = name

    print(f"  Found {len(id2name)} unique disease names.")
    return id2name


def encode_disease_names(id2name: dict, model_name: str, batch_size: int = 64, device: str = 'cpu') -> dict:
    """
    Encode every disease name with BiomedBERT → 768-d normalised embeddings.
    Returns {disease_id: tensor(768,)} on CPU.
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"  Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    ids = list(id2name.keys())
    names = [id2name[i] for i in ids]

    embeddings = {}
    print(f"  Encoding {len(names)} disease names in batches of {batch_size} ...")
    for start in tqdm(range(0, len(names), batch_size), desc="  Encoding"):
        batch_names = names[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]

        encoded = tokenizer(
            batch_names,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS token embedding
            cls_embeds = outputs.last_hidden_state[:, 0, :]
            # L2 normalise for cosine similarity
            cls_embeds = torch.nn.functional.normalize(cls_embeds, p=2, dim=1)

        for i, did in enumerate(batch_ids):
            embeddings[did] = cls_embeds[i].cpu()

    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build PubMedBERT disease embeddings for TxOptimus")
    parser.add_argument("--prime_data", type=str, required=True,
                        help="Path to TxGNN Prime data/ folder (contains kg.csv)")
    parser.add_argument("--optimus_data", type=str, required=True,
                        help="Path to TxGNN Optimus data/ folder (contains kg.csv)")
    parser.add_argument("--output_dir", type=str, default="./embeddings_output",
                        help="Directory to write embedding files")
    parser.add_argument("--model_name", type=str,
                        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
                        help="HuggingFace model name for encoding")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for encoding (cpu or cuda:0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for kg_label, data_dir in [("prime", args.prime_data), ("optimus", args.optimus_data)]:
        print(f"\n{'='*60}")
        print(f"Processing {kg_label.upper()} ({data_dir})")
        print(f"{'='*60}")

        # 1. Extract disease names
        id2name = extract_disease_names(data_dir)

        # 2. Save disease names JSON
        names_path = os.path.join(args.output_dir, f"disease_names_{kg_label}.json")
        with open(names_path, 'w', encoding='utf-8') as f:
            json.dump(id2name, f, indent=2, ensure_ascii=False)
        print(f"  Saved disease names → {names_path}")

        # 3. Encode with BiomedBERT
        embeddings = encode_disease_names(
            id2name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )

        # 4. Save embeddings tensor dict
        emb_path = os.path.join(args.output_dir, f"disease_embeddings_{kg_label}.pt")
        torch.save(embeddings, emb_path)
        file_size_mb = os.path.getsize(emb_path) / (1024 * 1024)
        print(f"  Saved {len(embeddings)} embeddings → {emb_path} ({file_size_mb:.1f} MB)")

    print(f"\n{'='*60}")
    print("All embeddings generated successfully!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
