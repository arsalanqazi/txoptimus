"""
Disease Term Fuzzy Matcher
==========================
Uses pre-computed PubMedBERT embeddings to fuzzy-match user-provided disease
terms against the full disease vocabulary of PrimeKG or OptimusKG.

At runtime, only the user's short query string needs to be encoded. The
pre-computed embeddings (~5-8 MB .pt files) are loaded from the data directory.
"""

import os
import json
import torch
import numpy as np


class DiseaseMatcher:
    """Fuzzy disease name matcher using PubMedBERT embeddings."""

    MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

    def __init__(self, data_dir: str, kg_label: str):
        """
        Parameters
        ----------
        data_dir : str
            Root data directory for the KG (e.g. ~/.txoptimus/prime)
        kg_label : str
            One of 'prime' or 'optimus'
        """
        self.kg_label = kg_label

        emb_path = os.path.join(data_dir, "embeddings", f"disease_embeddings_{kg_label}.pt")
        names_path = os.path.join(data_dir, "embeddings", f"disease_names_{kg_label}.json")

        if not os.path.exists(emb_path) or not os.path.exists(names_path):
            raise FileNotFoundError(
                f"Disease embeddings not found at {emb_path}. "
                f"Run 'txoptimus --setup {kg_label}' to download data files."
            )

        # Load pre-computed embeddings
        self.embeddings_dict = torch.load(emb_path, map_location='cpu')
        with open(names_path, 'r', encoding='utf-8') as f:
            self.names_dict = json.load(f)

        # Build matrix for fast cosine similarity
        self.disease_ids = list(self.embeddings_dict.keys())
        self.disease_names = [self.names_dict.get(did, did) for did in self.disease_ids]
        self.emb_matrix = torch.stack([self.embeddings_dict[did] for did in self.disease_ids])
        # Already L2-normalised during pre-computation

        self._tokenizer = None
        self._model = None

    def _load_encoder(self):
        """Lazy-load the BiomedBERT encoder (only when user queries are made)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModel
            print(f"  Loading BiomedBERT encoder for query matching...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)
            self._model.eval()

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a single query string → 768-d normalised embedding."""
        self._load_encoder()
        encoded = self._tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self._model(**encoded)
            cls_embed = outputs.last_hidden_state[:, 0, :]
            cls_embed = torch.nn.functional.normalize(cls_embed, p=2, dim=1)
        return cls_embed.squeeze(0)

    def exact_match(self, query: str) -> list:
        """Try exact (case-insensitive) match first. Returns list of (id, name)."""
        q_lower = query.strip().lower()
        matches = []
        for did, name in zip(self.disease_ids, self.disease_names):
            if name.lower() == q_lower:
                matches.append((did, name))
        return matches

    def fuzzy_match(self, query: str, top_k: int = 10) -> list:
        """
        Fuzzy-match using cosine similarity of PubMedBERT embeddings.

        Returns list of (similarity_score, disease_id, disease_name).
        """
        query_emb = self._encode_query(query)
        # Cosine similarity (embeddings are already L2-normalised)
        similarities = torch.matmul(self.emb_matrix, query_emb)
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        results = []
        for idx in top_indices:
            idx = idx.item()
            results.append((
                float(similarities[idx]),
                self.disease_ids[idx],
                self.disease_names[idx]
            ))
        return results

    def match(self, query: str, top_k: int = 10) -> list:
        """
        Match a disease query — tries exact match first, falls back to fuzzy.

        Returns list of (similarity_score, disease_id, disease_name).
        """
        exact = self.exact_match(query)
        if exact:
            return [(1.0, did, name) for did, name in exact]
        return self.fuzzy_match(query, top_k=top_k)


def interactive_disease_selection(matcher: DiseaseMatcher, queries: list) -> list:
    """
    For each user query, display matched terms and prompt for selection.

    Parameters
    ----------
    matcher : DiseaseMatcher
    queries : list of str
        User-provided disease term strings.

    Returns
    -------
    list of (disease_id, disease_name) tuples selected by the user.
    """
    all_selected = []

    for query in queries:
        print(f"\n  Disease term: \"{query}\"")

        matches = matcher.match(query, top_k=10)

        if not matches:
            print("    No matches found in the knowledge graph. Skipping.")
            continue

        # Check if we have an exact match
        if matches[0][0] == 1.0:
            print(f"    ✓ Exact match: {matches[0][2]} (ID: {matches[0][1]})")
            all_selected.append((matches[0][1], matches[0][2]))
            continue

        # Display fuzzy matches
        print(f"\n    Matched terms in {matcher.kg_label.upper()}:")
        for i, (score, did, name) in enumerate(matches, 1):
            print(f"      [{i}] {name} (similarity: {score:.3f})")

        # Prompt user
        user_input = input("\n    Select terms (e.g., '1 3 5' or 'all'): ").strip()

        if user_input.lower() == 'all':
            for score, did, name in matches:
                all_selected.append((did, name))
        else:
            try:
                indices = [int(x) - 1 for x in user_input.split()]
                for idx in indices:
                    if 0 <= idx < len(matches):
                        _, did, name = matches[idx]
                        all_selected.append((did, name))
                    else:
                        print(f"    Warning: index {idx+1} out of range, skipping.")
            except ValueError:
                print("    Invalid input. Skipping this term.")

    return all_selected
