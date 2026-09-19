"""
PrimeKG Engine
==============
Loads TxGNN with PrimeKG data and runs zero-shot drug repurposing evaluation
on user-specified disease targets under the cell_proliferation split.
"""

import os
import sys
import time
import json
import math
import pickle
import functools

import torch
import numpy as np
import pandas as pd

# Ensure prints are flushed
print = functools.partial(print, flush=True)


def run_prime(disease_ids: list, disease_names: list,
              data_dir: str, ckpt_dir: str,
              output_dir: str, prefix: str,
              top_k: int = 100, threads: int = 18,
              graphmask: bool = False):
    """
    Run TxGNN evaluation on PrimeKG for the given disease targets.

    Parameters
    ----------
    disease_ids : list of str
        Resolved disease IDs from the matcher.
    disease_names : list of str
        Corresponding human-readable disease names.
    data_dir : str
        Path to PrimeKG data/ folder.
    ckpt_dir : str
        Path to model_ckpt/ folder containing config.pkl and model.pt.
    output_dir : str
        Directory to save results.
    prefix : str
        File prefix for outputs.
    top_k : int
        Number of drug candidates to export per disease.
    threads : int
        CPU threads for PyTorch.
    graphmask : bool
        Whether to run GraphMask explainability after evaluation.
    """
    # Add data_dir parent to path so txgnn core can be found
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from txoptimus.core import TxData, TxGNN, TxEval
    from txoptimus.core.utils import evaluate_fb, evaluate_graph_construct, convert2str

    num_threads = min(os.cpu_count() or 18, threads)
    torch.set_num_threads(num_threads)
    device = 'cpu'

    print(f"\n{'='*60}")
    print(f" TxOptimus — PrimeKG Engine")
    print(f" Threads: {num_threads} | Device: {device}")
    print(f" Data: {data_dir}")
    print(f" Checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # 1. Load data
    print("\n[1/4] Loading TxData splits (PrimeKG: cell_proliferation)...")
    tx_data = TxData(data_folder_path=data_dir)
    tx_data.prepare_split(split='cell_proliferation', seed=42, no_kg=False)

    # 2. Load model
    print("\n[2/4] Initializing model and loading weights...")
    tx_gnn = TxGNN(
        data=tx_data,
        weight_bias_track=False,
        proj_name='TxOptimus_Prime',
        exp_name='TxOptimus_Prime',
        device=device
    )

    config_path = os.path.join(ckpt_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    print(f"  Config: {config}")
    tx_gnn.model_initialize(**config)

    weight_path = os.path.join(ckpt_dir, 'model.pt')
    state_dict = torch.load(weight_path, map_location=device)
    if next(iter(state_dict))[:7] == 'module.':
        from collections import OrderedDict
        state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
    tx_gnn.best_model.load_state_dict(state_dict)
    tx_gnn.model = tx_gnn.best_model
    tx_gnn.model.eval()

    # 3. Global test metrics
    print("\n[3/4] Running Global Test Set Zero-Shot Evaluation...")
    eval_start = time.time()

    test_metrics = {}
    if hasattr(tx_data, 'df_test') and len(tx_data.df_test) > 0:
        if not hasattr(tx_gnn, 'g_test_pos') or tx_gnn.g_test_pos is None:
            tx_gnn.g_test_pos, tx_gnn.g_test_neg = evaluate_graph_construct(
                tx_data.df_test, tx_gnn.G, 'fix_dst', 1, device
            )

        (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc), test_loss, _, _ = evaluate_fb(
            tx_gnn.best_model, tx_gnn.g_test_pos, tx_gnn.g_test_neg,
            tx_gnn.G, tx_gnn.dd_etypes, device, True, mode='test'
        )

        def clean_rel_dict(d):
            return {
                str(k[1]) if isinstance(k, tuple) and len(k) > 1 else str(k):
                float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in d.items()
            }

        test_metrics = {
            'test_loss': float(test_loss),
            'macro_auroc': float(macro_auroc),
            'micro_auroc': float(micro_auroc),
            'macro_auprc': float(macro_auprc),
            'micro_auprc': float(micro_auprc),
            'auroc_by_relation': clean_rel_dict(auroc_rel),
            'auprc_by_relation': clean_rel_dict(auprc_rel),
        }
        print(f"  Macro AUROC: {macro_auroc:.4f} | Micro AUROC: {micro_auroc:.4f}")
    else:
        print("  No test graph edges available. Skipping global evaluation.")

    eval_duration = time.time() - eval_start

    # 4. Disease-centric evaluation
    print(f"\n[4/4] Running Disease-Centric Evaluation on {len(disease_ids)} targets...")
    evaluator = TxEval(model=tx_gnn)
    id_mapping = tx_data.retrieve_id_mapping()

    id2idx_disease = {str(v): k for k, v in id_mapping['idx2id_disease'].items()}
    for k, v in list(id_mapping['idx2id_disease'].items()):
        v_str = str(v).strip()
        id2idx_disease[v_str] = k
        try:
            f_val = float(v_str)
            if math.isfinite(f_val):
                id2idx_disease[str(f_val)] = k
                id2idx_disease[str(int(f_val))] = k
        except (ValueError, TypeError, OverflowError):
            pass

    id2name_drug = id_mapping['id2name_drug']
    name2id_drug = {v: k for k, v in id2name_drug.items()}
    id2name_disease = id_mapping['id2name_disease']

    target_indices = []
    found_targets = []
    for did, dname in zip(disease_ids, disease_names):
        candidates = [did, str(did).strip()]
        try:
            f_val = float(did)
            if math.isfinite(f_val):
                candidates.extend([str(f_val), str(int(f_val))])
        except (ValueError, TypeError, OverflowError):
            pass

        idx = None
        for c in candidates:
            if c in id2idx_disease:
                idx = id2idx_disease[c]
                break

        if idx is not None:
            target_indices.append(float(idx))
            found_targets.append(dname)
            print(f"  ✓ '{dname}' → Index {idx}")
        else:
            print(f"  ✗ '{dname}' (ID: {did}) — not found in graph index")

    disease_centric_results = {}
    candidate_rows = []

    if target_indices:
        results = evaluator.eval_disease_centric(
            disease_idxs=target_indices,
            relation='indication',
            save_result=False,
            return_raw=True
        )

        main_results = results.get('result', {})
        prediction_dict = results.get('prediction', {})
        label_dict = results.get('label', {})
        ranked_list_dict = main_results.get('Ranked List', {})

        for dis_id, drug_list in ranked_list_dict.items():
            dis_name = id2name_disease.get(dis_id, str(dis_id))
            dis_auroc = float(main_results.get('AUROC', {}).get(dis_id, 0.0))
            dis_auprc = float(main_results.get('AUPRC', {}).get(dis_id, 0.0))
            num_pos = int(main_results.get('# of Pos', {}).get(dis_id, 0))

            disease_centric_results[dis_name] = {
                'disease_id': str(dis_id),
                'auroc': dis_auroc,
                'auprc': dis_auprc,
                'num_positives': num_pos,
            }

            print(f"\n  {dis_name}: AUROC={dis_auroc:.4f} | Positives={num_pos}")

            for rank, drug_name in enumerate(drug_list[:top_k], start=1):
                drug_id = name2id_drug.get(drug_name)
                score = prediction_dict.get(dis_id, {}).get(drug_id, np.nan)
                label = label_dict.get(dis_id, {}).get(drug_id, 0)
                candidate_rows.append({
                    'Disease': dis_name, 'Rank': rank, 'Drug Name': drug_name,
                    'Score': float(score) if isinstance(score, (float, np.floating)) else score,
                    'Known Ground Truth': int(label) if isinstance(label, (int, np.integer)) else label,
                })

    # Save outputs
    total_duration = time.time() - start_time

    if candidate_rows:
        csv_path = os.path.join(output_dir, f"{prefix}_prime_candidates.csv")
        pd.DataFrame(candidate_rows).to_csv(csv_path, index=False)
        print(f"\n  Saved candidates → {csv_path}")

    results_payload = {
        'engine': 'PrimeKG',
        'data_split': 'cell_proliferation',
        'test_metrics': test_metrics,
        'disease_centric': disease_centric_results,
        'graph_profile': {
            'total_nodes': int(tx_gnn.G.num_nodes()),
            'total_edges': int(tx_gnn.G.num_edges()),
            'targets_found': found_targets,
        },
        'timing': {'total_seconds': float(total_duration)},
    }
    json_path = os.path.join(output_dir, f"{prefix}_prime_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_payload, f, indent=2)
    print(f"  Saved metrics → {json_path}")

    # GraphMask
    if graphmask:
        print(f"\n  ⚠ GraphMask training requires ≥30 GB RAM and may take 2-4 hours on CPU.")
        _run_graphmask(tx_gnn, output_dir, prefix, "prime")

    print(f"\n  PrimeKG engine completed in {total_duration/60:.1f} minutes.")
    return results_payload


def _run_graphmask(tx_gnn, output_dir, prefix, label):
    """Run GraphMask explainability training."""
    import gc
    gc.collect()

    graphmask_dir = os.path.join(output_dir, f"{prefix}_{label}_graphmask")
    os.makedirs(graphmask_dir, exist_ok=True)
    tx_gnn._checkpoint_dir = graphmask_dir

    print(f"\n  Starting GraphMask training (indication)...")
    try:
        metrics = tx_gnn.train_graphmask(
            relation='indication',
            learning_rate=3e-4,
            allowance=0.005,
            epochs_per_layer=500,
            penalty_scaling=1,
            moving_average_window_size=100,
            valid_per_n=25,
            no_base=False,
            gate_hidden_size=32
        )
        print(f"  GraphMask completed: {metrics}")
    except Exception as e:
        print(f"  GraphMask failed: {e}")

    try:
        tx_gnn.save_graphmask_model(graphmask_dir)
        print(f"  Saved GraphMask model → {graphmask_dir}")
    except Exception as e:
        print(f"  Warning: save_graphmask_model failed: {e}")
        if hasattr(tx_gnn, 'best_graphmask_model'):
            torch.save(tx_gnn.best_graphmask_model.state_dict(),
                        os.path.join(graphmask_dir, 'graphmask_model.pt'))
