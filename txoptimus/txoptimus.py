#!/usr/bin/env python
"""
TxOptimus — Master Control Script
==================================
Unified CLI for zero-shot drug repurposing using TxGNN on PrimeKG and OptimusKG.

Usage:
    txoptimus --engine optimus --diseases "oral cavity cancer" "glioblastoma"
    txoptimus --engine benchmark --diseases "oral cavity cancer"
    txoptimus --engine prime --diseases "breast cancer" --graphmask
    txoptimus --setup
    txoptimus --setup prime
"""

import os
import sys
import argparse
import functools

print = functools.partial(print, flush=True)

from txoptimus.data_manager import setup_data, is_data_ready, get_data_dir, DEFAULT_DATA_DIR
from txoptimus.disease_matcher import DiseaseMatcher, interactive_disease_selection
from txoptimus.engines.prime import run_prime
from txoptimus.engines.optimus import run_optimus


BANNER = r"""
  ______      ____        __  _                    
 /_  __/  __ / __ \____  / /_(_)___ ___  __  ______
  / / | |/_// / / / __ \/ __/ / __ `__ \/ / / / ___/
 / / _>  < / /_/ / /_/ / /_/ / / / / / / /_/ (__  ) 
/_/ /_/|_| \____/ .___/\__/_/_/ /_/ /_/\__,_/____/  
               /_/                            v0.1.2
    Zero-shot Drug Repurposing via TxGNN
    PrimeKG + OptimusKG | by arsalanriaz
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="TxOptimus — Zero-shot drug repurposing via TxGNN on PrimeKG and OptimusKG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  txoptimus --engine optimus --diseases "oral cavity cancer"
  txoptimus --engine benchmark --diseases "oral cavity cancer" "glioblastoma"
  txoptimus --engine prime --diseases "breast cancer" --graphmask
  txoptimus --setup
  txoptimus --setup optimus
        """
    )

    # Setup mode
    parser.add_argument("--setup", nargs='?', const='both', default=None,
                        choices=['prime', 'optimus', 'both'],
                        help="Download data files. No arg = both, or specify 'prime'/'optimus'.")

    # Engine selection
    parser.add_argument("--engine", type=str, default="optimus",
                        choices=["prime", "optimus", "benchmark"],
                        help="Engine to use: prime, optimus, or benchmark (runs both)")

    # Disease targets
    parser.add_argument("--diseases", nargs='+', type=str, default=None,
                        help="Disease terms for zero-shot repurposing (fuzzy matched)")

    # GraphMask
    parser.add_argument("--graphmask", action='store_true', default=False,
                        help="Enable GraphMask explainability (requires ≥30GB RAM, 2-4 hours)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./txoptimus_output",
                        help="Output directory for results (default: ./txoptimus_output)")
    parser.add_argument("--prefix", type=str, default="txoptimus",
                        help="File prefix for all output files (default: txoptimus)")

    # Performance
    parser.add_argument("--threads", type=int, default=18,
                        help="CPU threads for PyTorch (default: 18)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of drug candidates per disease (default: 100)")

    # Data location override
    parser.add_argument("--data_dir", type=str, default=None,
                        help=f"Override default data directory (default: {DEFAULT_DATA_DIR})")

    return parser.parse_args()


def resolve_diseases(engine: str, queries: list, base_dir: str) -> dict:
    """
    Resolve disease queries for the requested engine(s).
    Returns dict like {'prime': [(id, name), ...], 'optimus': [(id, name), ...]}.
    """
    engines_to_match = []
    if engine == 'benchmark':
        engines_to_match = ['prime', 'optimus']
    else:
        engines_to_match = [engine]

    resolved = {}
    for eng in engines_to_match:
        data_dir = get_data_dir(eng, base_dir)
        print(f"\n  Matching disease terms against {eng.upper()} knowledge graph...")

        try:
            matcher = DiseaseMatcher(data_dir, eng)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            print(f"  Run 'txoptimus --setup {eng}' to download data files first.")
            sys.exit(1)

        selected = interactive_disease_selection(matcher, queries)

        if not selected:
            print(f"\n  No valid disease targets resolved for {eng.upper()}. Aborting.")
            sys.exit(1)

        print(f"\n  Selected {len(selected)} target(s) for {eng.upper()}:")
        for did, dname in selected:
            print(f"    • {dname} (ID: {did})")

        resolved[eng] = selected

    return resolved


def run_comparison(prime_results: dict, optimus_results: dict,
                   output_dir: str, prefix: str):
    """Generate a comparison markdown report from both engine results."""
    import json

    report_lines = [
        "# TxOptimus Benchmark Comparison: PrimeKG vs OptimusKG",
        "",
        "## Global Test Metrics",
        "",
        "| Metric | PrimeKG | OptimusKG | Delta |",
        "| :--- | :---: | :---: | :---: |",
    ]

    pm = prime_results.get('test_metrics', {})
    om = optimus_results.get('test_metrics', {})

    for metric_key, label in [
        ('macro_auroc', 'Macro AUROC'),
        ('micro_auroc', 'Micro AUROC'),
        ('macro_auprc', 'Macro AUPRC'),
        ('micro_auprc', 'Micro AUPRC'),
        ('test_loss', 'Test Loss'),
    ]:
        pv = pm.get(metric_key, 'N/A')
        ov = om.get(metric_key, 'N/A')
        if isinstance(pv, (int, float)) and isinstance(ov, (int, float)):
            delta = ov - pv
            report_lines.append(f"| **{label}** | {pv:.4f} | {ov:.4f} | {delta:+.4f} |")
        else:
            report_lines.append(f"| **{label}** | {pv} | {ov} | — |")

    report_lines.extend([
        "",
        "## Graph Scale",
        "",
        "| Dimension | PrimeKG | OptimusKG |",
        "| :--- | :---: | :---: |",
    ])

    pgp = prime_results.get('graph_profile', {})
    ogp = optimus_results.get('graph_profile', {})
    report_lines.append(f"| **Nodes** | {pgp.get('total_nodes', 'N/A'):,} | {ogp.get('total_nodes', 'N/A'):,} |")
    report_lines.append(f"| **Edges** | {pgp.get('total_edges', 'N/A'):,} | {ogp.get('total_edges', 'N/A'):,} |")

    report_lines.extend(["", "---", f"*Generated by TxOptimus v0.1.2*", ""])

    report_path = os.path.join(output_dir, f"{prefix}_comparison_report.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Saved comparison report → {report_path}")


def main():
    args = parse_args()
    print(BANNER)

    base_dir = args.data_dir or DEFAULT_DATA_DIR

    # --- Setup mode ---
    if args.setup is not None:
        label = None if args.setup == 'both' else args.setup
        setup_data(kg_label=label, base_dir=base_dir)
        return

    # --- Inference / Benchmark mode ---
    if not args.diseases:
        print("  ERROR: --diseases is required for inference/benchmark mode.")
        print("  Example: txoptimus --engine optimus --diseases \"oral cavity cancer\"")
        sys.exit(1)

    # Check data availability
    engines_needed = ['prime', 'optimus'] if args.engine == 'benchmark' else [args.engine]
    for eng in engines_needed:
        if not is_data_ready(eng, base_dir):
            print(f"  ERROR: {eng.upper()} data not found at {get_data_dir(eng, base_dir)}")
            print(f"  Run 'txoptimus --setup {eng}' to download data files first.")
            sys.exit(1)

    # Resolve disease terms
    resolved = resolve_diseases(args.engine, args.diseases, base_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run engine(s) ---
    prime_results = None
    optimus_results = None

    if args.engine in ('prime', 'benchmark'):
        disease_ids = [did for did, _ in resolved['prime']]
        disease_names = [name for _, name in resolved['prime']]
        prime_data = get_data_dir('prime', base_dir)

        prime_results = run_prime(
            disease_ids=disease_ids,
            disease_names=disease_names,
            data_dir=os.path.join(prime_data, 'data'),
            ckpt_dir=os.path.join(prime_data, 'model_ckpt'),
            output_dir=args.output_dir,
            prefix=args.prefix,
            top_k=args.top_k,
            threads=args.threads,
            graphmask=args.graphmask,
        )

    if args.engine in ('optimus', 'benchmark'):
        disease_ids = [did for did, _ in resolved['optimus']]
        disease_names = [name for _, name in resolved['optimus']]
        optimus_data = get_data_dir('optimus', base_dir)

        optimus_results = run_optimus(
            disease_ids=disease_ids,
            disease_names=disease_names,
            data_dir=os.path.join(optimus_data, 'data'),
            ckpt_dir=os.path.join(optimus_data, 'model_ckpt'),
            output_dir=args.output_dir,
            prefix=args.prefix,
            top_k=args.top_k,
            threads=args.threads,
            graphmask=args.graphmask,
        )

    # --- Benchmark comparison ---
    if args.engine == 'benchmark' and prime_results and optimus_results:
        run_comparison(prime_results, optimus_results, args.output_dir, args.prefix)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f" TxOptimus run complete!")
    print(f" Output directory: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
