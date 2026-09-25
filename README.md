# TxOptimus

**Zero-shot drug repurposing via TxGNN on PrimeKG and OptimusKG.**

TxOptimus is a unified Python package that wraps [TxGNN](https://github.com/mims-harvard/TxGNN) (Huang et al., *Nature Medicine*, 2024) with two knowledge graph backends:

- **TxGNN Prime** — the original baseline trained on Harvard [PrimeKG](https://github.com/mims-harvard/PrimeKG)
- **TxGNN Optimus** — an enriched model trained on the Zitnik Lab's [OptimusKG](https://zitniklab.hms.harvard.edu/projects/OptimusKG/), a multimodal knowledge graph validated by LLM extraction (PaperQA3)

OptimusKG achieves a **~34% improvement in Macro AUROC** over PrimeKG in zero-shot drug repurposing for oncology under the `cell_proliferation` split.

## Installation

```bash
# Create a dedicated environment (recommended)
conda create -n txoptimus python=3.8
conda activate txoptimus

# Install DGL first (not available on PyPI, requires separate wheel)
# For CUDA 12.1:
pip install https://data.dgl.ai/wheels/torch-2.4/cu121/dgl-2.4.0%2Bcu121-cp38-cp38-manylinux1_x86_64.whl
# For CPU only:
# pip install https://data.dgl.ai/wheels/torch-2.4/cpu/dgl-2.4.0-cp38-cp38-manylinux1_x86_64.whl

# Install TxOptimus
pip install txoptimus
```

> **⚠ System Requirements:**
> - RAM: ≥32 GB (OptimusKG loads a 21.8M-edge graph)
> - Storage: ≥8 GB free disk space
> - Python: 3.8+ (tested on 3.8.20)
> - GPU: Optional (CPU inference supported)

## Quick Start

### 1. Download data files

```bash
txoptimus --setup              # downloads both PrimeKG and OptimusKG data
txoptimus --setup prime        # only PrimeKG
txoptimus --setup optimus      # only OptimusKG
```

### 2. Run zero-shot drug repurposing

```bash
# Using OptimusKG (recommended)
txoptimus --engine optimus --diseases "oral cavity cancer" "glioblastoma"

# Using PrimeKG baseline
txoptimus --engine prime --diseases "oral cavity cancer"

# Benchmark mode — runs both and generates a comparison report
txoptimus --engine benchmark --diseases "oral cavity cancer"
```

### 3. Fuzzy disease matching

TxOptimus uses **PubMedBERT embeddings** to fuzzy-match your disease terms against the knowledge graph vocabulary. If your input doesn't match exactly, the tool will show the top-10 closest matches and let you pick:

```
Disease term: "breast cancer"

Matched terms in OPTIMUS:
  [1] invasive breast carcinoma (similarity: 0.94)
  [2] breast carcinoma (similarity: 0.91)
  [3] triple-negative breast cancer (similarity: 0.89)

Select terms (e.g., '1 3' or 'all'): 1 3
```

### 4. Enable explainability & Subgraph Extraction (v0.1.8)

TxOptimus can automatically extract **drug-specific, 2-hop biological subgraphs** (e.g., `Drug -> Target -> Disease`) for your top candidate drugs to explain *why* the model made its prediction.

```bash
# Enable GraphMask for PrimeKG
txoptimus --engine prime --diseases "oral cavity cancer" --graphmask

# OptimusKG is too large for standard GraphMask. Use the --low-memory gradient workaround!
txoptimus --engine optimus --diseases "oral cavity cancer" --graphmask --low-memory
```

> **Note:** Full GraphMask training requires ≥30 GB RAM and takes 2-4 hours on CPU. The `--low-memory` flag for Optimus bypasses this by using gradient attribution and explicit neighborhood intersection.

## Full CLI Reference

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--engine` | str | `optimus` | `prime`, `optimus`, or `benchmark` |
| `--diseases` | str[] | *(required)* | Disease terms (fuzzy matched) |
| `--graphmask` | flag | `False` | Enable GraphMask explainability |
| `--low-memory` | flag | `False` | Use low-memory integrated gradients for Optimus |
| `--output_dir` | str | `./txoptimus_output` | Output directory |
| `--prefix` | str | `txoptimus` | File prefix |
| `--threads` | int | `18` | CPU threads |
| `--top_k` | int | `100` | Drug candidates per disease |
| `--setup` | str? | — | Download data (`prime`/`optimus`/both) |
| `--data_dir` | str | `~/.txoptimus` | Data directory override |

## Output Files

| File | Description |
|:---|:---|
| `{prefix}_{engine}_results.json` | Full metrics (AUROC, AUPRC, per-relation, graph profile) |
| `{prefix}_{engine}_candidates.csv` | Top-K drug candidates ranked by score |
| `{prefix}_comparison_report.md` | Side-by-side comparison (benchmark mode only) |
| `{prefix}_{engine}_drug_subgraphs.csv` | Automated 2-hop biological subgraphs explaining predictions |

## Citation

If you use TxOptimus in your research, please cite the original TxGNN paper:

```bibtex
@article{huang2024txgnn,
  title={Zero-shot prediction of therapeutic use with geometric deep learning and clinician centered design},
  author={Huang, Kexin and Chandak, Payal and Wang, Qianwen and Haber, Shreyas and Zitnik, Marinka},
  journal={Nature Medicine},
  year={2024}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**arsalanriaz38[AT]gmail[DOT]com**
