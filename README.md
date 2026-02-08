# alpha_fold_viewer

**AF3 ZIP → Standalone HTML Report** — a single-file tool that converts AlphaFold3 output ZIP files into beautiful, self-contained HTML reports.

## What it does

Takes an AlphaFold3 prediction ZIP file and produces a single HTML file containing:

- **Input summary** — sequences, chain types, lengths
- **Confidence overview** — all models ranked by `ranking_score` with ipTM, pTM, fraction disordered, clash status
- **PAE heatmaps** — full predicted aligned error matrices with chain boundaries (embedded as base64 images)
- **Interface analysis** — inter-chain contacts with residue counts, mean PAE, pLDDT, and high-confidence percentages
- **Per-model details** — collapsible sections with chain info and interface residue ranges

The HTML is fully standalone — all images are embedded as data URIs, CSS is inline, no external dependencies. Open it in any browser, share via email, or include in presentations.

## Installation

```bash
git clone https://github.com/aglabx/alpha_fold_viewer.git
cd alpha_fold_viewer
pip install -r requirements.txt
```

Requirements: Python 3.8+, numpy, scipy, matplotlib.

## Usage

```bash
# Basic usage — generates fold_tigd4_dimer_report.html
python af3_report.py fold_tigd4_dimer.zip

# Custom output path
python af3_report.py fold_tigd4_dimer.zip -o my_report.html

# Stricter contact distance (default: 8.0 Å)
python af3_report.py fold_tigd4_dimer.zip --contact-dist 6.0

# Keep extracted temp files for debugging
python af3_report.py fold_tigd4_dimer.zip --keep-tmp
```

### CLI Reference

```
python af3_report.py INPUT_ZIP [-o OUTPUT_HTML] [--contact-dist 8.0] [--keep-tmp]

Positional:
  INPUT_ZIP          Path to AlphaFold3 output ZIP file

Options:
  -o, --output       Output HTML file (default: {zip_name}_report.html)
  --contact-dist     Inter-atomic contact threshold in Å (default: 8.0)
  --keep-tmp         Keep extracted temporary files
```

## Output Description

### Confidence Overview

Models are sorted by `ranking_score` (highest first). The best model is highlighted in green. Columns:

| Column | Description |
|--------|-------------|
| Ranking Score | AF3 composite confidence metric (higher = better) |
| ipTM | Interface predicted TM-score (0–1, higher = better interface) |
| pTM | Predicted TM-score for overall structure |
| Frac. Disordered | Fraction of residues predicted as disordered |
| Clash | Whether the model has steric clashes |

### Interface Analysis

For multi-chain models, inter-chain contacts are detected using a KDTree spatial search. Each interface reports:

| Metric | Description |
|--------|-------------|
| Res. A / Res. B | Number of residues at the interface per chain |
| Atom Contacts | Total inter-chain atom pairs within contact distance |
| pLDDT A / pLDDT B | Mean predicted local confidence at interface residues |
| Avg PAE | Mean predicted aligned error across interface residue pairs |
| PAE <10Å | Percentage of PAE values below 10Å (higher = more confident) |
| High-conf | Percentage of contacts where both atoms have pLDDT ≥ 70 |

### PAE Heatmaps

Each model gets a full PAE matrix heatmap with chain boundary lines. The colormap runs from dark blue (low PAE = high confidence) through green/yellow to red (high PAE = low confidence). Scale: 0–30 Å.

For multi-chain models, per-interface sub-matrices are also shown with mean PAE and <10Å percentage annotations.

## How it works

1. Extracts the AF3 ZIP to a temporary directory
2. Auto-discovers model files (`*_model_*.cif`, `*_full_data_*.json`, `*_summary_confidences_*.json`)
3. Parses mmCIF structures to extract atom coordinates, chain IDs, pLDDT values
4. Loads PAE matrices from full_data JSONs
5. Detects inter-chain interfaces using scipy KDTree
6. Cross-references interfaces with PAE data
7. Generates PAE heatmaps in-memory using matplotlib (→ base64 PNGs)
8. Assembles everything into a single standalone HTML file

## License

MIT
