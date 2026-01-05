# Data Preparation Guide

This guide explains how to prepare training data for RoofMapNet from the RID2 (Roof Information Dataset 2) edge labels.

## Quick Start

### Using Just (Recommended)

```bash
# Prepare data with default settings
just prepare-data

# Prepare data with visualizations
just prepare-data-viz

# Prepare data with custom paths
just prepare-data-custom /path/to/input /path/to/output
```

### Using Python directly

```bash
# Basic usage (processes and splits data)
python -m roofmapnet.train.prepare_data

# With visualization
python -m roofmapnet.train.prepare_data --visualize

# With custom paths
python -m roofmapnet.train.prepare_data \
    --input-dir /path/to/edge_labels \
    --output-dir /path/to/output
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--input-dir` | Path | `~/datasets/roof_information_dataset_2/preprocessing_output/edge_labels` | Directory containing input NPZ files with edge labels |
| `--output-dir` | Path | `rid2_edges` | Output directory for processed data |
| `--train-split` | Float | `0.7` | Fraction of data to use for training |
| `--valid-split` | Float | `0.15` | Fraction of data to use for validation |
| `--test-split` | Float | `0.15` | Fraction of data to use for testing |
| `--seed` | Int | `42` | Random seed for reproducible splits |
| `--visualize` | Flag | False | Generate visualizations of processed data |
| `--skip-processing` | Flag | False | Skip preprocessing step (only split existing data) |
| `--skip-split` | Flag | False | Skip splitting step (only preprocess data) |

## Examples

### Custom split ratios

```bash
python -m roofmapnet.train.prepare_data \
    --train-split 0.8 \
    --valid-split 0.1 \
    --test-split 0.1
```

### Only preprocess (no splitting)

```bash
python -m roofmapnet.train.prepare_data --skip-split
```

### Only split existing data

```bash
python -m roofmapnet.train.prepare_data --skip-processing
```

### Re-split with different ratios

```bash
# First, move all files back to root
mv rid2_edges/train/*.npz rid2_edges/
mv rid2_edges/valid/*.npz rid2_edges/
mv rid2_edges/test/*.npz rid2_edges/

# Then re-split with new ratios
python -m roofmapnet.train.prepare_data \
    --skip-processing \
    --train-split 0.8 \
    --valid-split 0.15 \
    --test-split 0.05
```

## Output Structure

After running the script, your output directory will have the following structure:

```
rid2_edges/
├── train/
│   ├── file1.npz
│   ├── file2.npz
│   └── ...
├── valid/
│   ├── file3.npz
│   └── ...
└── test/
    ├── file4.npz
    └── ...
```

Each NPZ file contains:
- `jmap`: Junction heatmap [n_jtyp, H, W]
- `joff`: Junction offset [n_jtyp, 2, H, W]
- `lmap`: Line heatmap [H, W]
- `junc`: Junction coordinates [N, 3]
- `Lpos`: Positive line adjacency [M, 2]
- `Lneg`: Negative line adjacency [M, 2]
- `lpos`: Positive line coordinates [Np, 2, 3]
- `lneg`: Negative line coordinates [Nn, 2, 3]

## Visualization

When using `--visualize`, the script generates a PNG file showing:
- Junction heatmap
- Junction offsets (X and Y)
- Line heatmap

This is saved as `visualization_{filename}.png` in the output directory.
