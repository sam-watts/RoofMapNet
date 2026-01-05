# RoofMapNet Training Module

This module provides a complete training pipeline for RoofMapNet using PyTorch Lightning.

## Overview

The training module consists of:

- **`main.py`**: PyTorch Lightning training script with:
  - `RoofMapNetLightningModule`: Lightning module handling model, loss computation, and optimization
  - `RoofMapNetDataModule`: Data loading and preprocessing
  - `train()`: Main training function
  - CLI interface for easy training execution

- **`preprocess_rid2.py`**: Data preprocessing utilities for converting roof line annotations to RoofMapNet format

## Installation

First, install PyTorch Lightning (if not already installed):

```bash
pip install pytorch-lightning
```

Or add to your `requirements.txt`:
```
pytorch-lightning>=2.0.0
```

## Data Preparation

The training script expects data in the following structure:

```
data_root/
├── train/
│   ├── image_001.png
│   ├── image_001_label.npz
│   ├── image_002.png
│   ├── image_002_label.npz
│   └── ...
└── valid/
    ├── image_001.png
    ├── image_001_label.npz
    └── ...
```

### Data Format

Each `*_label.npz` file should contain:

- `jmap`: Junction heat maps `[n_jtyp, H, W]` - Binary heatmap indicating junction locations
- `joff`: Junction offsets `[n_jtyp, 2, H, W]` - Sub-pixel offsets for precise junction localization
- `lmap`: Line heat map `[H, W]` - Anti-aliased line rendering
- `junc`: Junction coordinates `[N_junc, 3]` - Array of (y, x, type)
- `Lpos`: Positive line connectivity `[N_pos, 2]` - Junction index pairs for actual lines
- `Lneg`: Negative line connectivity `[N_neg, 2]` - Junction index pairs for non-lines
- `lpos`: Positive line coordinates `[N_pos, 2, 3]` - Actual line endpoints with types
- `lneg`: Negative line coordinates `[N_neg, 2, 3]` - Non-line junction pairs

### Preprocessing Data

To convert your line annotations to the RoofMapNet format:

```python
from roofmapnet.train.preprocess_rid2 import preprocess_roof_lines

# Your line data: list of line segments
lines = [
    [(y1, x1), (y2, x2)],  # Line 1
    [(y3, x3), (y4, x4)],  # Line 2
    # ...
]

# Convert to RoofMapNet format
preprocess_roof_lines(
    lines=lines,
    output_filename="output/image_001_label.npz",
    image_size=512,      # Original image size
    heatmap_size=128     # Heatmap size (typically 1/4 of image size)
)
```

## Training

### Basic Usage

Train with default configuration:

```bash
python -m roofmapnet.train.main \
    --config config/config.yaml \
    --data-root /path/to/data
```

### Advanced Options

```bash
python -m roofmapnet.train.main \
    --config config/config.yaml \
    --data-root /path/to/data \
    --gpus 1 \
    --precision 16 \
    --max-epochs 30 \
    --resume-from logs/checkpoints/last.ckpt
```

### Command Line Arguments

- `--config`: Path to configuration file (default: `config/config.yaml`)
- `--data-root`: Root directory containing train/ and valid/ subdirectories
- `--gpus`: Number of GPUs to use (0 for CPU, default: 1)
- `--precision`: Training precision - '32', '16', or 'bf16' (default: '32')
- `--max-epochs`: Maximum training epochs (overrides config)
- `--resume-from`: Path to checkpoint to resume training from

### Programmatic Usage

```python
from roofmapnet.train.main import train

train(
    config_path='config/config.yaml',
    data_root='/path/to/data',
    gpus=1,
    precision='16',
    max_epochs=30,
)
```

## Configuration

Edit `config/config.yaml` to customize training:

```yaml
io:
  logdir: logs/                    # Output directory for logs and checkpoints
  datadir: /data/roofmapset/      # Default data directory
  num_workers: 8                   # Number of data loading workers
  validation_interval: 1000        # Validation frequency (steps)

model:
  batch_size: 8                    # Training batch size
  batch_size_eval: 1               # Validation batch size
  
  # Model architecture
  depth: 4                         # Hourglass network depth
  num_stacks: 2                    # Number of hourglass stacks
  num_blocks: 1                    # Residual blocks per stack
  
  # Sampling parameters
  n_stc_posl: 300                  # Static positive line samples
  n_stc_negl: 40                   # Static negative line samples
  n_dyn_junc: 300                  # Dynamic junction samples
  
  # Network dimensions
  dim_loi: 128                     # Line of interest feature dimension
  dim_fc: 1024                     # Fully connected layer dimension

optim:
  name: Adam
  lr: 0.001                        # Learning rate
  weight_decay: 0                  # L2 regularization
  amsgrad: True                    # Use AMSGrad variant
  max_epoch: 30                    # Maximum training epochs
  lr_decay_epoch: 10               # LR decay interval
```

## Model Architecture

The RoofMapNet model consists of:

1. **Backbone**: Stacked hourglass network for feature extraction
2. **Multi-task Heads**:
   - Junction detection (heatmap + offset)
   - Line segmentation (heatmap)
3. **Line Vectorizer**: Classification network for line verification

## Loss Functions

The training script uses a multi-task loss based on the paper, with six distinct subtasks:

```python
total_loss = λ_gauss * L_gauss +    # Dynamic Gaussian heatmap (Focal Loss)
             λ_qcc * L_qcc +        # Quadratic coordinate calibration (BCE)
             λ_lhr * L_lhr +        # Line heatmap regression (BCE)
             λ_off * L_off +        # Coordinate offset regression (L2)
             λ_lv * L_lv +          # Line validation (BCE)
             λ_sf * L_sf            # Segmentation factor regression (L2)
```

### Loss Components

1. **L_gauss** (Focal Loss): Dynamic Gaussian heatmap regression
   - Addresses sparse junction distribution and class imbalance
   - Formula: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
   - Parameters: α=2.0 (balances pos/neg), γ=4.0 (reduces easy sample weight)

2. **L_qcc** (BCE): Quadratic coordinate calibration
   - Binary classification for junction detection on heatmap
   
3. **L_lhr** (BCE): Line heatmap regression
   - Binary classification for line segmentation
   
4. **L_off** (L2/MSE): Coordinate offset regression
   - Sub-pixel junction localization (masked to junction locations)
   
5. **L_lv** (BCE): Line validation
   - Classification of positive vs negative lines
   
6. **L_sf** (L2/MSE): Segmentation factor regression
   - Line segmentation features

Default weights: `{gauss: 1.0, qcc: 1.0, lhr: 1.0, off: 0.25, lv: 2.0, sf: 0.5}`

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser.

### Logged Metrics

- `train/jmap_loss`: Junction heatmap loss
- `train/joff_loss`: Junction offset loss
- `train/lmap_loss`: Line heatmap loss
- `train/line_cls_loss`: Line classification loss
- `train/total_loss`: Weighted total loss
- `val/*_loss`: Validation metrics
- Learning rate per epoch

## Checkpointing

Checkpoints are automatically saved to `{logdir}/checkpoints/`:

- `last.ckpt`: Most recent checkpoint
- `roofmapnet-epoch=XX-val_total_loss=X.XXXX.ckpt`: Top 3 best models

To resume training:

```bash
python -m roofmapnet.train.main \
    --resume-from logs/checkpoints/last.ckpt \
    --data-root /path/to/data
```

## Validation

Validation runs automatically during training at intervals specified by `validation_interval` in the config.

## Tips for Better Training

1. **Data augmentation**: The dataset includes horizontal flipping (see `datasets.py`)
2. **Batch size**: Adjust based on GPU memory (reduce if OOM errors occur)
3. **Learning rate**: Start with 0.001, reduce if loss plateaus
4. **Mixed precision**: Use `--precision 16` to reduce memory usage and speed up training
5. **GPU utilization**: Increase `num_workers` for faster data loading

## Inference

After training, use the trained model for inference:

```python
import torch
from roofmapnet.models.roofmapnet import RoofMapNet
from roofmapnet.train.main import RoofMapNetLightningModule

# Load trained model
checkpoint = torch.load('logs/checkpoints/last.ckpt')
model = RoofMapNetLightningModule.load_from_checkpoint(
    'logs/checkpoints/last.ckpt'
)
model.eval()

# Or use the existing inference.py script
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Use `--precision 16` for mixed precision training
- Reduce `num_workers`

### Training is slow

- Increase `num_workers` for faster data loading
- Use `--precision 16` for faster computation
- Enable pin_memory (already enabled by default)

### Loss is NaN

- Reduce learning rate
- Check data preprocessing
- Enable gradient clipping (already at 1.0)

### Poor validation performance

- Train for more epochs
- Adjust loss weights
- Check data quality and preprocessing
- Try different learning rate schedules

## Example Training Session

```bash
# 1. Prepare data
python -m roofmapnet.train.preprocess_rid2 \
    --input data/train.json \
    --output data/train/

# 2. Start training
python -m roofmapnet.train.main \
    --config config/config.yaml \
    --data-root data/ \
    --gpus 1 \
    --precision 16

# 3. Monitor with TensorBoard
tensorboard --logdir logs/

# 4. Resume if interrupted
python -m roofmapnet.train.main \
    --resume-from logs/checkpoints/last.ckpt \
    --data-root data/
```

## License

Same as the parent RoofMapNet project.
