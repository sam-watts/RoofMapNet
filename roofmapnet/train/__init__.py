"""
RoofMapNet training module.

This module provides training utilities for the RoofMapNet model:
- PyTorch Lightning training pipeline
- Data preprocessing for RID2 format
- Configuration management
"""

from .main import (
    RoofMapNetLightningModule,
    RoofMapNetDataModule,
    train,
)
from .preprocess_rid2 import (
    preprocess_roof_lines,
    generate_all_negative_lines,
)

__all__ = [
    'RoofMapNetLightningModule',
    'RoofMapNetDataModule',
    'train',
    'preprocess_roof_lines',
    'generate_all_negative_lines',
]
