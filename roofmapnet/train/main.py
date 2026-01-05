"""
PyTorch Lightning training script for RoofMapNet model.

This module provides a complete training pipeline using PyTorch Lightning,
including the LightningModule, DataModule, and training loop.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from roofmapnet.config import M, C
from roofmapnet.datasets import WireframeDataset, collate
from roofmapnet.models.roofmapnet import RoofMapNet


class RoofMapNetLightningModule(pl.LightningModule):
    """PyTorch Lightning module for RoofMapNet training and validation.
    
    This module implements the multi-task learning framework described in the paper,
    with six distinct loss subtasks:
    
    1. L_gauss: Dynamic Gaussian heatmap regression (Focal Loss)
       - Addresses sparse junction distribution and sample imbalance
    2. L_qcc: Quadratic coordinate calibration (BCE)
       - Junction heatmap binary classification
    3. L_lhr: Line heatmap regression (BCE)
       - Line segmentation heatmap
    4. L_off: Coordinate offset regression (L2/MSE)
       - Sub-pixel junction localization
    5. L_lv: Line validation (BCE)
       - Positive vs negative line classification
    6. L_sf: Segmentation factor regression (L2/MSE)
       - Line segmentation features
    
    Total loss: L_total = Σ λ_i * L_i for i ∈ {gauss, qcc, lhr, off, lv, sf}
    """
    
    def __init__(
        self,
        depth: int = 4,
        head: List[List[int]] = [[2], [1], [2]],
        num_stacks: int = 2,
        num_blocks: int = 1,
        num_classes: int = 5,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        lr_decay_epoch: int = 10,
        max_epoch: int = 30,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the RoofMapNet Lightning module.
        
        Args:
            depth: Hourglass network depth
            head: Multi-task head configuration
            num_stacks: Number of hourglass stacks
            num_blocks: Number of residual blocks per stack
            num_classes: Total number of output channels
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            lr_decay_epoch: Epoch interval for learning rate decay
            max_epoch: Maximum training epochs
            loss_weights: Dictionary of loss weights for different tasks
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = RoofMapNet(
            depth=depth,
            head=head,
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            num_classes=num_classes,
        )
        
        # Loss weights - based on the paper's multi-task loss function
        if loss_weights is None:
            self.loss_weights = {
                'gauss': 8.0,    # L_gauss: Dynamic Gaussian heatmap regression (focal loss)
                'qcc': 10.0,      # L_qcc: Quadratic coordinate calibration (BCE)
                'lhr': 20.0,      # L_lhr: Line heatmap regression (BCE)
                'off': 1,     # L_off: Coordinate offset regression (L2)
                'lv': 1.0,       # L_lv: Line validation (BCE)
                'sf': 1.0,       # L_sf: Segmentation factor regression (L2)
            }
        else:
            self.loss_weights = loss_weights
        
        # Focal loss parameters for Gaussian heatmap
        self.focal_alpha = 2.0  # α_t: balances positive/negative samples
        self.focal_gamma = 4.0  # γ: reduces weight of easy samples
        
        # For logging
        self.validation_outputs = []
    
    def forward(self, batch):
        """Forward pass through the model.
        
        Args:
            batch: Tuple of (images, meta_list, targets)
            
        Returns:
            Model output dictionary
        """
        images, meta_list, targets = batch
        
        input_dict = {
            'image': images,
            'meta': meta_list,
            'target': targets,
            'mode': 'training' if self.training else 'validation',
        }
        
        return self.model(input_dict)
    
    def focal_loss(self, pred, target, alpha=2.0, gamma=4.0):
        """Focal loss for addressing class imbalance.
        
        Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
        
        Args:
            pred: Predicted probabilities [B, C, H, W]
            target: Ground truth [B, C, H, W]
            alpha: Weight coefficient balancing positive/negative samples
            gamma: Modulation factor to reduce weight of easy samples
            
        Returns:
            Focal loss value
        """
        # Compute p_t: probability of correct classification
        p_t = pred * target + (1 - pred) * (1 - target)
        
        # Focal loss formula
        focal_weight = alpha * torch.pow(1 - p_t, gamma)
        loss = -focal_weight * torch.log(p_t.clamp(min=1e-8))
        
        return loss.mean()
    
    def compute_loss(self, result, targets, meta_list):
        """Compute multi-task loss for RoofMapNet based on the paper.
        
        The total loss is computed as a weighted sum of six distinct subtasks:
        L_total = λ_gauss*L_gauss + λ_qcc*L_qcc + λ_lhr*L_lhr + 
                  λ_off*L_off + λ_lv*L_lv + λ_sf*L_sf
        
        Loss functions:
        - L_gauss: Focal loss for dynamic Gaussian heatmap regression
        - L_qcc: BCE for quadratic coordinate calibration  
        - L_lhr: BCE for line heatmap regression
        - L_off: L2 loss for coordinate offset regression
        - L_lv: BCE for line validation
        - L_sf: L2 loss for segmentation factor regression
        
        Args:
            result: Model output dictionary containing predictions
            targets: Ground truth targets dictionary
            meta_list: List of metadata for each sample
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        losses = {}
        preds = result['preds']
        device = next(iter(preds.values())).device
        
        # Use targets from model result if available (contains gaussjmap)
        # Otherwise use the original targets from batch
        if 'targets' in result:
            targets = result['targets']
        
        # Prepare ground truth tensors with proper dimensions
        # Check if targets are already permuted (from model) or need permuting (from batch)
        if targets['jmap'].dim() == 4 and targets['jmap'].shape[0] == 1:
            # Already in CNHW format from model [1, B, H, W]
            jmap_gt = targets['jmap'][0]  # [B, H, W]
            # Add junction type dimension to match predictions
            jmap_gt = jmap_gt.unsqueeze(1)  # [B, 1, H, W]
        elif targets['jmap'].dim() == 3:
            # Shape [B, H, W], add junction type dimension
            jmap_gt = targets['jmap'].unsqueeze(1)  # [B, 1, H, W]
        else:
            # Original JBHW format from batch [n_jtyp, B, H, W]
            jmap_gt = targets['jmap'].permute(1, 0, 2, 3)  # [B, n_jtyp, H, W]
        
        if targets['joff'].dim() == 5 and targets['joff'].shape[0] == 1:
            # Already in CNHW format from model [1, 2, B, H, W]
            joff_gt = targets['joff'][0]  # [2, B, H, W]
            joff_gt = joff_gt.permute(1, 0, 2, 3).unsqueeze(1)  # [B, 1, 2, H, W]
        elif targets['joff'].dim() == 4:
            # Shape [B, 2, H, W], add junction type dimension
            joff_gt = targets['joff'].unsqueeze(1)  # [B, 1, 2, H, W]
        else:
            # Original format from batch [n_jtyp, 2, B, H, W]
            joff_gt = targets['joff'].permute(2, 0, 1, 3, 4)  # [B, n_jtyp, 2, H, W]
        
        # Get predictions
        jmap_pred = preds['jmap']  # [B, n_jtyp, H, W]
        joff_pred = preds['joff']  # [B, n_jtyp, 2, H, W]
        lmap_pred = preds['lmap']  # [B, H, W]
        lmap_gt = targets['lmap']  # [B, H, W]
        
        # ===== 1. L_gauss: Dynamic Gaussian heatmap regression (Focal Loss) =====
        # The Gaussian heatmap from targets['gaussjmap'] if available, else use jmap
        if 'gaussjmap' in targets:
            if targets['gaussjmap'].dim() == 4 and targets['gaussjmap'].shape[0] == 1:
                # Already in CNHW format from model [1, B, H, W]
                gauss_gt = targets['gaussjmap'][0]  # [B, H, W]
                gauss_gt = gauss_gt.unsqueeze(1)  # [B, 1, H, W]
            elif targets['gaussjmap'].dim() == 3:
                # Shape [B, H, W], add junction type dimension
                gauss_gt = targets['gaussjmap'].unsqueeze(1)  # [B, 1, H, W]
            else:
                # Original format [n_jtyp, B, H, W]
                gauss_gt = targets['gaussjmap'].permute(1, 0, 2, 3)  # [B, n_jtyp, H, W]
        else:
            gauss_gt = jmap_gt
        
        losses['gauss'] = self.focal_loss(
            jmap_pred, gauss_gt, 
            alpha=self.focal_alpha, 
            gamma=self.focal_gamma
        )
        
        # ===== 2. L_qcc: Quadratic coordinate calibration (BCE) =====
        # This is the junction heatmap binary classification
        losses['qcc'] = F.binary_cross_entropy(
            jmap_pred, jmap_gt, reduction='mean'
        )
        
        # ===== 3. L_lhr: Line heatmap regression (BCE) =====
        losses['lhr'] = F.binary_cross_entropy(
            lmap_pred, lmap_gt, reduction='mean'
        )
        
        # ===== 4. L_off: Coordinate offset regression (L2/MSE) =====
        # Only compute loss where junctions exist (masked by junction heatmap)
        jmap_mask = (jmap_gt > 0.5).unsqueeze(2)  # [B, n_jtyp, 1, H, W]
        
        if jmap_mask.sum() > 0:
            # MSE loss (L2 loss) on junction offsets
            joff_pred_masked = joff_pred[jmap_mask.expand_as(joff_pred)]
            joff_gt_masked = joff_gt[jmap_mask.expand_as(joff_gt)]
            losses['off'] = F.mse_loss(joff_pred_masked, joff_gt_masked, reduction='mean')
        else:
            losses['off'] = torch.tensor(0.0, device=device)
        
        # ===== 5. L_lv: Line validation (BCE) =====
        # Line classification: positive vs negative lines
        if 'line_logits' in result:
            line_logits = result['line_logits']  # [N_lines]
            line_labels = []
            
            for meta in meta_list:
                if 'lpre_label' in meta:
                    line_labels.append(meta['lpre_label'])
            
            if line_labels:
                line_labels = torch.cat(line_labels)
                # BCE with logits for line validation
                losses['lv'] = F.binary_cross_entropy_with_logits(
                    line_logits, line_labels, reduction='mean'
                )
            else:
                losses['lv'] = torch.tensor(0.0, device=device)
        else:
            losses['lv'] = torch.tensor(0.0, device=device)
        
        # ===== 6. L_sf: Segmentation factor regression (L2/MSE) =====
        # This corresponds to the line segmentation features (e.g., from xline in detection.py)
        # If available in the model output, compute L2 loss
        if 'seg_factors' in result and 'seg_factors_gt' in targets:
            seg_pred = result['seg_factors']
            seg_gt = targets['seg_factors_gt']
            losses['sf'] = F.mse_loss(seg_pred, seg_gt, reduction='mean')
        else:
            # If segmentation factors are not explicitly available, use a placeholder
            losses['sf'] = torch.tensor(0.0, device=device)
        
        # ===== Total Loss: Weighted sum of all subtasks =====
        total_loss = (
            self.loss_weights['gauss'] * losses['gauss'] +
            self.loss_weights['qcc'] * losses['qcc'] +
            self.loss_weights['lhr'] * losses['lhr'] +
            self.loss_weights['off'] * losses['off'] +
            self.loss_weights['lv'] * losses['lv'] +
            self.loss_weights['sf'] * losses['sf']
        )
        losses['total'] = total_loss
        
        return losses
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: Batch of data (images, meta_list, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        images, meta_list, targets = batch
        
        # Forward pass
        result = self.forward(batch)
        
        # Compute losses
        losses = self.compute_loss(result, targets, meta_list)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'train/{k}_loss', v, on_step=True, on_epoch=True, prog_bar=(k == 'total'))
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Batch of data (images, meta_list, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        images, meta_list, targets = batch
        
        # Forward pass
        result = self.forward(batch)
        
        # Compute losses
        losses = self.compute_loss(result, targets, meta_list)
        
        # Log losses
        for k, v in losses.items():
            self.log(f'val/{k}_loss', v, on_step=False, on_epoch=True, prog_bar=(k == 'total'))
        
        # Store outputs for epoch-level metrics
        self.validation_outputs.append({
            'loss': losses['total'].item(),
            'batch_size': len(images),
        })
        
        return losses['total']
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level validation metrics."""
        if not self.validation_outputs:
            return
        
        # Compute average loss
        total_loss = sum(out['loss'] * out['batch_size'] for out in self.validation_outputs)
        total_samples = sum(out['batch_size'] for out in self.validation_outputs)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        self.log('val/epoch_loss', avg_loss, prog_bar=True)
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and lr_scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
        )
        
        # StepLR: decay learning rate every lr_decay_epoch
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.lr_decay_epoch,
            gamma=0.1,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }


class RoofMapNetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for RoofMapNet.
    
    Handles data loading, preprocessing, and batch preparation for training
    and validation using the preprocessed data format from preprocess_rid2.
    """
    
    def __init__(
        self,
        data_root: str,
        image_dir: Optional[str] = None,
        batch_size: int = 8,
        batch_size_eval: int = 1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        """Initialize the DataModule.
        
        Args:
            data_root: Root directory containing train/ and valid/ subdirectories with NPZ files
            image_dir: Optional directory containing PNG images (if different from data_root)
            batch_size: Batch size for training
            batch_size_eval: Batch size for validation
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.image_dir = Path(image_dir) if image_dir else None
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training and validation.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            # Training dataset
            train_path = self.data_root / 'train'
            if train_path.exists():
                self.train_dataset = WireframeDataset(
                    rootdir=str(self.data_root),
                    split='train',
                    image_dir=str(self.image_dir) if self.image_dir else None
                )
            else:
                raise ValueError(f"Training data not found at {train_path}")
            
            # Validation dataset
            valid_path = self.data_root / 'valid'
            if valid_path.exists():
                self.val_dataset = WireframeDataset(
                    rootdir=str(self.data_root),
                    split='valid',
                    image_dir=str(self.image_dir) if self.image_dir else None
                )
            else:
                raise ValueError(f"Validation data not found at {valid_path}")
    
    def train_dataloader(self):
        """Create training data loader.
        
        Returns:
            DataLoader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
            drop_last=True,  # Drop last incomplete batch
        )
    
    def val_dataloader(self):
        """Create validation data loader.
        
        Returns:
            DataLoader for validation
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate,
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file and update global config.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update global M config with model parameters
    if 'model' in config:
        for key, value in config['model'].items():
            setattr(M, key, value)
    
    # Update global C config with IO parameters
    if 'io' in config:
        for key, value in config['io'].items():
            setattr(C, key, value)
    
    # Update global C config with optimization parameters
    if 'optim' in config:
        for key, value in config['optim'].items():
            setattr(C, key, value)
    
    return config


def train(
    config_path: str = 'config/config.yaml',
    data_root: Optional[str] = None,
    image_dir: Optional[str] = None,
    gpus: int = 1,
    precision: str = '32',
    max_epochs: Optional[int] = None,
    resume_from: Optional[str] = None,
    **kwargs
):
    """Main training function.
    
    Args:
        config_path: Path to configuration file
        data_root: Root directory of training data with NPZ files (overrides config)
        image_dir: Directory containing PNG images (if different from data_root)
        gpus: Number of GPUs to use (0 for CPU)
        precision: Training precision ('32', '16', or 'bf16')
        max_epochs: Maximum training epochs (overrides config)
        resume_from: Path to checkpoint to resume from
        **kwargs: Additional arguments to override config
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command line arguments
    if data_root is not None:
        C.datadir = data_root
    if max_epochs is not None:
        C.max_epoch = max_epochs
    if resume_from is not None:
        C.resume_from = resume_from
    
    # Initialize data module
    data_module = RoofMapNetDataModule(
        data_root=C.datadir,
        image_dir=image_dir,
        batch_size=M.batch_size,
        batch_size_eval=M.batch_size_eval,
        num_workers=C.num_workers,
    )
    
    # Initialize model
    model = RoofMapNetLightningModule(
        depth=M.depth,
        head=M.head_size,
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=sum(sum(M.head_size, [])),
        lr=C.lr,
        weight_decay=C.weight_decay,
        lr_decay_epoch=C.lr_decay_epoch,
        max_epoch=C.max_epoch,
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(C.logdir, 'checkpoints'),
            filename='roofmapnet-{epoch:02d}-{val/total_loss:.4f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Setup logger
    logger = WandbLogger(
        project='roofmapnet',
        save_dir=str(C.logdir),
        log_model=False,  # Log model checkpoints to wandb
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=C.max_epoch,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 'auto',
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=C.get('validation_interval', None),
        gradient_clip_val=1.0,  # Prevent gradient explosion
    )
    
    # Train model
    if C.resume_from:
        trainer.fit(model, data_module, ckpt_path=C.resume_from)
    else:
        trainer.fit(model, data_module)
    
    print(f"\nTraining complete! Checkpoints saved to: {C.logdir}/checkpoints")
    print(f"View training logs at: https://wandb.ai/{logger.experiment.project_name()}")


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RoofMapNet model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root directory containing train/ and valid/ NPZ label files'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Directory containing PNG images (if different from data-root)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use (0 for CPU)'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='32',
        choices=['32', '16', 'bf16'],
        help='Training precision'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_root=args.data_root,
        image_dir=args.image_dir,
        gpus=args.gpus,
        precision=args.precision,
        max_epochs=args.max_epochs,
        resume_from=args.resume_from,
    )


if __name__ == '__main__':
    main()
