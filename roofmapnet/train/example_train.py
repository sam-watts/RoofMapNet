"""
Example script demonstrating how to train RoofMapNet.

This script shows:
1. How to prepare data
2. How to configure and run training
3. How to monitor training progress
"""

import json
from pathlib import Path
import numpy as np

from roofmapnet.train import preprocess_roof_lines, train


def prepare_example_data(json_file: str, output_dir: str, image_dir: str):
    """
    Prepare training data from a JSON file containing line annotations.
    
    Args:
        json_file: Path to JSON file with line annotations
        output_dir: Directory to save preprocessed .npz files
        image_dir: Directory containing the source images
    
    Expected JSON format:
    [
        {
            "filename": "image_001.png",
            "lines": [
                [y1, x1, y2, x2],
                [y3, x3, y4, x4],
                ...
            ]
        },
        ...
    ]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_dir)
    
    print(f"Loading annotations from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} images...")
    for idx, item in enumerate(data):
        filename = item['filename']
        lines = item['lines']
        
        # Convert lines from [y1, x1, y2, x2] to [[(y1, x1), (y2, x2)], ...]
        line_segments = [
            [(line[0], line[1]), (line[2], line[3])]
            for line in lines
        ]
        
        # Generate output filename
        base_name = Path(filename).stem
        output_file = output_path / f"{base_name}_label.npz"
        
        # Preprocess and save
        preprocess_roof_lines(
            lines=line_segments,
            output_filename=str(output_file),
            image_size=512,
            heatmap_size=128,
        )
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(data)} images...")
    
    print(f"Data preparation complete! Files saved to {output_dir}")


def train_model_example():
    """
    Example of training RoofMapNet with custom configuration.
    """
    print("Starting RoofMapNet training...")
    
    # Option 1: Train with default config
    train(
        config_path='config/config.yaml',
        data_root='data/',  # Should contain train/ and valid/ subdirectories
        gpus=1,
        precision='32',
        max_epochs=30,
    )
    
    print("Training complete!")


def train_with_custom_config():
    """
    Example of training with programmatically modified configuration.
    """
    from roofmapnet.train import RoofMapNetLightningModule, RoofMapNetDataModule
    from roofmapnet.config import M, C
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    import os
    
    # Set custom configuration
    M.batch_size = 4  # Smaller batch size for limited GPU memory
    M.depth = 4
    M.num_stacks = 2
    M.num_blocks = 1
    
    C.lr = 0.0005  # Lower learning rate
    C.max_epoch = 50
    C.num_workers = 4
    
    # Initialize data module
    data_module = RoofMapNetDataModule(
        data_root='data/',
        batch_size=M.batch_size,
        batch_size_eval=1,
        num_workers=C.num_workers,
    )
    
    # Initialize model with custom loss weights
    model = RoofMapNetLightningModule(
        depth=M.depth,
        head=[[2], [1], [2]],
        num_stacks=M.num_stacks,
        num_blocks=M.num_blocks,
        num_classes=5,
        lr=C.lr,
        loss_weights={
            'gauss': 1.5,   # Emphasize Gaussian heatmap (focal loss)
            'qcc': 1.2,     # Quadratic coordinate calibration
            'lhr': 1.0,     # Line heatmap regression
            'off': 0.3,     # Coordinate offset
            'lv': 2.5,      # Emphasize line validation
            'sf': 0.5,      # Segmentation factor
        }
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='logs/checkpoints',
            filename='roofmapnet-{epoch:02d}-{val/total_loss:.4f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=5,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir='logs/',
        name='roofmapnet_custom',
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=C.max_epoch,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Use mixed precision
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )
    
    # Train
    print("Starting training with custom configuration...")
    trainer.fit(model, data_module)
    print("Training complete!")


def quick_start():
    """
    Quick start example: prepare data and train.
    """
    print("=== RoofMapNet Training Quick Start ===\n")
    
    # Step 1: Prepare training data
    print("Step 1: Preparing training data...")
    prepare_example_data(
        json_file='data/train.json',
        output_dir='data/train/',
        image_dir='data/images/',
    )
    
    # Step 2: Prepare validation data
    print("\nStep 2: Preparing validation data...")
    prepare_example_data(
        json_file='data/valid.json',
        output_dir='data/valid/',
        image_dir='data/images/',
    )
    
    # Step 3: Train model
    print("\nStep 3: Training model...")
    train_model_example()
    
    print("\n=== Quick Start Complete ===")
    print("View training logs with: tensorboard --logdir logs/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RoofMapNet training examples')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['prepare', 'train', 'custom', 'quickstart'],
        default='quickstart',
        help='Which example to run'
    )
    parser.add_argument(
        '--json-file',
        type=str,
        default='data/train.json',
        help='Path to JSON file with annotations (for prepare mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/train/',
        help='Output directory for preprocessed data (for prepare mode)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/images/',
        help='Directory containing images (for prepare mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        prepare_example_data(args.json_file, args.output_dir, args.image_dir)
    elif args.mode == 'train':
        train_model_example()
    elif args.mode == 'custom':
        train_with_custom_config()
    elif args.mode == 'quickstart':
        quick_start()
