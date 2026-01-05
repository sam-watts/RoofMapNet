"""
Data preparation script for RoofMapNet training.

This script processes raw edge labels from RID2 dataset and prepares them
for training by:
1. Converting edge labels to RoofMapNet format (NPZ files)
2. Splitting data into train/valid/test sets
3. Optionally visualizing the processed data
"""

from pathlib import Path
import random
import shutil

import click
import numpy as np
from tqdm import tqdm

from roofmapnet.train.preprocess_rid2 import preprocess_roof_lines


@click.command()
@click.option(
    '--input-dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default='/Users/swatts/datasets/roof_information_dataset_2/preprocessing_output/edge_labels',
    help='Directory containing input NPZ files with edge labels'
)
@click.option(
    '--output-dir',
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default='rid2_edges',
    help='Output directory for processed data'
)
@click.option(
    '--train-split',
    type=float,
    default=0.7,
    help='Fraction of data to use for training (default: 0.7)'
)
@click.option(
    '--valid-split',
    type=float,
    default=0.15,
    help='Fraction of data to use for validation (default: 0.15)'
)
@click.option(
    '--test-split',
    type=float,
    default=0.15,
    help='Fraction of data to use for testing (default: 0.15)'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    help='Random seed for reproducible splits (default: 42)'
)
@click.option(
    '--visualize',
    is_flag=True,
    help='Generate visualizations of processed data'
)
@click.option(
    '--skip-processing',
    is_flag=True,
    help='Skip preprocessing step (only split existing data)'
)
@click.option(
    '--skip-split',
    is_flag=True,
    help='Skip splitting step (only preprocess data)'
)
def prepare_data(
    input_dir: Path,
    output_dir: Path,
    train_split: float,
    valid_split: float,
    test_split: float,
    seed: int,
    visualize: bool,
    skip_processing: bool,
    skip_split: bool,
):
    """Prepare RoofMapNet training data from RID2 edge labels.
    
    This script will:
    1. Convert edge labels to RoofMapNet format (unless --skip-processing)
    2. Split data into train/valid/test sets (unless --skip-split)
    3. Optionally visualize the data (if --visualize)
    """
    
    # Validate splits sum to 1.0
    total_split = train_split + valid_split + test_split
    if not (0.99 <= total_split <= 1.01):
        raise click.BadParameter(
            f"Splits must sum to 1.0 (got {total_split}). "
            f"Adjust --train-split, --valid-split, and --test-split."
        )
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # ===== Step 1: Preprocess edge labels =====
    if not skip_processing:
        click.echo(f"\n{'='*60}")
        click.echo("Step 1: Preprocessing edge labels")
        click.echo(f"{'='*60}")
        click.echo(f"Input directory:  {input_dir}")
        click.echo(f"Output directory: {output_dir}")
        
        input_files = list(input_dir.glob("*.npz"))
        if not input_files:
            raise click.ClickException(f"No NPZ files found in {input_dir}")
        
        click.echo(f"Found {len(input_files)} files to process")
        
        with click.progressbar(
            input_files,
            label='Processing files',
            show_pos=True
        ) as files:
            for p in files:
                try:
                    edges = np.load(p)["edges"]
                    preprocess_roof_lines(edges, output_dir / p.stem)
                except Exception as e:
                    click.echo(f"\nWarning: Failed to process {p.name}: {e}", err=True)
        
        click.echo(f"✓ Preprocessing complete!")
    else:
        click.echo("Skipping preprocessing step (--skip-processing)")
    
    # ===== Step 2: Split into train/valid/test =====
    if not skip_split:
        click.echo(f"\n{'='*60}")
        click.echo("Step 2: Splitting data into train/valid/test")
        click.echo(f"{'='*60}")
        
        # Get all npz files in output_dir (not in subdirectories)
        all_files = sorted([f for f in output_dir.glob("*.npz") if f.is_file()])
        
        if not all_files:
            raise click.ClickException(
                f"No NPZ files found in {output_dir}. "
                f"Run without --skip-processing first."
            )
        
        click.echo(f"Total files: {len(all_files)}")
        click.echo(f"Split ratios: Train={train_split:.1%}, Valid={valid_split:.1%}, Test={test_split:.1%}")
        click.echo(f"Random seed: {seed}")
        
        # Shuffle with fixed seed for reproducibility
        random.seed(seed)
        random.shuffle(all_files)
        
        # Calculate split sizes
        n_total = len(all_files)
        n_train = int(train_split * n_total)
        n_valid = int(valid_split * n_total)
        
        train_files = all_files[:n_train]
        valid_files = all_files[n_train:n_train + n_valid]
        test_files = all_files[n_train + n_valid:]
        
        click.echo(f"Split sizes: Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}")
        
        # Create subdirectories
        train_dir = output_dir / "train"
        valid_dir = output_dir / "valid"
        test_dir = output_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        valid_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Move files to respective directories
        def move_files(files, target_dir, split_name):
            with click.progressbar(
                files,
                label=f'Moving to {split_name}',
                show_pos=True
            ) as file_list:
                for f in file_list:
                    try:
                        shutil.move(str(f), str(target_dir / f.name))
                    except Exception as e:
                        click.echo(f"\nWarning: Failed to move {f.name}: {e}", err=True)
        
        move_files(train_files, train_dir, "train")
        move_files(valid_files, valid_dir, "valid")
        move_files(test_files, test_dir, "test")
        
        click.echo(f"\n✓ Data split complete!")
        click.echo(f"  Train: {len(list(train_dir.glob('*.npz')))} files in {train_dir}")
        click.echo(f"  Valid: {len(list(valid_dir.glob('*.npz')))} files in {valid_dir}")
        click.echo(f"  Test:  {len(list(test_dir.glob('*.npz')))} files in {test_dir}")
    else:
        click.echo("Skipping split step (--skip-split)")
    
    # ===== Step 3: Visualize (optional) =====
    if visualize:
        click.echo(f"\n{'='*60}")
        click.echo("Step 3: Generating visualizations")
        click.echo(f"{'='*60}")
        
        import matplotlib.pyplot as plt
        
        # Find a sample file to visualize
        sample_file = None
        for subdir in ['train', 'valid', 'test', '.']:
            search_dir = output_dir / subdir if subdir != '.' else output_dir
            npz_files = list(search_dir.glob("*.npz"))
            if npz_files:
                sample_file = npz_files[0]
                break
        
        if sample_file is None:
            click.echo("No NPZ files found to visualize", err=True)
            return
        
        click.echo(f"Visualizing: {sample_file}")
        
        data = np.load(sample_file)
        
        # Display keys and shapes
        click.echo("\nData structure:")
        for k in data.keys():
            click.echo(f"  {k}: {data[k].shape}")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Data Visualization: {sample_file.name}", fontsize=14)
        
        # Junction heatmap
        ax = axes[0, 0]
        im = ax.imshow(data["jmap"][0], cmap='hot')
        ax.set_title("Junction Heatmap (jmap)")
        plt.colorbar(im, ax=ax)
        
        # Junction offset (x)
        ax = axes[0, 1]
        im = ax.imshow(data["joff"][0][0], cmap='coolwarm')
        ax.set_title("Junction Offset X (joff[0])")
        plt.colorbar(im, ax=ax)
        
        # Junction offset (y)
        ax = axes[1, 0]
        im = ax.imshow(data["joff"][0][1], cmap='coolwarm')
        ax.set_title("Junction Offset Y (joff[1])")
        plt.colorbar(im, ax=ax)
        
        # Line heatmap
        ax = axes[1, 1]
        im = ax.imshow(data["lmap"], cmap='hot')
        ax.set_title("Line Heatmap (lmap)")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / f"visualization_{sample_file.stem}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        click.echo(f"✓ Visualization saved to: {viz_path}")
        
        # Show interactively if possible
        try:
            plt.show()
        except Exception:
            pass  # Headless environment
    
    click.echo(f"\n{'='*60}")
    click.echo("✓ All steps complete!")
    click.echo(f"{'='*60}")


if __name__ == '__main__':
    prepare_data()
