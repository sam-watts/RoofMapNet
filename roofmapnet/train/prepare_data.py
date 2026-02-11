"""
Unified data preparation script for RoofMapNet training.

Supports multiple source datasets:
- rid2:       RID2 edge-label NPZ files (requires random train/valid/test split)
- roofmapnet: RoofMapNet JSON annotations with pre-defined train/valid splits

Each dataset's source splits are respected.  Processed NPZ files are written
into a shared output directory under <dataset>/train/, <dataset>/valid/, etc.
"""

from pathlib import Path
import json
import random
import shutil
from typing import List, Optional

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
from tqdm import tqdm

from roofmapnet.train.preprocess_rid2 import preprocess_roof_lines

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DATASETS = ("rid2", "roofmapnet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _convert_lines(lines_raw):
    """Convert flat [y1, x1, y2, x2] lists into endpoint pairs."""
    lines = []
    for line in lines_raw:
        if len(line) != 4:
            raise ValueError(f"Expected 4 values per line, got {len(line)}")
        y1, x1, y2, x2 = line
        lines.append([(y1, x1), (y2, x2)])
    return lines


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------


def process_rid2(
    label_dir: Path,
    image_dir: Optional[Path],
    output_dir: Path,
    train_split: float,
    valid_split: float,
    seed: int,
    skip_existing: bool,
):
    """Process RID2 edge-label NPZs.  Source has no splits so we create them."""
    click.echo(f"\n  Label dir: {label_dir}")
    if image_dir:
        click.echo(f"  Image dir: {image_dir}")

    input_files = sorted(label_dir.glob("*.npz"))
    if not input_files:
        raise click.ClickException(f"No NPZ files found in {label_dir}")
    click.echo(f"  Found {len(input_files)} label files")

    # -- preprocess each file into a flat staging area, then split ----------
    staging = output_dir / "_rid2_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for p in tqdm(input_files, desc="  rid2 preprocess"):
        out_path = staging / f"{p.stem}.npz"
        if skip_existing and out_path.exists():
            continue
        try:
            edges = np.load(p)["edges"]
            edges[:, :, [0, 1]] = edges[:, :, [1, 0]]  # flip x/y
            preprocess_roof_lines(edges, out_path)
        except Exception as e:
            click.echo(f"\n  Warning: {p.name}: {e}", err=True)

    # -- random split -------------------------------------------------------
    all_staged = sorted(staging.glob("*.npz"))
    random.seed(seed)
    random.shuffle(all_staged)

    n = len(all_staged)
    n_train = int(train_split * n)
    n_valid = int(valid_split * n)

    splits = {
        "train": all_staged[:n_train],
        "valid": all_staged[n_train : n_train + n_valid],
        "test": all_staged[n_train + n_valid :],
    }

    for split_name, files in splits.items():
        dest = output_dir / split_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(dest / f.name))

    shutil.rmtree(staging, ignore_errors=True)

    for s in ("train", "valid", "test"):
        count = len(list((output_dir / s).glob("*.npz")))
        click.echo(f"  {s}: {count} files")


def process_roofmapnet(
    label_dir: Path,
    image_dir: Optional[Path],
    output_dir: Path,
    skip_existing: bool,
):
    """Process RoofMapNet JSON annotations.  Source already has train/valid."""
    click.echo(f"\n  Label dir: {label_dir}")
    if image_dir:
        click.echo(f"  Image dir: {image_dir}")

    train_json = label_dir / "train.json"
    valid_json = label_dir / "valid.json"

    if not train_json.exists() or not valid_json.exists():
        raise click.ClickException(
            f"Expected train.json and valid.json in {label_dir}"
        )

    split_map = {"train": train_json, "valid": valid_json}

    for split_name, json_path in split_map.items():
        dest = output_dir / split_name
        dest.mkdir(parents=True, exist_ok=True)

        annotations = _load_json(json_path)
        click.echo(f"  {split_name}: {len(annotations)} samples")

        for sample in tqdm(annotations, desc=f"  roofmapnet {split_name}"):
            filename = Path(sample["filename"]).stem
            out_path = dest / f"{filename}.npz"
            if skip_existing and out_path.exists():
                continue

            lines = _convert_lines(sample.get("lines", []))
            preprocess_roof_lines(lines, out_path)

    for s in ("train", "valid"):
        count = len(list((output_dir / s).glob("*.npz")))
        click.echo(f"  {s}: {count} files")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_dataset(
    dataset_name: str,
    output_dir: Path,
    image_dir: Optional[Path],
    n_samples: int = 5,
):
    """Generate *n_samples* visualizations for a processed dataset."""
    click.echo(f"\n  Generating {n_samples} visualizations for {dataset_name}")

    # Collect sample files from all splits
    samples: list[Path] = []
    for split in ("train", "valid", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            samples.extend(sorted(split_dir.glob("*.npz")))
    if not samples:
        click.echo(f"  No processed files found for {dataset_name}", err=True)
        return

    # Pick up to n_samples evenly spaced across the dataset
    step = max(1, len(samples) // n_samples)
    chosen = samples[::step][:n_samples]

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for sample_file in chosen:
        data = np.load(sample_file)

        # Attempt to load the source image
        gt_image = None
        if image_dir is not None:
            for ext in (".png", ".jpg", ".jpeg", ".tif"):
                candidate = image_dir / f"{sample_file.stem}{ext}"
                if candidate.exists():
                    try:
                        gt_image = imread(str(candidate))
                    except Exception:
                        pass
                    break

        n_cols = 3
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        fig.suptitle(f"{dataset_name}: {sample_file.stem}", fontsize=14)

        # 1 – Source image
        ax = axes[0, 0]
        if gt_image is not None:
            ax.imshow(gt_image)
            ax.set_title("Source Image")
        else:
            ax.text(
                0.5,
                0.5,
                "Source image\nnot available",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_title("Source Image")
        ax.axis("off")

        # 2 – Junction heatmap
        ax = axes[0, 1]
        im = ax.imshow(data["jmap"][0], cmap="hot")
        ax.set_title("Junction Heatmap (jmap)")
        plt.colorbar(im, ax=ax)

        # 3 – Junction offset X
        ax = axes[0, 2]
        im = ax.imshow(data["joff"][0][0], cmap="coolwarm")
        ax.set_title("Junction Offset X")
        plt.colorbar(im, ax=ax)

        # 4 – Junction offset Y
        ax = axes[1, 0]
        im = ax.imshow(data["joff"][0][1], cmap="coolwarm")
        ax.set_title("Junction Offset Y")
        plt.colorbar(im, ax=ax)

        # 5 – Line heatmap
        ax = axes[1, 1]
        im = ax.imshow(data["lmap"], cmap="hot")
        ax.set_title("Line Heatmap (lmap)")
        plt.colorbar(im, ax=ax)

        # 6 – Stats text
        ax = axes[1, 2]
        stats = (
            f"junctions: {data['junc'].shape[0]}\n"
            f"pos lines: {data['Lpos'].shape[0]}\n"
            f"neg lines: {data['Lneg'].shape[0]}\n"
            f"jmap range: [{data['jmap'].min():.2f}, {data['jmap'].max():.2f}]\n"
            f"lmap range: [{data['lmap'].min():.2f}, {data['lmap'].max():.2f}]"
        )
        ax.text(0.1, 0.5, stats, fontsize=12, va="center", family="monospace")
        ax.set_title("Stats")
        ax.axis("off")

        plt.tight_layout()
        viz_path = viz_dir / f"viz_{dataset_name}_{sample_file.stem}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    click.echo(f"  ✓ Saved {len(chosen)} images to {viz_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--datasets",
    type=click.Choice(VALID_DATASETS, case_sensitive=False),
    multiple=True,
    required=True,
    help="Which datasets to include (can be specified multiple times)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="processed",
    help="Root output directory for all processed data",
)
# -- RID2 paths --
@click.option(
    "--rid2-label-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="/Users/swatts/datasets/roof_information_dataset_2/preprocessing_output/edge_labels",
    help="RID2: directory containing edge-label NPZ files",
)
@click.option(
    "--rid2-image-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="RID2: directory containing source images (for visualization)",
)
# -- RoofMapNet paths --
@click.option(
    "--roofmapnet-label-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data",
    help="RoofMapNet: directory containing train.json / valid.json",
)
@click.option(
    "--roofmapnet-image-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data/images",
    help="RoofMapNet: directory containing source images (for visualization)",
)
# -- Split config (RID2 only – RoofMapNet splits come from JSON) --
@click.option("--train-split", type=float, default=0.7, help="RID2 train fraction")
@click.option("--valid-split", type=float, default=0.15, help="RID2 valid fraction")
@click.option("--test-split", type=float, default=0.15, help="RID2 test fraction")
@click.option("--seed", type=int, default=42, help="Random seed for RID2 split")
# -- Flags --
@click.option(
    "--visualize", is_flag=True, help="Generate 5 visualizations per dataset"
)
@click.option("--skip-existing", is_flag=True, help="Skip files that already exist")
def prepare_data(
    datasets: List[str],
    output_dir: Path,
    rid2_label_dir: Path,
    rid2_image_dir: Optional[Path],
    roofmapnet_label_dir: Path,
    roofmapnet_image_dir: Optional[Path],
    train_split: float,
    valid_split: float,
    test_split: float,
    seed: int,
    visualize: bool,
    skip_existing: bool,
):
    """Prepare training data from one or more source datasets.

    Source splits are respected: RoofMapNet uses train.json / valid.json;
    RID2 is randomly split using --train-split / --valid-split / --test-split.

    \b
    Examples
    --------
        python -m roofmapnet.train.prepare_data --datasets rid2
        python -m roofmapnet.train.prepare_data --datasets rid2 --datasets roofmapnet --visualize
    """
    datasets_lower = [d.lower() for d in datasets]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"{'=' * 60}")
    click.echo(f"Preparing datasets: {', '.join(datasets_lower)}")
    click.echo(f"Output directory:   {output_dir}")
    click.echo(f"{'=' * 60}")

    # ── RID2 ──────────────────────────────────────────────────────────────
    if "rid2" in datasets_lower:
        total = train_split + valid_split + test_split
        if not (0.99 <= total <= 1.01):
            raise click.BadParameter(
                f"RID2 splits must sum to 1.0 (got {total})"
            )

        click.echo(f"\n{'─' * 60}")
        click.echo("Dataset: rid2")
        click.echo(f"{'─' * 60}")

        rid2_out = output_dir / "rid2"
        rid2_out.mkdir(parents=True, exist_ok=True)

        process_rid2(
            label_dir=rid2_label_dir,
            image_dir=rid2_image_dir,
            output_dir=rid2_out,
            train_split=train_split,
            valid_split=valid_split,
            seed=seed,
            skip_existing=skip_existing,
        )

        if visualize:
            visualize_dataset("rid2", rid2_out, rid2_image_dir, n_samples=5)

    # ── RoofMapNet ────────────────────────────────────────────────────────
    if "roofmapnet" in datasets_lower:
        click.echo(f"\n{'─' * 60}")
        click.echo("Dataset: roofmapnet")
        click.echo(f"{'─' * 60}")

        rmn_out = output_dir / "roofmapnet"
        rmn_out.mkdir(parents=True, exist_ok=True)

        process_roofmapnet(
            label_dir=roofmapnet_label_dir,
            image_dir=roofmapnet_image_dir,
            output_dir=rmn_out,
            skip_existing=skip_existing,
        )

        if visualize:
            visualize_dataset(
                "roofmapnet", rmn_out, roofmapnet_image_dir, n_samples=5
            )

    # ── Summary ───────────────────────────────────────────────────────────
    click.echo(f"\n{'=' * 60}")
    click.echo("Summary")
    click.echo(f"{'=' * 60}")

    for ds in datasets_lower:
        ds_dir = output_dir / ds
        for split in ("train", "valid", "test"):
            split_dir = ds_dir / split
            if split_dir.exists():
                n = len(list(split_dir.glob("*.npz")))
                click.echo(f"  {ds}/{split}: {n} files")

    click.echo("\n✓ All done!")


if __name__ == "__main__":
    prepare_data()
