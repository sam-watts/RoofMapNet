"""Prepare RoofMapNet dataset from repo /data annotations.

Reads data/train.json and data/valid.json (list of line segments per image)
and converts each entry to RoofMapNet NPZ format using preprocess_rid2.
"""

from pathlib import Path
import json

import click
import numpy as np
from tqdm import tqdm

from roofmapnet.train.preprocess_rid2 import preprocess_roof_lines


def _load_annotations(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


def _convert_lines(lines_raw):
    """Convert flat [y1,x1,y2,x2] lists into endpoint pairs."""
    lines = []
    for line in lines_raw:
        if len(line) != 4:
            raise ValueError(f"Expected 4 values per line, got {len(line)}")
        y1, x1, y2, x2 = line
        lines.append([(y1, x1), (y2, x2)])
    return lines


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="data",
    help="Path to repo data directory containing train.json/valid.json and images/",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="data/processed",
    help="Output directory for processed NPZ files",
)
@click.option(
    "--image-size",
    type=int,
    default=512,
    help="Input image size used when generating heatmaps",
)
@click.option(
    "--heatmap-size",
    type=int,
    default=128,
    help="Heatmap size used for junction/line maps",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    help="Skip entries whose NPZ output already exists",
)
def prepare_roofmapnet_data(
    data_dir: Path,
    output_dir: Path,
    image_size: int,
    heatmap_size: int,
    skip_existing: bool,
):
    """Preprocess RoofMapNet dataset located in the repo /data directory."""
    train_json = data_dir / "train.json"
    valid_json = data_dir / "valid.json"

    if not train_json.exists() or not valid_json.exists():
        raise click.ClickException(
            f"Expected train.json and valid.json in {data_dir}"
        )

    output_dir = output_dir.resolve()
    train_out = output_dir / "train"
    valid_out = output_dir / "valid"
    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    click.echo(f"Output directory: {output_dir}")

    def process_split(split_name: str, json_path: Path, split_out: Path):
        annotations = _load_annotations(json_path)
        click.echo(f"{split_name}: {len(annotations)} samples")

        for sample in tqdm(annotations, desc=f"Processing {split_name}"):
            filename = Path(sample["filename"]).stem
            out_path = split_out / f"{filename}.npz"
            if skip_existing and out_path.exists():
                continue

            lines_raw = sample.get("lines", [])
            lines = _convert_lines(lines_raw)
            preprocess_roof_lines(
                lines,
                out_path,
                image_size=image_size,
                heatmap_size=heatmap_size,
            )

    process_split("train", train_json, train_out)
    process_split("valid", valid_json, valid_out)

    click.echo("✓ Done")


if __name__ == "__main__":
    prepare_roofmapnet_data()
