"""Unit tests for preprocess_rid2 clipping behavior."""

from pathlib import Path
import tempfile

import numpy as np

from roofmapnet.train.preprocess_rid2 import preprocess_roof_lines


def test_line_clipping_creates_edge_junctions():
    """Ensure lines intersecting image edges create junctions within bounds."""
    lines = [[(-10.0, 0.0), (600.0, 600.0)]]  # crosses image bounds
    image_size = 512
    heatmap_size = 128

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sample.npz"
        preprocess_roof_lines(
            lines,
            out_path,
            image_size=image_size,
            heatmap_size=heatmap_size,
        )

        data = np.load(out_path)
        junc = data["junc"]
        lpos = data["lpos"]
        Lpos = data["Lpos"]

        # Expect at least two junctions from the clipped endpoints
        assert junc.shape[0] >= 2
        assert Lpos.shape[0] == 1
        assert lpos.shape[0] == 1

    # Junction coordinates should be within heatmap bounds
    assert np.all(junc[:, 0] >= 0) and np.all(junc[:, 0] <= heatmap_size + 1e-6)
    assert np.all(junc[:, 1] >= 0) and np.all(junc[:, 1] <= heatmap_size + 1e-6)


if __name__ == "__main__":
    test_line_clipping_creates_edge_junctions()
    print("All preprocess_rid2 tests passed.")
