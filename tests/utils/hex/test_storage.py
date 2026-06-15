"""Tests for hex storage operations (align, permute) with human inspection
figures.

These tests demonstrate:
1. Data alignment between mismatched hex grids (use case: multi-resolution)
2. Rotation permutation for data augmentation (use case: training augmentation)
3. Reflection symmetry preservation (use case: bilateral symmetry)
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from btorch.utils.file import save_fig
from btorch.utils.hex import align, disk, permute, reflect_index, spiral
from btorch.utils.hex.coords import disk_count
from btorch.utils.hex.storage import axial_to_rect_index, rect_index_to_axial


def test_permute_identity():
    """6 rotations should be identity permutation."""
    for rad in [2, 3, 4]:
        perm = permute(rad, 6)
        assert np.array_equal(perm, np.arange(disk_count(rad)))


@pytest.mark.parametrize("axis", ["q", "r", "s"])
def test_reflect_index_self_inverse(axis):
    """Reflecting twice should return to original."""
    values = np.arange(disk_count(4), dtype=float)
    perm = reflect_index(4, axis)
    assert np.allclose(values, values[perm][perm])


def test_align_same_grid():
    """Aligning a grid to itself should preserve values."""
    q, r = disk(3)
    values = np.arange(len(q), dtype=float)
    aligned = align(q, r, q, r, values, fill=-1)
    assert np.allclose(aligned, values)


def test_align_fill_value():
    """Aligning a small grid onto a large one should fill missing with fill."""
    q_src, r_src = disk(1)
    values_src = np.array([10.0])
    q_tgt, r_tgt = disk(3)
    aligned = align(q_tgt, r_tgt, q_src, r_src, values_src, fill=-999)
    assert np.any(aligned == -999), "fill value should appear for missing hexes"


@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_rect_index_roundtrip(orientation):
    """Axial → rect_index → axial must roundtrip."""
    q, r = disk(3)
    row, col = axial_to_rect_index(q, r, orientation=orientation)
    q_b, r_b = rect_index_to_axial(row, col, orientation=orientation)
    assert np.array_equal(q, q_b) and np.array_equal(r, r_b)


def test_align_mismatched_grids():
    """Align data from a larger grid to a smaller grid.

    Example use case: Connectome data at different resolutions,
    or mapping between two eyes with different extents.
    """
    # Create source grid (larger)
    q_src, r_src = disk(5)
    # Source data: a Gaussian blob off-center
    values_src = np.exp(-((q_src - 2) ** 2 + (r_src - 1) ** 2) / 4)

    # Create target grid (smaller)
    q_tgt, r_tgt = disk(3)

    # Align source to target
    values_aligned = align(q_tgt, r_tgt, q_src, r_src, values_src, fill=np.nan)

    # Visualize
    from btorch.utils.hex import to_pixel

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x_src, y_src = to_pixel(q_src, r_src)
    im1 = axes[0].scatter(x_src, y_src, c=values_src, cmap="viridis", s=100)
    axes[0].set_title(f"Source Grid (r=5, n={len(q_src)})")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0])

    x_tgt, y_tgt = to_pixel(q_tgt, r_tgt)
    im2 = axes[1].scatter(x_tgt, y_tgt, c=values_aligned, cmap="viridis", s=100)
    axes[1].set_title(f"Aligned to Target (r=3, n={len(q_tgt)})")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1])

    # Show missing values as black circles
    missing = np.isnan(values_aligned)
    present = ~missing
    if present.any():
        axes[2].scatter(
            x_tgt[present],
            y_tgt[present],
            c=values_aligned[present],
            cmap="viridis",
            s=100,
            vmin=0,
            vmax=values_src.max(),
        )
    if missing.any():
        axes[2].scatter(
            x_tgt[missing],
            y_tgt[missing],
            facecolors="none",
            edgecolors="red",
            s=100,
            linewidths=2,
            label="Missing",
        )
    axes[2].set_title("Missing Values (red outline = filled)")
    axes[2].set_aspect("equal")
    axes[2].axis("off")

    plt.suptitle("Data Alignment Between Mismatched Grids", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "align_mismatched_grids")
    plt.close()

    # Verify alignment correctness
    # Overlapping region should have same values
    for i, (qt, rt) in enumerate(zip(q_tgt, r_tgt)):
        if not np.isnan(values_aligned[i]):
            # Find in source
            match = (q_src == qt) & (r_src == rt)
            if match.any():
                src_idx = np.where(match)[0][0]
                assert np.isclose(
                    values_aligned[i], values_src[src_idx]
                ), f"Alignment mismatch at ({qt}, {rt})"


def test_align_multidim_features():
    """Align feature vectors (channels, n_hexes) between mismatched grids.

    Typical use case: aligning neuron-type feature maps from one grid
    resolution to another.
    """
    q_src, r_src = disk(5)
    q_tgt, r_tgt = disk(3)
    n_channels = 4

    # Per-channel constant values to detect cross-channel bleeding
    values = np.arange(n_channels)[:, None] * np.ones((n_channels, len(q_src)))
    values = values.astype(float)

    result = align(q_tgt, r_tgt, q_src, r_src, values, fill=np.nan)
    assert result.shape == (n_channels, len(q_tgt))

    # Where target matches source, each channel retains its constant
    for i, (qt, rt) in enumerate(zip(q_tgt, r_tgt)):
        match = (q_src == qt) & (r_src == rt)
        if match.any():
            src_idx = np.where(match)[0][0]
            np.testing.assert_array_equal(result[:, i], values[:, src_idx])
        else:
            assert np.all(np.isnan(result[:, i]))


def test_align_multidim_batch():
    """Align a batch of spatial maps (batch, time, n_hexes).

    Verifies that leading dimensions are treated independently.
    """
    q_src, r_src = disk(4)
    q_tgt, r_tgt = disk(2)

    rng = np.random.RandomState(42)
    values = rng.randn(3, 5, len(q_src))

    result = align(q_tgt, r_tgt, q_src, r_src, values, fill=0.0)
    assert result.shape == (3, 5, len(q_tgt))

    # Each (b, t) slice should independently align correctly
    for b in range(3):
        for t in range(5):
            slice_result = align(q_tgt, r_tgt, q_src, r_src, values[b, t], fill=0.0)
            np.testing.assert_array_equal(result[b, t], slice_result)


def test_permute_rotation_invariance():
    """Visualize how permutation rotates data on hex grid.

    Example use case: Data augmentation by rotating inputs while
    preserving the spatial structure.
    """
    radius = 3
    q, r = spiral(radius)

    # Create a pattern: horizontal gradient
    values = q.astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    from btorch.utils.hex import to_pixel

    for n_rot in range(6):
        perm = permute(radius, n_rot)
        values_rot = values[perm]

        x, y = to_pixel(q, r)

        ax = axes[n_rot]
        ax.scatter(x, y, c=values_rot, cmap="RdBu_r", s=150, vmin=-3, vmax=3)
        ax.set_title(f"Rotation {n_rot}×60°")
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Data Rotation via Permutation (Horizontal Gradient)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "permute_rotation_invariance")
    plt.close()

    # Verify 6 rotations returns to original
    perm_6 = permute(radius, 6)
    assert np.array_equal(perm_6, np.arange(len(q))), "6 rotations should be identity"


def test_reflect_index_symmetry():
    """Visualize reflection permutations - bilateral symmetry.

    Example use case: Enforcing bilateral symmetry in networks,
    or data augmentation via reflection.
    """
    radius = 4
    q, r = spiral(radius)

    # Create asymmetric pattern
    values = np.sin(q * np.pi / 2) + np.cos(r * np.pi / 3)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    from btorch.utils.hex import to_pixel

    # Original
    x, y = to_pixel(q, r)
    axes[0, 0].scatter(x, y, c=values, cmap="RdBu_r", s=100)
    axes[0, 0].set_title("Original")
    axes[0, 0].set_aspect("equal")
    axes[0, 0].axis("off")

    # Reflect across q axis
    perm_q = reflect_index(radius, "q")
    axes[0, 1].scatter(x, y, c=values[perm_q], cmap="RdBu_r", s=100)
    axes[0, 1].set_title("Reflect across q axis")
    axes[0, 1].set_aspect("equal")
    axes[0, 1].axis("off")

    # Reflect across r axis
    perm_r = reflect_index(radius, "r")
    axes[1, 0].scatter(x, y, c=values[perm_r], cmap="RdBu_r", s=100)
    axes[1, 0].set_title("Reflect across r axis")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].axis("off")

    # Reflect across s axis
    perm_s = reflect_index(radius, "s")
    axes[1, 1].scatter(x, y, c=values[perm_s], cmap="RdBu_r", s=100)
    axes[1, 1].set_title("Reflect across s axis")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].axis("off")

    plt.suptitle("Reflection Permutations (Bilateral Symmetry)", fontsize=14)
    plt.tight_layout()
    save_fig(fig, "reflect_index_symmetry")
    plt.close()

    # Verify reflection is its own inverse
    perm_q_inv = reflect_index(radius, "q")
    values_double = values[perm_q][perm_q_inv]
    assert np.allclose(values, values_double), "Reflection should be self-inverse"


def test_spiral_ordering():
    """Visualize spiral ordering used for permutation operations.

    Shows the ordering: center first, then ring by ring.
    This is essential for rotation invariance.
    """
    radius = 4
    q, r = spiral(radius)

    from btorch.utils.hex import to_pixel

    x, y = to_pixel(q, r)

    indices = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x, y, c=indices, cmap="tab20", s=200, edgecolors="black")

    # Annotate ring boundaries
    for ring_r in range(radius + 1):
        _ = 1 if ring_r == 0 else 6 * ring_r
        if ring_r == 0:
            idx = 0
        else:
            idx = disk_count(ring_r - 1)
        if idx < len(q):
            ax.annotate(
                f"ring {ring_r}",
                (x[idx], y[idx]),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_title(f"Spiral Ordering (radius={radius}, n={len(q)})")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(scatter, ax=ax, label="Spiral index")

    save_fig(fig, "spiral_ordering")
    plt.close()
