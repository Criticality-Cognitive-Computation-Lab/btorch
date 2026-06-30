"""Tests for hex animate module (HexScatter, HexQuiver).

Saves plots/animation under fig_path() for manual inspection. Intended-
usage scenarios and edge cases only.
"""

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from btorch.utils.file import fig_path
from btorch.utils.hex.coords import disk
from btorch.utils.hex.offset import axial_to_zigzag
from btorch.utils.hex.transform import to_pixel
from btorch.visualisation.hex.animate import HexQuiver, HexScatter


OUT = fig_path()


# -- helpers ---------------------------------------------------------------


def _disk_data(n_frames: int = 5, radius: int = 3):
    """Return (values, q, r) for a hex disk with smooth temporal gradient."""
    q, r = disk(radius)
    n_hex = len(q)
    t = np.linspace(0, 2 * np.pi, n_frames)
    phase = np.arange(n_hex) / n_hex
    values = np.stack(
        [np.sin(t_i + 2 * np.pi * phase) for t_i in t]
    )  # (n_frames, n_hex)
    return values.astype(float), q.astype(float), r.astype(float)


def _flow_data(n_frames: int = 4, radius: int = 2):
    """Return (flow, q, r) with rotating unit vectors."""
    q, r = disk(radius)
    n_hex = len(q)
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    flow = np.zeros((n_frames, 2, n_hex))
    for i, a in enumerate(angles):
        flow[i, 0] = np.cos(a)
        flow[i, 1] = np.sin(a)
    return flow, q.astype(float), r.astype(float)


# -- HexScatter ------------------------------------------------------------


class TestHexScatter:
    def test_axial_coords_gif(self):
        """Typical use: animate neural activity on axial hex grid."""
        values, q, r = _disk_data(n_frames=8, radius=3)
        anim = HexScatter(values, q, r, coord_format="axial", interval=100)
        path = OUT / "scatter_axial.gif"
        anim.save(str(path), writer="pillow", fps=5)
        assert path.exists() and path.stat().st_size > 0

    def test_zigzag_coords(self):
        """Zigzag coords commonly come from connectome adjacency data."""
        values, q, r = _disk_data(n_frames=4, radius=2)
        x, y = axial_to_zigzag(q.astype(int), r.astype(int))
        anim = HexScatter(
            values, x.astype(float), y.astype(float), coord_format="zigzag"
        )
        path = OUT / "scatter_zigzag.gif"
        anim.save(str(path), writer="pillow", fps=5)
        assert path.exists()

    def test_pixel_coords(self):
        """Pixel coords from to_pixel — no conversion inside animate."""
        values, q, r = _disk_data(n_frames=4, radius=2)
        px, py = to_pixel(q, r)
        anim = HexScatter(values, px, py, coord_format="pixel")
        path = OUT / "scatter_pixel.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_nan_values(self):
        """NaN hexes (e.g. masked neurons) should render without crash."""
        values, q, r = _disk_data(n_frames=3, radius=2)
        values[:, :2] = np.nan
        anim = HexScatter(values, q, r, coord_format="axial")
        path = OUT / "scatter_nan.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_single_frame(self):
        """Edge case: only one frame (static image via animation)."""
        values, q, r = _disk_data(n_frames=1, radius=2)
        anim = HexScatter(values, q, r, coord_format="axial")
        path = OUT / "scatter_single.gif"
        anim.save(str(path), writer="pillow", fps=1)
        assert path.exists()

    def test_custom_vmin_vmax(self):
        """Fixed color range — useful for comparing across time."""
        values, q, r = _disk_data(n_frames=4, radius=2)
        anim = HexScatter(
            values, q, r, coord_format="axial", vmin=-2.0, vmax=2.0, cmap="coolwarm"
        )
        assert anim.vmin == -2.0
        assert anim.vmax == 2.0
        path = OUT / "scatter_custom_range.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_external_axes(self):
        """Reusing an existing axes — common in multi-panel figures."""
        values, q, r = _disk_data(n_frames=3, radius=2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        HexScatter(values, q, r, coord_format="axial", ax=ax1)
        ax2.set_title("placeholder")
        path = OUT / "scatter_external_ax.png"
        fig.savefig(str(path))
        plt.close(fig)
        assert path.exists()

    def test_invalid_coord_format(self):
        values, q, r = _disk_data(n_frames=2, radius=1)
        with pytest.raises(ValueError, match="Unknown coord_format"):
            HexScatter(values, q, r, coord_format="bad")

    def test_update_changes_array(self):
        """Verify update actually swaps the scatter data."""
        values, q, r = _disk_data(n_frames=3, radius=1)
        anim = HexScatter(values, q, r, coord_format="axial")
        anim.update(2)
        np.testing.assert_allclose(anim.sc.get_array(), values[2], atol=1e-10)


# -- HexQuiver ---------------------------------------------------------------


class TestHexQuiver:
    def test_axial_coords_gif(self):
        """Typical use: animate optic flow vectors on hex grid."""
        flow, q, r = _flow_data(n_frames=6, radius=2)
        anim = HexQuiver(flow, q, r, coord_format="axial", interval=150)
        path = OUT / "flow_axial.gif"
        anim.save(str(path), writer="pillow", fps=5)
        assert path.exists() and path.stat().st_size > 0

    def test_zigzag_coords(self):
        flow, q, r = _flow_data(n_frames=4, radius=2)
        x, y = axial_to_zigzag(q.astype(int), r.astype(int))
        anim = HexQuiver(flow, x.astype(float), y.astype(float), coord_format="zigzag")
        path = OUT / "flow_zigzag.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_pixel_coords(self):
        flow, q, r = _flow_data(n_frames=3, radius=2)
        px, py = to_pixel(q, r)
        anim = HexQuiver(flow, px, py, coord_format="pixel")
        path = OUT / "flow_pixel.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_scale_amplifies_vectors(self):
        """Scale factor should make small vectors visible."""
        flow, q, r = _flow_data(n_frames=2, radius=2)
        flow *= 0.01  # tiny flow
        anim = HexQuiver(flow, q, r, coord_format="axial", scale=50.0)
        assert anim.scale == 50.0
        path = OUT / "flow_scaled.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_colorwheel_disabled(self):
        """Cwheel=False hides the direction legend."""
        flow, q, r = _flow_data(n_frames=2, radius=1)
        anim = HexQuiver(flow, q, r, coord_format="axial", cwheel=False)
        assert anim.cwheel is False
        path = OUT / "flow_no_cwheel.gif"
        anim.save(str(path), writer="pillow", fps=2)
        assert path.exists()

    def test_single_frame(self):
        flow, q, r = _flow_data(n_frames=1, radius=1)
        anim = HexQuiver(flow, q, r, coord_format="axial")
        path = OUT / "flow_single.gif"
        anim.save(str(path), writer="pillow", fps=1)
        assert path.exists()

    def test_external_axes(self):
        flow, q, r = _flow_data(n_frames=2, radius=2)
        fig, ax = plt.subplots(figsize=(5, 5))
        HexQuiver(flow, q, r, coord_format="axial", ax=ax)
        path = OUT / "flow_external_ax.png"
        fig.savefig(str(path))
        plt.close(fig)
        assert path.exists()

    def test_invalid_coord_format(self):
        flow, q, r = _flow_data(n_frames=2, radius=1)
        with pytest.raises(ValueError, match="Unknown coord_format"):
            HexQuiver(flow, q, r, coord_format="bad")

    def test_update_changes_quiver(self):
        """Verify update swaps the quiver UV data."""
        flow, q, r = _flow_data(n_frames=3, radius=1)
        anim = HexQuiver(flow, q, r, coord_format="axial")
        anim.update(2)
        expected_u = flow[2, 0] * anim.scale
        expected_v = flow[2, 1] * anim.scale
        np.testing.assert_allclose(anim.quiver.U, expected_u, atol=1e-10)
        np.testing.assert_allclose(anim.quiver.V, expected_v, atol=1e-10)
