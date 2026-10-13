"""Filtrations on a 3D point cloud ``(row, column, intensity)``.

Instead of a random subsample of above-threshold pixels (the 2D point-cloud
pipeline), a *regular* grid of nodes is laid over the image (one node every
``step`` pixels), background nodes are discarded, and each remaining node
becomes a 3D point whose third coordinate is its intensity::

    point = (row, column, intensity)   for intensity >= threshold

Proximity therefore depends both on position and on similarity of signal.
The three axes are min-max normalised to [0, 1] before computing distances;
otherwise pixel coordinates (0-218) would dwarf intensities (0-1) and the
filtration would collapse back to 2D. Radii are expressed in these
normalised units.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config, preprocessing, topology

DEFAULT_STEP = 4
DEFAULT_MAX_RADIUS = 0.3
SUPPORTED = ("rips", "alpha", "delaunaycech")


def image_to_3d_point_cloud(img, step: int = DEFAULT_STEP, threshold: float | None = None, quantile: float | None = None) -> np.ndarray:
    """``(N, 3)`` cloud of grid nodes above the threshold.

    ``threshold=None`` uses ``config.POINT_CLOUD_INTENSITY_THRESHOLD``; a
    ``quantile`` overrides it with a per-image adaptive threshold.
    """
    img = np.asarray(img, dtype=np.float64)
    if quantile is not None:
        threshold = preprocessing.quantile_threshold_value(img, quantile)
    elif threshold is None:
        threshold = config.POINT_CLOUD_INTENSITY_THRESHOLD

    rr, cc = np.meshgrid(np.arange(0, img.shape[0], step), np.arange(0, img.shape[1], step), indexing="ij")
    rows, cols = rr.ravel(), cc.ravel()
    intensity = img[rows, cols]
    keep = intensity >= threshold
    return np.column_stack([rows[keep], cols[keep], intensity[keep]]).astype(np.float64)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Min-max normalise each axis to [0, 1] (constant axes map to 0)."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return points
    mins = points.min(axis=0)
    ranges = points.max(axis=0) - mins
    ranges[ranges == 0] = 1.0
    return (points - mins) / ranges


def compute_filtration_3d(points_3d, filtration_type="alpha", max_radius=DEFAULT_MAX_RADIUS, max_dimension=2) -> dict:
    """H0/H1 persistence diagrams of a normalised 3D cloud."""
    ftype = topology.resolve_filtration_name(filtration_type)
    if ftype not in SUPPORTED:
        raise ValueError(f"3D filtration must be one of {SUPPORTED}.")

    points_3d = np.asarray(points_3d, dtype=np.float64)
    if points_3d.shape[0] == 0:
        return {"filtration_type": ftype, "diag_h0": topology.empty_diagram(), "diag_h1": topology.empty_diagram()}

    st = topology.simplex_tree_for_points_at_radius(points_3d, ftype, radius=max_radius, max_dimension=max_dimension)
    st.compute_persistence()
    persistence = st.persistence()
    return {
        "filtration_type": ftype,
        "diag_h0": topology.extract_diagram(persistence, 0, min_persistence=0.0, cap_infinite_at=max_radius),
        "diag_h1": topology.extract_diagram(persistence, 1, min_persistence=0.0, cap_infinite_at=max_radius),
    }


def compute_tda_one_image_3d(img, filtration_type="alpha", step=DEFAULT_STEP, threshold=None, quantile=None, max_radius=DEFAULT_MAX_RADIUS) -> dict:
    """3D cloud -> filtration -> vectorisations, in the common TDA format
    (plus ``points``). Vectorisations are sampled on ``[0, max_radius]``."""
    points = normalize_point_cloud(image_to_3d_point_cloud(img, step=step, threshold=threshold, quantile=quantile))
    diags = compute_filtration_3d(points, filtration_type=filtration_type, max_radius=max_radius)
    return {**topology.vectorise(diags["diag_h0"], diags["diag_h1"], (0.0, max_radius)), "points": points}


def compute_tda_for_images_3d(images, filtration_type="alpha", step=DEFAULT_STEP, threshold=None, quantile=None, max_radius=DEFAULT_MAX_RADIUS) -> dict:
    """Batch version of :func:`compute_tda_one_image_3d`, same output format
    as :func:`topology.compute_tda_for_images`."""
    out = {key: [] for key in topology.TDA_KEYS}
    out.update({"grid": [], "filtration_type": filtration_type})
    for img in images:
        tda = compute_tda_one_image_3d(img, filtration_type, step=step, threshold=threshold, quantile=quantile, max_radius=max_radius)
        out["grid"].append(tda["points"])
        for key in topology.TDA_KEYS:
            out[key].append(tda[key])
    return out


def compute_template_tda_3d(templates: dict, filtration_type="alpha", step=DEFAULT_STEP, threshold=None, quantile=None, max_radius=DEFAULT_MAX_RADIUS) -> dict:
    """3D TDA of the class-average display templates (``<cls>_display``)."""
    return {
        cls: compute_tda_one_image_3d(templates[f"{cls}_display"], filtration_type, step=step, threshold=threshold, quantile=quantile, max_radius=max_radius)
        for cls in ("AD", "CN")
    }


def plot_point_cloud_and_diagram(points, diags, title="", max_radius=DEFAULT_MAX_RADIUS):
    """3D cloud coloured by intensity next to its H0/H1 persistence diagram."""
    fig = plt.figure(figsize=(11, 4.8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax1.scatter(points[:, 1], points[:, 0], points[:, 2], c=points[:, 2], cmap="viridis", s=6)
    ax1.set_xlabel("column")
    ax1.set_ylabel("row")
    ax1.set_zlabel("intensity")
    ax1.set_title("3D point cloud (normalised)")
    fig.colorbar(sc, ax=ax1, shrink=0.6, label="intensity")

    ax2 = fig.add_subplot(1, 2, 2)
    for key, marker, lab in (("diag_h0", "o", "H0"), ("diag_h1", "^", "H1")):
        d = diags[key]
        if d.size:
            ax2.scatter(d[:, 0], d[:, 1], marker=marker, label=lab, alpha=0.7)
    ax2.plot([0, max_radius], [0, max_radius], "k--", linewidth=0.8)
    ax2.set_xlim(0, max_radius * 1.05)
    ax2.set_ylim(0, max_radius * 1.05)
    ax2.set_xlabel("birth")
    ax2.set_ylabel("death")
    ax2.set_title(f"Persistence diagram ({diags['filtration_type']})")
    ax2.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
