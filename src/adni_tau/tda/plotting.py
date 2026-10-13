"""Plots for the TDA pipeline. Every function returns the matplotlib figure.

Conventions: red = AD, blue = CN; dashed = H0, solid = H1.
"""

import matplotlib.pyplot as plt
import numpy as np

from . import config

CLASS_COLORS = {0: "tab:red", 1: "tab:blue"}
CLASS_NAMES = {0: "AD", 1: "CN"}

# Per-descriptor settings: key prefix in the TDA dict, curve symbol, y label.
CURVES = {
    "betti": ("betti", "β", "Number of alive features"),
    "lifetime": ("lifetime", "L", "Accumulated persistence"),
}


def _grid(sample_range, n_points):
    sample_range = config.LANDSCAPE_RANGE if sample_range is None else sample_range
    return np.linspace(sample_range[0], sample_range[1], n_points or config.N_LANDSCAPE_POINTS)


def _default_subjects(y, n_per_class):
    y = np.asarray(y)
    return list(np.where(y == 0)[0][:n_per_class]) + list(np.where(y == 1)[0][:n_per_class])


# ─────────────────────────────────────────────────────────────────────────────
# Single subject
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagram_and_barcode(diag_h0, diag_h1, title=""):
    """Persistence diagram (H0 and H1) and barcode of one subject."""
    fig, (ax_d, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
    allv = [d for d in (diag_h0, diag_h1) if d.size]
    for d, lab, marker in ((diag_h0, "H0", "o"), (diag_h1, "H1", "^")):
        if d.size:
            ax_d.scatter(d[:, 0], d[:, 1], s=14, alpha=0.7, marker=marker, label=lab)
    if allv:
        lo, hi = float(np.vstack(allv).min()), float(np.vstack(allv).max())
        ax_d.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax_d.set(xlabel="Birth", ylabel="Death", title="Persistence diagram")
    ax_d.legend()

    row = 0
    for d, style, lab in ((diag_h0, "--", "H0"), (diag_h1, "-", "H1")):
        order = np.argsort(d[:, 0]) if d.size else []
        for j, (b, de) in enumerate(d[order] if d.size else []):
            ax_b.plot([b, de], [row, row], linestyle=style, color="tab:blue" if lab == "H0" else "tab:orange", linewidth=1, label=lab if j == 0 else None)
            row += 1
    ax_b.set(xlabel="Filtration value", ylabel="Feature", title="Barcode")
    ax_b.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def _landscape_layer(land, layer):
    n = config.N_LANDSCAPE_POINTS
    return np.asarray(land).reshape(-1)[layer * n : (layer + 1) * n]


def plot_landscape_layers(land_h0, land_h1, layers=(0, 1, 2), sample_range=None, title=""):
    """Landscape layers of one subject, one panel per layer (independent
    y-scales, because layer 0 is usually dominated by one long feature)."""
    x = _grid(sample_range, config.N_LANDSCAPE_POINTS)
    fig, axes = plt.subplots(1, len(layers), figsize=(4.5 * len(layers), 3.8), squeeze=False)
    for ax, ly in zip(axes[0], layers):
        ax.plot(x, _landscape_layer(land_h0, ly), "k--", label="H0")
        ax.plot(x, _landscape_layer(land_h1, ly), "k-", label="H1")
        ax.set(title=f"λ{ly + 1}", xlabel="Filtration value")
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# AD vs CN comparisons
# ─────────────────────────────────────────────────────────────────────────────

def plot_landscape_comparison(tda_split, y, subject_indices=None, n_per_class=2, layers=(0, 1, 4), sample_range=None, title=""):
    """Landscape layers of a few AD and CN subjects overlaid."""
    subjects = subject_indices if subject_indices is not None else _default_subjects(y, n_per_class)
    x = _grid(sample_range, config.N_LANDSCAPE_POINTS)
    fig, axes = plt.subplots(1, len(layers), figsize=(5.5 * len(layers), 4), squeeze=False)
    for ax, ly in zip(axes[0], layers):
        for i in subjects:
            c = int(y[i])
            ax.plot(x, _landscape_layer(tda_split["land_h0"][i], ly), "--", color=CLASS_COLORS[c], alpha=0.8)
            ax.plot(x, _landscape_layer(tda_split["land_h1"][i], ly), "-", color=CLASS_COLORS[c], alpha=0.8, label=f"{CLASS_NAMES[c]} #{i}")
        ax.set(title=f"λ{ly + 1}", xlabel="Filtration value")
        ax.legend(fontsize=7)
    fig.suptitle(title or "Persistence landscapes (red = AD, blue = CN; dashed = H0, solid = H1)")
    fig.tight_layout()
    return fig


def plot_curve_bands(tda_split, y, kind="betti", band="iqr", sample_range=None, title="", ax=None):
    """Median (or mean) curve per class with an IQR (or ±1 std) band."""
    prefix, symbol, ylabel = CURVES[kind]
    y = np.asarray(y)
    x = _grid(sample_range, len(np.ravel(tda_split[f"{prefix}_h0"][0])))
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    for cls in (0, 1):
        for dim, style in (("h0", "--"), ("h1", "-")):
            curves = np.vstack([np.ravel(c) for c in tda_split[f"{prefix}_{dim}"]])[y == cls]
            if band == "std":
                center, spread = curves.mean(axis=0), curves.std(axis=0)
                lo, hi = center - spread, center + spread
            else:
                center = np.median(curves, axis=0)
                lo, hi = np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
            ax.plot(x, center, style, color=CLASS_COLORS[cls], label=f"{CLASS_NAMES[cls]} {symbol}{dim[1]}")
            ax.fill_between(x, lo, hi, color=CLASS_COLORS[cls], alpha=0.12)

    ax.set(xlabel="Filtration value", ylabel=ylabel)
    ax.set_title(title or f"{kind.capitalize()} curves: median and IQR per class")
    ax.legend(fontsize=8)
    if own:
        fig.tight_layout()
    return fig


def plot_curve_bands_by_view(tda_by_view, y, views, kind="betti", split="train", sample_range=None, title=""):
    """:func:`plot_curve_bands` for every view, side by side."""
    fig, axes = plt.subplots(1, len(views), figsize=(5.5 * len(views), 4.2), squeeze=False)
    for ax, view in zip(axes[0], views):
        plot_curve_bands(tda_by_view[view][split], y, kind=kind, sample_range=sample_range, title=view, ax=ax)
    fig.suptitle(title or f"{kind.capitalize()} curves by class (median and IQR, {split} set)")
    fig.tight_layout()
    return fig


def plot_diagram_comparison(tda_split, y, subject_indices=None, n_per_class=2, title=""):
    """H0 and H1 persistence diagrams of a few AD and CN subjects."""
    subjects = subject_indices if subject_indices is not None else _default_subjects(y, n_per_class)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key in zip(axes, ("diag_h0", "diag_h1")):
        allv = []
        for i in subjects:
            d = np.asarray(tda_split[key][i])
            if d.size:
                c = int(y[i])
                ax.scatter(d[:, 0], d[:, 1], s=12, color=CLASS_COLORS[c], alpha=0.6, label=f"{CLASS_NAMES[c]} #{i}")
                allv.append(d)
        if allv:
            lo, hi = float(np.vstack(allv).min()), float(np.vstack(allv).max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set(xlabel="Birth", ylabel="Death", title=key[-2:].upper())
        ax.legend(fontsize=7)
    fig.suptitle(title or "Persistence diagrams (red = AD, blue = CN)")
    fig.tight_layout()
    return fig


def plot_class_templates(templates_by_view, views):
    """AD template, CN template and AD - CN difference for each view."""
    fig, axes = plt.subplots(len(views), 3, figsize=(10, 3.2 * len(views)), squeeze=False)
    for row, view in zip(axes, views):
        t = templates_by_view[view]
        vmax = np.abs(t["diff_display"]).max() or 1.0
        for ax, key, cmap, kw in (
            (row[0], "AD_display", "gray", {}),
            (row[1], "CN_display", "gray", {}),
            (row[2], "diff_display", "bwr", {"vmin": -vmax, "vmax": vmax}),
        ):
            im = ax.imshow(t[key], cmap=cmap, **kw)
            ax.set_title(f"{view}: {key.replace('_display', '').replace('diff', 'AD - CN')}", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Simplicial complexes on point clouds
# ─────────────────────────────────────────────────────────────────────────────

def _draw_complex(ax, background, points, radius, filtration_type, weights=None, point_size=6):
    from .topology import simplex_tree_for_points_at_radius

    ax.imshow(background, cmap="gray")
    if points.shape[0]:
        st = simplex_tree_for_points_at_radius(points, filtration_type, radius=radius, weights=weights)
        for simplex, _ in st.get_simplices():
            if len(simplex) == 2:
                p, q = points[simplex[0]], points[simplex[1]]
                ax.plot([p[0], q[0]], [p[1], q[1]], color="cyan", linewidth=0.5, alpha=0.7, zorder=2)
            elif len(simplex) == 3:
                ax.add_patch(plt.Polygon(points[list(simplex)], closed=True, facecolor="cyan", alpha=0.15, edgecolor=None, zorder=1))
        ax.scatter(points[:, 0], points[:, 1], s=point_size, color="red", zorder=3)
    ax.axis("off")


def plot_point_cloud_threshold_preview(templates, view, filtration_type="alpha", radius_fractions=(0.1, 0.25, 0.5, 1.0, 2.0)):
    """Template point cloud (AD top row, CN bottom row) and its simplicial
    complex at growing radii, as fractions of the configured maximum radius.

    Useful to choose the intensity threshold of a view: too low gives a dense
    cloud that fills the brain immediately, too high leaves sparse noise.
    """
    from .preprocessing import point_cloud_and_weights_from_mask, point_cloud_from_mask
    from .topology import resolve_filtration_name

    ftype = resolve_filtration_name(filtration_type)
    max_radius = config.MAX_RADIUS_BY_FILTRATION[ftype]
    fig, axes = plt.subplots(2, len(radius_fractions), figsize=(3.2 * len(radius_fractions), 6.8), squeeze=False)

    for row, cls in enumerate(("AD", "CN")):
        background, mask = templates[f"{cls}_display"], templates[f"{cls}_foreground_mask"]
        if ftype == "weighted_rips":
            points, weights = point_cloud_and_weights_from_mask(mask, background)
        else:
            points, weights = point_cloud_from_mask(mask), None
        for ax, frac in zip(axes[row], radius_fractions):
            _draw_complex(ax, background, points, max_radius * frac, ftype, weights=weights)
            ax.set_title(f"{cls}: r = {max_radius * frac:g} px", fontsize=9)

    fig.suptitle(f"{view}: {ftype} complex on the class templates (threshold {templates['point_cloud_threshold']})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PCA and results
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_scree(pca_result, max_components=50, title="Explained variance"):
    ev = pca_result["pca"].explained_variance_ratio_
    n = min(max_components, len(ev))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, n + 1), ev[:n], color="steelblue", alpha=0.7, label="per component")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, n + 1), np.cumsum(ev)[:n], color="darkorange", marker="o", markersize=3, label="cumulative")
    ax2.axhline(0.95, linestyle="--", color="gray", linewidth=1)
    ax2.set_ylim(0, 1.05)
    ax.set(xlabel="Principal component", ylabel="Explained variance", title=title)
    ax2.set_ylabel("Cumulative")
    fig.tight_layout()
    return fig


def plot_pca_scatter(pca_result, y_train, y_test, pc_x=0, pc_y=1, title=""):
    """Train (circles) and test (crosses) subjects on two PCs."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ev = pca_result["pca"].explained_variance_ratio_
    for Z, y, marker, part in ((pca_result["Z_train"], y_train, "o", "train"), (pca_result["Z_test"], y_test, "X", "test")):
        y = np.asarray(y)
        for cls in (0, 1):
            m = y == cls
            ax.scatter(Z[m, pc_x], Z[m, pc_y], c=CLASS_COLORS[cls], marker=marker, alpha=0.6 if part == "train" else 1.0,
                       edgecolors="k", linewidths=0.3, label=f"{CLASS_NAMES[cls]} ({part})")
    ax.set(xlabel=f"PC{pc_x + 1} ({ev[pc_x]:.1%})", ylabel=f"PC{pc_y + 1} ({ev[pc_y]:.1%})", title=title or "PCA of topological features")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_results_bars(summary, metric="balanced_accuracy", title=""):
    """Horizontal bars of ``metric`` per model, train vs test (from
    ``classification.summarize_results``)."""
    pivot = summary.pivot_table(index="model", columns="split", values=metric, aggfunc="first")
    pivot = pivot.reindex(columns=[c for c in ("train", "test") if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pivot) + 1.5))
    pivot.plot.barh(ax=ax, color={"train": "lightgray", "test": "tab:green"})
    ax.axvline(0.5, linestyle="--", color="k", linewidth=0.8)
    ax.set(xlabel=metric.replace("_", " "), xlim=(0, 1.05), ylabel="", title=title)
    fig.tight_layout()
    return fig
