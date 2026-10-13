"""Class-average templates (AD / CN) built on the training set only, their
TDA, and bottleneck / Wasserstein distances of each subject to them
("nearest-template" classification)."""

import gudhi
import numpy as np
import pandas as pd

from . import config
from .preprocessing import (
    foreground_mask,
    foreground_mask_quantile,
    point_cloud_and_weights_from_mask,
    point_cloud_from_mask,
    preprocess_for_tda,
    preprocess_for_tda_quantile,
    preprocess_for_template_display,
)
from .topology import compute_tda_from_grid, compute_tda_from_points, resolve_filtration_name

CLASS_LABELS = (("AD", 0), ("CN", 1))


def compute_class_templates_for_view(train_imgs, y_train, lower_star_threshold=None, point_cloud_threshold=None) -> dict:
    """Class-average templates of one view (training images only).

    - ``<cls>_display``: mean of the images with background set to 0 (for
      display and for the 3D filtration);
    - ``<cls>_tda_grid``: mean of the lower-star grids (background set to 1);
    - ``<cls>_foreground_mask``: pixels above ``point_cloud_threshold`` in at
      least 50% of the class, used to build the template point cloud.
    """
    train_imgs, y_train = np.asarray(train_imgs), np.asarray(y_train)

    display = np.stack([preprocess_for_template_display(img, threshold=lower_star_threshold) for img in train_imgs])
    grids = np.stack([preprocess_for_tda(img, threshold=lower_star_threshold) for img in train_imgs])
    masks = np.stack([foreground_mask(img, threshold=point_cloud_threshold) for img in train_imgs])

    templates = {"lower_star_threshold": lower_star_threshold, "point_cloud_threshold": point_cloud_threshold}
    for cls, label in CLASS_LABELS:
        sel = y_train == label
        templates[f"{cls}_display"] = display[sel].mean(axis=0)
        templates[f"{cls}_tda_grid"] = grids[sel].mean(axis=0)
        templates[f"{cls}_foreground_mask"] = masks[sel].mean(axis=0) >= 0.5
    templates["diff_display"] = templates["AD_display"] - templates["CN_display"]
    return templates


def _template_tda_from_parts(grid, mask, display, filtration_type):
    sample_range = config.LANDSCAPE_RANGE_BY_FILTRATION[filtration_type]
    if filtration_type == "lower_star":
        return compute_tda_from_grid(grid, sample_range=sample_range)
    if filtration_type == "weighted_rips":
        points, weights = point_cloud_and_weights_from_mask(mask, display)
        return compute_tda_from_points(points, filtration_type, sample_range=sample_range, weights=weights)
    return compute_tda_from_points(point_cloud_from_mask(mask), filtration_type, sample_range=sample_range)


def compute_template_tda(templates: dict, filtration_type=None) -> dict:
    """TDA of the AD and CN templates with the chosen filtration.
    Returns ``{"AD": tda_dict, "CN": tda_dict}``."""
    ftype = config.FILTRATION_TYPE if filtration_type is None else resolve_filtration_name(filtration_type)
    return {
        cls: _template_tda_from_parts(
            templates[f"{cls}_tda_grid"], templates[f"{cls}_foreground_mask"], templates[f"{cls}_display"], ftype
        )
        for cls, _ in CLASS_LABELS
    }


def compute_class_template_tda_quantile(train_imgs, y_train, quantile, filtration_type="lower_star") -> dict:
    """Template TDA with per-image quantile thresholds: every subject is
    thresholded with its own quantile *before* averaging."""
    ftype = resolve_filtration_name(filtration_type)
    train_imgs, y_train = np.asarray(train_imgs), np.asarray(y_train)

    grids = np.stack([preprocess_for_tda_quantile(img, quantile) for img in train_imgs])
    masks = np.stack([foreground_mask_quantile(img, quantile) for img in train_imgs])
    display = np.stack([preprocess_for_template_display(img) for img in train_imgs])

    return {
        cls: _template_tda_from_parts(
            grids[y_train == label].mean(axis=0),
            masks[y_train == label].mean(axis=0) >= 0.5,
            display[y_train == label].mean(axis=0),
            ftype,
        )
        for cls, label in CLASS_LABELS
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distances between diagrams
# ─────────────────────────────────────────────────────────────────────────────

def safe_bottleneck_distance(diag_a, diag_b) -> float:
    if diag_a.shape[0] == 0 and diag_b.shape[0] == 0:
        return 0.0
    return gudhi.bottleneck_distance(diag_a, diag_b)


def safe_wasserstein_distance(diag_a, diag_b) -> float:
    """1-Wasserstein distance (Hera backend, no POT needed)."""
    if diag_a.shape[0] == 0 and diag_b.shape[0] == 0:
        return 0.0
    return gudhi.hera.wasserstein_distance(diag_a, diag_b, order=1, internal_p=np.inf)


_DISTANCES = {"bottleneck": safe_bottleneck_distance, "wasserstein": safe_wasserstein_distance}


def distance_to_class_templates_one_view(sample_tda, template_tda, metric="bottleneck"):
    """``(d_AD, d_CN)`` for one subject and one view, with d = d_H0 + d_H1."""
    if metric not in _DISTANCES:
        raise ValueError("metric must be 'bottleneck' or 'wasserstein'.")
    dist = _DISTANCES[metric]
    return tuple(
        dist(sample_tda["diag_h0"], template_tda[cls]["diag_h0"]) + dist(sample_tda["diag_h1"], template_tda[cls]["diag_h1"])
        for cls in ("AD", "CN")
    )


def collect_template_distance_features(tda_by_view, template_tda_by_view, y, views, split="test", metric="bottleneck") -> pd.DataFrame:
    """Per-view and total distances to the AD/CN templates, and the
    nearest-template prediction (0 = AD if closer to AD, 1 = CN otherwise)."""
    rows = []
    for i in range(len(y)):
        row = {"subject_index": i}
        total_ad = total_cn = 0.0
        for view in views:
            sample = {k: tda_by_view[view][split][k][i] for k in ("diag_h0", "diag_h1")}
            d_ad, d_cn = distance_to_class_templates_one_view(sample, template_tda_by_view[view], metric=metric)
            row.update({f"{view}_d_AD": d_ad, f"{view}_d_CN": d_cn})
            total_ad += d_ad
            total_cn += d_cn
        row.update({"total_d_AD": total_ad, "total_d_CN": total_cn, "pred": 0 if total_ad <= total_cn else 1})
        rows.append(row)
    return pd.DataFrame(rows)
