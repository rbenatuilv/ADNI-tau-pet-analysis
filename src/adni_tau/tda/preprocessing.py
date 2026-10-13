"""Image preprocessing before TDA: thresholding, downsampling and point clouds.

Functions that apply a threshold accept an optional ``threshold``; when it is
``None`` the global default from :mod:`config` is used. The ``*_quantile``
variants compute the threshold *per image* as a quantile of its own intensity
distribution instead of using one absolute value for every subject.
"""

import numpy as np

from . import config


def downsample_2d_mean(img, factor=1):
    """Block-mean downsampling by ``factor`` (the trailing border is cropped)."""
    if factor is None or factor <= 1:
        return img
    h, w = img.shape
    h_new, w_new = h // factor, w // factor
    img = img[: h_new * factor, : w_new * factor]
    return img.reshape(h_new, factor, w_new, factor).mean(axis=(1, 3))


def _prepare(img):
    img = np.asarray(img, dtype=np.float64).copy()
    if config.USE_DOWNSAMPLING:
        img = downsample_2d_mean(img, factor=config.DOWNSAMPLE_FACTOR)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Fixed thresholds
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(img, low_value_fill, threshold=None):
    """Set pixels below ``threshold`` to ``low_value_fill``
    (``config.TDA_LOW_VALUE_FILL`` for lower-star, ``config.TEMPLATE_LOW_VALUE_FILL``
    for display)."""
    threshold = config.LOWER_STAR_INTENSITY_THRESHOLD if threshold is None else threshold
    img = _prepare(img)
    img[img < threshold] = low_value_fill
    return img


def preprocess_for_tda(img, threshold=None):
    """Grid ready for the lower-star filtration (background -> 1.0)."""
    return apply_threshold(img, low_value_fill=config.TDA_LOW_VALUE_FILL, threshold=threshold)


def preprocess_for_template_display(img, threshold=None):
    """Image ready for display / template averaging (background -> 0.0)."""
    return apply_threshold(img, low_value_fill=config.TEMPLATE_LOW_VALUE_FILL, threshold=threshold)


def foreground_mask(img, threshold=None):
    """Boolean mask of pixels ``>= threshold`` (default
    ``config.POINT_CLOUD_INTENSITY_THRESHOLD``). Unlike :func:`apply_threshold`,
    background pixels are excluded rather than moved."""
    threshold = config.POINT_CLOUD_INTENSITY_THRESHOLD if threshold is None else threshold
    return _prepare(img) >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive (per-image quantile) thresholds
# ─────────────────────────────────────────────────────────────────────────────
# An absolute threshold leaves more or less foreground depending on the global
# brightness of each scan (scanner calibration, noise, physiology). A quantile
# threshold instead keeps the same *fraction* of pixels for every subject,
# e.g. quantile=0.75 keeps the 25% brightest pixels of each image.

def quantile_threshold_value(img, quantile):
    """Intensity value at ``quantile`` of this image's pixel distribution."""
    return float(np.quantile(np.asarray(img, dtype=np.float64), quantile))


def apply_threshold_quantile(img, quantile, low_value_fill):
    img = _prepare(img)
    img[img < quantile_threshold_value(img, quantile)] = low_value_fill
    return img


def preprocess_for_tda_quantile(img, quantile):
    """Like :func:`preprocess_for_tda` with a per-image quantile threshold."""
    return apply_threshold_quantile(img, quantile, low_value_fill=config.TDA_LOW_VALUE_FILL)


def foreground_mask_quantile(img, quantile):
    """Like :func:`foreground_mask` with a per-image quantile threshold."""
    img = _prepare(img)
    return img >= quantile_threshold_value(img, quantile)


# ─────────────────────────────────────────────────────────────────────────────
# Point clouds
# ─────────────────────────────────────────────────────────────────────────────

def _subsample(n_points, max_points, seed):
    if max_points is None or n_points <= max_points:
        return None
    return np.random.default_rng(seed).choice(n_points, size=max_points, replace=False)


def point_cloud_from_mask(mask, max_points=None, seed=None):
    """``(N, 2)`` array of ``(x, y)`` pixel coordinates of the ``True`` pixels,
    randomly subsampled to at most ``max_points`` (fixed seed).

    Coordinates are raw pixels, so distances are directly comparable with the
    pixel radii in :mod:`config`.
    """
    max_points = config.POINT_CLOUD_MAX_POINTS if max_points is None else max_points
    seed = config.POINT_CLOUD_SEED if seed is None else seed

    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = np.column_stack([cols, rows]).astype(np.float64)
    idx = _subsample(len(points), max_points, seed)
    return points if idx is None else points[idx]


def image_to_point_cloud(img, max_points=None, seed=None, threshold=None):
    """Point cloud of the above-threshold pixels of a raw image."""
    return point_cloud_from_mask(foreground_mask(img, threshold=threshold), max_points=max_points, seed=seed)


def image_to_point_cloud_quantile(img, quantile, max_points=None, seed=None):
    """Point cloud with a per-image quantile threshold."""
    return point_cloud_from_mask(foreground_mask_quantile(img, quantile), max_points=max_points, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted point clouds (weighted Rips)
# ─────────────────────────────────────────────────────────────────────────────

def _weights_from_intensity(intensity):
    """Per-point weights following ``config.WEIGHTED_RIPS_MODE``."""
    intensity = np.asarray(intensity, dtype=np.float64)
    base = intensity if config.WEIGHTED_RIPS_MODE == "weak_first" else 1.0 - intensity
    return np.clip(base, 0.0, None) * config.WEIGHTED_RIPS_WEIGHT_SCALE


def point_cloud_and_weights_from_mask(mask, intensity_grid, max_points=None, seed=None):
    """Like :func:`point_cloud_from_mask`, also returning one weight per point
    computed from ``intensity_grid`` (same shape as ``mask``)."""
    max_points = config.POINT_CLOUD_MAX_POINTS if max_points is None else max_points
    seed = config.POINT_CLOUD_SEED if seed is None else seed

    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    points = np.column_stack([cols, rows]).astype(np.float64)
    intensities = np.asarray(intensity_grid, dtype=np.float64)[rows, cols]

    idx = _subsample(len(points), max_points, seed)
    if idx is not None:
        points, intensities = points[idx], intensities[idx]
    return points, _weights_from_intensity(intensities)


def weighted_point_cloud_from_image(img, max_points=None, seed=None, threshold=None):
    """``(points, weights)`` for the weighted Rips filtration."""
    threshold = config.POINT_CLOUD_INTENSITY_THRESHOLD if threshold is None else threshold
    img = _prepare(img)
    return point_cloud_and_weights_from_mask(img >= threshold, img, max_points=max_points, seed=seed)


def weighted_point_cloud_from_image_quantile(img, quantile, max_points=None, seed=None):
    """Weighted point cloud with a per-image quantile threshold."""
    img = _prepare(img)
    mask = img >= quantile_threshold_value(img, quantile)
    return point_cloud_and_weights_from_mask(mask, img, max_points=max_points, seed=seed)
