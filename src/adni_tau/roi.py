"""Brain masks and a data-driven region of interest (ROI).

The ROI is found in four steps, all fitted on *training* images only:

1. **Brain mask** - keep pixels that are above a small intensity threshold in
   most images (removes the background around the brain).
2. **PCA** - SVD of the centred, masked images; each right singular vector is
   an "eigenimage" (a spatial mode of variation across subjects).
3. **Group test** - a Welch ANOVA on each PC score across diagnostic groups
   selects the components whose scores differ between groups.
4. **ROI** - the absolute eigenimages of the selected PCs are combined with
   their F statistics as weights, smoothed, thresholded at a high percentile
   and cleaned with morphological operations. Connected components whose
   mean weight is high enough form the final ROI(s).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label

from .stats import select_pcs_by_group_test


def brain_mask(X: np.ndarray, shape: tuple[int, int], intensity_threshold: float = 0.02, coverage: float = 0.7) -> np.ndarray:
    """Binary ``(H, W)`` mask of pixels above ``intensity_threshold`` in at
    least ``coverage`` of the images in ``X`` (``(n, H*W)``, values in [0, 1])."""
    frequency = (np.asarray(X) >= intensity_threshold).mean(axis=0)
    return (frequency > coverage).astype(np.uint8).reshape(shape)


@dataclass
class ROIResult:
    """Everything produced by :func:`pca_roi`, kept for plotting and reporting."""

    shape: tuple[int, int]
    brain_mask: np.ndarray
    eigenimages: np.ndarray            # (k, H, W), zero outside the brain mask
    explained_variance: np.ndarray     # (k,) fraction of variance per PC
    scores: np.ndarray                 # (n_train, k)
    group_tests: pd.DataFrame          # output of select_pcs_by_group_test
    selected_pcs: list[int]            # 0-based indices used to build the ROI
    weight_map: np.ndarray             # (H, W) smoothed F-weighted eigenimage map
    components: list[np.ndarray] = field(default_factory=list)  # kept ROI components, (H, W) each

    @property
    def roi(self) -> np.ndarray:
        """Union of all kept components as one binary ``(H, W)`` mask."""
        if not self.components:
            return np.zeros(self.shape, dtype=np.uint8)
        return np.any(self.components, axis=0).astype(np.uint8)

    def masks(self) -> dict[str, np.ndarray]:
        """Named masks: brain mask, full ROI and each ROI component."""
        out = {"brain": self.brain_mask, "roi": self.roi}
        out.update({f"roi_comp{i + 1}": c.astype(np.uint8) for i, c in enumerate(self.components)})
        return out


def _touches_border(component: np.ndarray, margin: int) -> bool:
    """Whether ``component`` reaches within ``margin`` pixels of the image edge."""
    return bool(
        component[:margin].any() or component[-margin:].any() or component[:, :margin].any() or component[:, -margin:].any()
    )


def pca_roi(
    X: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    n_components: int = 15,
    max_selected: int = 3,
    alpha: float = 0.01,
    correction: str = "bonferroni",
    intensity_threshold: float = 0.02,
    smoothing_sigma: float = 1.0,
    roi_percentile: float = 90,
    keep_percentile: float = 95,
    morph_size: int = 5,
    drop_border_components: bool = True,
) -> ROIResult:
    """Find a region of interest whose intensity pattern separates the groups.

    Parameters
    ----------
    X, labels
        Training images ``(n, H*W)`` and their group labels.
    n_components
        Number of PCs to test.
    max_selected
        Use at most this many significant PCs (the most significant first).
    alpha, correction
        Welch-ANOVA significance level and multiple-testing correction.
    roi_percentile
        Pixels of the smoothed weight map above this percentile form the
        candidate ROI.
    keep_percentile
        Connected components whose mean weight is below this percentile of
        the weight map are discarded.
    drop_border_components
        Discard components touching the image border; in these slices they
        fall on the neck or skull edge rather than on brain tissue.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = brain_mask(X, shape, intensity_threshold=intensity_threshold)
    inside = mask.reshape(-1) == 1

    X_masked = X[:, inside]
    X_centered = X_masked - X_masked.mean(axis=0)
    _, s, Vt = svd(X_centered, full_matrices=False)
    k = min(n_components, Vt.shape[0])

    scores = X_centered @ Vt[:k].T
    explained = (s**2 / np.sum(s**2))[:k]

    eigenimages = np.zeros((k, shape[0] * shape[1]))
    eigenimages[:, inside] = Vt[:k]
    eigenimages = eigenimages.reshape(k, *shape)

    tests = select_pcs_by_group_test(scores, labels, method="welch", alpha=alpha, correction=correction)
    significant = tests[tests["reject_null"]]
    selected = [int(pc[2:]) - 1 for pc in significant["pc"]][:max_selected]
    f_by_pc = significant.set_index("pc")["F"]

    weight_map = np.zeros(shape)
    for pc in selected:
        weight_map += f_by_pc[f"PC{pc + 1}"] * np.abs(eigenimages[pc])
    weight_map = gaussian_filter(weight_map, sigma=smoothing_sigma)

    components = []
    if selected:
        candidate = weight_map >= np.percentile(weight_map, roi_percentile)
        structure = np.ones((morph_size, morph_size))
        candidate = binary_closing(binary_opening(candidate, structure=structure), structure=structure)

        labelled, n_found = label(candidate)
        keep_threshold = np.percentile(weight_map, keep_percentile)
        scored = []
        for i in range(1, n_found + 1):
            comp = labelled == i
            if drop_border_components and _touches_border(comp, margin=3):
                continue  # at the image edge: neck / skull, not brain tissue
            mean_weight = weight_map[comp].mean()
            if mean_weight >= keep_threshold:
                scored.append((mean_weight, comp.astype(np.uint8)))
        components = [c for _, c in sorted(scored, key=lambda t: t[0], reverse=True)]

    return ROIResult(
        shape=shape,
        brain_mask=mask,
        eigenimages=eigenimages,
        explained_variance=explained,
        scores=scores,
        group_tests=tests,
        selected_pcs=selected,
        weight_map=weight_map,
        components=components,
    )
