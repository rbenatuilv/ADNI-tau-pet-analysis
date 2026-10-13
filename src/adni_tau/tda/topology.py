"""Filtrations, persistence diagrams and their vectorisations.

The lower-star filtration works directly on the pixel grid
(``gudhi.CubicalComplex``). The geometric filtrations work on a 2D point
cloud of above-threshold pixels (see :mod:`preprocessing`):

- ``rips``          ``gudhi.RipsComplex`` (Vietoris-Rips);
- ``delaunaycech``  ``gudhi.DelaunayCechComplex``: same persistent homology as
                    the Cech complex, built on the Delaunay triangulation;
- ``alpha``         ``gudhi.AlphaComplex``;
- ``weighted_rips`` ``WeightedRipsComplex``: Rips with per-point weights.

Every function returns the same dictionary format (``diag_h0``, ``diag_h1``,
``land_h0``, ``land_h1``, ``betti_h0``, ``betti_h1``, ``lifetime_h0``,
``lifetime_h1``), so downstream code does not depend on the filtration.
"""

import gudhi
import numpy as np
from gudhi.representations import Landscape
from gudhi.weighted_rips_complex import WeightedRipsComplex
from scipy.spatial.distance import pdist, squareform

from . import config, preprocessing

TDA_KEYS = ("diag_h0", "diag_h1", "land_h0", "land_h1", "betti_h0", "betti_h1", "lifetime_h0", "lifetime_h1")


def empty_diagram():
    return np.empty((0, 2), dtype=np.float64)


def extract_diagram(persistence, homology_dim, min_persistence=None, cap_infinite_at=None):
    """Persistence diagram ``(n, 2)`` of dimension ``homology_dim``.

    Points with persistence below ``min_persistence`` are dropped. Points with
    infinite death are dropped by default; if ``cap_infinite_at`` is given
    their death is set to that value instead. This matters for point-cloud
    filtrations, which are truncated at a maximum radius: a feature still
    alive at that radius is very persistent and should be kept, not ignored.
    """
    min_persistence = config.MIN_PERSISTENCE if min_persistence is None else min_persistence

    points = []
    for dim, (birth, death) in persistence:
        if dim != homology_dim:
            continue
        if death == float("inf"):
            if cap_infinite_at is None:
                continue
            death = cap_infinite_at
        points.append([birth, death])

    if not points:
        return empty_diagram()

    diag = np.asarray(points, dtype=np.float64)
    diag = diag[(diag[:, 1] - diag[:, 0]) >= min_persistence]
    return diag if diag.size else empty_diagram()


# ─────────────────────────────────────────────────────────────────────────────
# Vectorisations
# ─────────────────────────────────────────────────────────────────────────────

def _alive_matrix(diag, sample_range, n_points):
    """Grid ``t`` and boolean ``(n_features, n_points)`` "alive at t" matrix,
    with the convention birth <= t < death."""
    t = np.linspace(sample_range[0], sample_range[1], n_points)
    births, deaths = diag[:, :1], diag[:, 1:2]
    return (births <= t[None, :]) & (t[None, :] < deaths)


def betti_curve(diag, sample_range=None, n_points=None):
    """Betti curve beta(t): number of features alive at filtration value t.

    Simpler than a landscape (it counts features instead of describing each
    one) but very easy to read, e.g. "at t = 0.4 AD subjects have 15 more
    components than CN subjects".
    """
    sample_range = config.LANDSCAPE_RANGE if sample_range is None else sample_range
    n_points = config.N_BETTI_POINTS if n_points is None else n_points
    if diag is None or diag.shape[0] == 0:
        return np.zeros(n_points, dtype=np.float64)
    return _alive_matrix(diag, sample_range, n_points).sum(axis=0).astype(np.float64)


def lifetime_curve(diag, sample_range=None, n_points=None):
    """Lifetime curve L(t), a.k.a. accumulated persistence function
    (Biscio & Moller, 2016): sum of the persistence (death - birth) of the
    features alive at t. Between the Betti curve (every feature counts 1) and
    the landscape: long-lived features weigh more than short-lived ones."""
    sample_range = config.LANDSCAPE_RANGE if sample_range is None else sample_range
    n_points = config.N_LIFETIME_POINTS if n_points is None else n_points
    if diag is None or diag.shape[0] == 0:
        return np.zeros(n_points, dtype=np.float64)
    lifetimes = diag[:, 1:2] - diag[:, :1]
    return (_alive_matrix(diag, sample_range, n_points) * lifetimes).sum(axis=0).astype(np.float64)


def safe_landscape(diag, sample_range=None):
    """Flattened persistence landscape ``(1, layers * points)``; zeros for an
    empty diagram so a subject without features does not break the pipeline."""
    sample_range = config.LANDSCAPE_RANGE if sample_range is None else sample_range
    if diag is None or diag.shape[0] == 0:
        return np.zeros((1, config.N_LANDSCAPE_LAYERS * config.N_LANDSCAPE_POINTS), dtype=np.float64)
    return Landscape(
        num_landscapes=config.N_LANDSCAPE_LAYERS,
        resolution=config.N_LANDSCAPE_POINTS,
        sample_range=sample_range,
    ).fit_transform([diag])


def vectorise(h0, h1, sample_range):
    """All vectorisations of a pair of H0/H1 diagrams, in the common format."""
    return {
        "diag_h0": h0,
        "diag_h1": h1,
        "land_h0": safe_landscape(h0, sample_range=sample_range),
        "land_h1": safe_landscape(h1, sample_range=sample_range),
        "betti_h0": betti_curve(h0, sample_range=sample_range),
        "betti_h1": betti_curve(h1, sample_range=sample_range),
        "lifetime_h0": lifetime_curve(h0, sample_range=sample_range),
        "lifetime_h1": lifetime_curve(h1, sample_range=sample_range),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Lower-star filtration
# ─────────────────────────────────────────────────────────────────────────────

def compute_tda_from_grid(grid, sample_range=None):
    """Lower-star (sublevel-set) filtration of an already thresholded image."""
    sample_range = config.LANDSCAPE_RANGE if sample_range is None else sample_range
    grid = np.asarray(grid, dtype=np.float64)

    cc = gudhi.CubicalComplex(dimensions=grid.shape, top_dimensional_cells=grid.flatten())
    cc.compute_persistence()
    persistence = cc.persistence()

    return vectorise(extract_diagram(persistence, 0), extract_diagram(persistence, 1), sample_range)


# ─────────────────────────────────────────────────────────────────────────────
# Point-cloud filtrations
# ─────────────────────────────────────────────────────────────────────────────

def resolve_filtration_name(name: str) -> str:
    """Map a human-friendly name ("Vietoris Rips", "Delaunay-Cech", ...) to
    the internal one ("rips", "delaunaycech", ...)."""
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if key not in config.FILTRATION_ALIASES:
        valid = sorted(set(config.FILTRATION_ALIASES.values()))
        raise ValueError(f"Unknown filtration '{name}'. Available: {valid}.")
    return config.FILTRATION_ALIASES[key]


def _weighted_rips_simplex_tree(points, weights, max_radius, max_dimension=2):
    """Weighted Rips filtration (Buchet-Chazal-Oudot-Sheehy, p = 1):

    - vertex i enters at ``w_i``;
    - edge (i, j) enters at ``max(w_i, w_j, d(i, j) + w_i/2 + w_j/2)``;
    - higher simplices by flag expansion, as in the plain Rips complex.

    ``WeightedRipsComplex`` uses half-weights (its vertex filtration is twice
    the weight), so ``weights / 2`` is passed to make vertex values equal
    the weights computed in :mod:`preprocessing`.
    """
    if points.shape[0] == 0:
        return gudhi.SimplexTree()
    dist = squareform(pdist(points))
    wrc = WeightedRipsComplex(distance_matrix=dist, weights=np.asarray(weights) / 2.0, max_filtration=max_radius)
    return wrc.create_simplex_tree(max_dimension=max_dimension)


def simplex_tree_for_points_at_radius(points, filtration_type, radius, weights=None, max_dimension=2):
    """Simplex tree of ``filtration_type`` truncated at ``radius`` (for
    ``rips`` the radius is an edge length). Works for 2D and 3D points."""
    points = np.asarray(points, dtype=np.float64)

    if filtration_type == "rips":
        return gudhi.RipsComplex(points=points, max_edge_length=radius).create_simplex_tree(max_dimension=max_dimension)
    if filtration_type == "delaunaycech":
        # output_squared_values=False keeps filtration values in radius units.
        return gudhi.DelaunayCechComplex(points=points).create_simplex_tree(
            max_alpha_square=radius**2, output_squared_values=False
        )
    if filtration_type == "alpha":
        return gudhi.AlphaComplex(points=points).create_simplex_tree(
            max_alpha_square=radius**2, output_squared_values=False
        )
    if filtration_type == "weighted_rips":
        if weights is None:
            raise ValueError("weighted_rips needs one weight per point (see preprocessing.weighted_point_cloud_from_image).")
        return _weighted_rips_simplex_tree(points, weights, max_radius=radius, max_dimension=max_dimension)
    raise ValueError(f"'{filtration_type}' is not a point-cloud filtration.")


def compute_tda_from_points(points, filtration_type, sample_range=None, min_persistence=None, weights=None):
    """Persistence diagrams and vectorisations of a point cloud."""
    sample_range = config.LANDSCAPE_RANGE_BY_FILTRATION[filtration_type] if sample_range is None else sample_range
    if min_persistence is None:
        min_persistence = config.MIN_PERSISTENCE_BY_FILTRATION[filtration_type]
    points = np.asarray(points, dtype=np.float64)

    if points.shape[0] == 0:
        return vectorise(empty_diagram(), empty_diagram(), sample_range)

    st = simplex_tree_for_points_at_radius(
        points, filtration_type, radius=config.MAX_RADIUS_BY_FILTRATION[filtration_type], weights=weights
    )
    st.compute_persistence()
    persistence = st.persistence()

    # Features still alive at the maximum radius are capped there, not dropped.
    cap = sample_range[1]
    h0 = extract_diagram(persistence, 0, min_persistence=min_persistence, cap_infinite_at=cap)
    h1 = extract_diagram(persistence, 1, min_persistence=min_persistence, cap_infinite_at=cap)
    return vectorise(h0, h1, sample_range)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch on a single image / a batch of images
# ─────────────────────────────────────────────────────────────────────────────

def compute_tda_one_image(img, filtration_type=None, threshold=None, quantile=None):
    """Preprocess one image and compute its TDA with ``filtration_type``.

    Either an absolute ``threshold`` (default: the global one of
    :mod:`config`) or a per-image ``quantile`` threshold is used.

    Returns ``(representation, tda_dict)`` where the representation is the
    thresholded grid (lower-star) or the point cloud.
    """
    ftype = config.FILTRATION_TYPE if filtration_type is None else resolve_filtration_name(filtration_type)
    adaptive = quantile is not None

    if ftype == "lower_star":
        grid = preprocessing.preprocess_for_tda_quantile(img, quantile) if adaptive else preprocessing.preprocess_for_tda(img, threshold)
        return grid, compute_tda_from_grid(grid)

    if ftype == "weighted_rips":
        points, weights = (
            preprocessing.weighted_point_cloud_from_image_quantile(img, quantile)
            if adaptive
            else preprocessing.weighted_point_cloud_from_image(img, threshold=threshold)
        )
        return points, compute_tda_from_points(points, ftype, weights=weights)

    points = (
        preprocessing.image_to_point_cloud_quantile(img, quantile)
        if adaptive
        else preprocessing.image_to_point_cloud(img, threshold=threshold)
    )
    return points, compute_tda_from_points(points, ftype)


def compute_tda_for_images(images, filtration_type=None, threshold=None, quantile=None, name="dataset", verbose=False):
    """TDA of a batch of images.

    Returns a dict of per-image lists (one entry per image for every key in
    ``TDA_KEYS``) plus ``grid`` (the representations), ``filtration_type``
    and, for quantile thresholds, the resulting absolute ``thresholds``.
    """
    ftype = config.FILTRATION_TYPE if filtration_type is None else resolve_filtration_name(filtration_type)
    out = {key: [] for key in TDA_KEYS}
    out.update({"grid": [], "filtration_type": ftype, "threshold": threshold})
    if quantile is not None:
        out["thresholds"] = []

    for i, img in enumerate(images):
        if verbose:
            print(f"[{name}] image {i + 1}/{len(images)} ({ftype})")
        if quantile is not None:
            out["thresholds"].append(preprocessing.quantile_threshold_value(preprocessing._prepare(img), quantile))
        rep, tda = compute_tda_one_image(img, ftype, threshold=threshold, quantile=quantile)
        out["grid"].append(rep)
        for key in TDA_KEYS:
            out[key].append(tda[key])
    return out
