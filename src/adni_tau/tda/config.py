"""Tunable constants of the TDA pipeline.

Intensity thresholds defined here are defaults only: the notebooks set one
threshold per view and pass it explicitly (``threshold=`` arguments).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

VIEWS = ["axial", "sagittal", "coronal"]

# ImageLoader expects image_size=(W, H); images are then reshaped to (H, W).
VIEW_IMAGE_SIZE = {
    "axial": (218, 182),
    "sagittal": (218, 182),
    "coronal": (182, 182),
}

# Balanced AD/CN subset: all 106 AD subjects and as many CN subjects.
N_TRAIN_PER_CLASS = 96
N_TEST_PER_CLASS = 10

# Seed of the random, class-stratified train/test assignment.
SPLIT_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Thresholding
# ─────────────────────────────────────────────────────────────────────────────

# Pixels below the threshold are treated as background. The two filtration
# families use it differently:
#   - lower-star keeps the full grid and only moves background pixels to
#     TDA_LOW_VALUE_FILL, so a permissive threshold costs nothing;
#   - point-cloud filtrations keep only the above-threshold pixels, so a more
#     selective threshold avoids a dense cloud dominated by a single blob.
LOWER_STAR_INTENSITY_THRESHOLD = 0.25
POINT_CLOUD_INTENSITY_THRESHOLD = 0.3

# Optional block-mean downsampling to reduce TDA cost.
USE_DOWNSAMPLING = False
DOWNSAMPLE_FACTOR = 2

# In a lower-star (sublevel) filtration low values enter first. Background
# pixels are therefore set to the maximum value so they enter last, all at
# once, instead of dominating the filtration.
TDA_LOW_VALUE_FILL = 1.0

# For displaying class-average templates the background is set to 0 instead.
TEMPLATE_LOW_VALUE_FILL = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence diagrams and vectorisations
# ─────────────────────────────────────────────────────────────────────────────

MIN_PERSISTENCE = 0.0       # lower-star: features shorter than this are dropped
N_LANDSCAPE_LAYERS = 5      # number of landscape functions lambda_1..lambda_k
N_LANDSCAPE_POINTS = 400    # samples per landscape layer
LANDSCAPE_RANGE = (0.0, 1.0)

# Betti and lifetime curves share the landscape grid so the x axes match.
N_BETTI_POINTS = N_LANDSCAPE_POINTS
N_LIFETIME_POINTS = N_LANDSCAPE_POINTS


# ─────────────────────────────────────────────────────────────────────────────
# Filtration selection
# ─────────────────────────────────────────────────────────────────────────────

FILTRATION_TYPE = "lower_star"

POINT_CLOUD_FILTRATIONS = ["delaunaycech", "alpha", "rips", "weighted_rips"]

# Human-friendly names accepted by topology.resolve_filtration_name
# (case, spaces, dashes and underscores are ignored).
FILTRATION_ALIASES = {
    "lowerstar": "lower_star",
    "cubical": "lower_star",
    "rips": "rips",
    "vr": "rips",
    "vietorisrips": "rips",
    "delaunaycech": "delaunaycech",
    "alpha": "alpha",
    "alphacomplex": "alpha",
    "weightedrips": "weighted_rips",
    "weightedvietorisrips": "weighted_rips",
}


# ─────────────────────────────────────────────────────────────────────────────
# Point clouds (Rips / Delaunay-Cech / alpha / weighted Rips)
# ─────────────────────────────────────────────────────────────────────────────

# Rips grows combinatorially with the number of points, so at most this many
# above-threshold pixels are kept (random subsample with a fixed seed). The
# other filtrations use the same cloud for a fair comparison.
POINT_CLOUD_MAX_POINTS = 400
POINT_CLOUD_SEED = 1

# Maximum filtration scale, in pixels (point coordinates are raw pixel
# coordinates). Beyond these scales the complex fills the brain and no
# interesting topology is left.
RIPS_MAX_EDGE_LENGTH = 32
ALPHA_MAX_RADIUS = 32
DELAUNAYCECH_MAX_RADIUS = 22


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Rips
# ─────────────────────────────────────────────────────────────────────────────

# Weighted Rips (Buchet-Chazal-Oudot-Sheehy, p = 1) gives each point a weight
# derived from its intensity; low-weight points enter the filtration first.
#   "strong_first": weight = 1 - intensity, so the strong-signal core connects
#                   first and weaker regions are added later (opposite of the
#                   lower-star convention, on purpose).
#   "weak_first":   weight = intensity (same direction as lower-star).
WEIGHTED_RIPS_MODE = "strong_first"

# Weights live in [0, 1] while distances are in pixels, so with a scale of 1
# the weights barely matter; values close to RIPS_MAX_EDGE_LENGTH give the
# intensity real influence.
WEIGHTED_RIPS_WEIGHT_SCALE = 1.0
WEIGHTED_RIPS_MAX_RADIUS = 54


# ─────────────────────────────────────────────────────────────────────────────
# Per-filtration parameters
# ─────────────────────────────────────────────────────────────────────────────

# Lower-star works on intensities in [0, 1]; point-cloud filtrations on pixel
# distances, so they need their own minimum persistence and sampling range.
MIN_PERSISTENCE_BY_FILTRATION = {
    "lower_star": MIN_PERSISTENCE,
    "rips": 4,
    "delaunaycech": 4,
    "alpha": 4,
    "weighted_rips": 4,
}

LANDSCAPE_RANGE_BY_FILTRATION = {
    "lower_star": LANDSCAPE_RANGE,
    "rips": (0.0, RIPS_MAX_EDGE_LENGTH),
    "delaunaycech": (0.0, DELAUNAYCECH_MAX_RADIUS),
    "alpha": (0.0, ALPHA_MAX_RADIUS),
    "weighted_rips": (0.0, WEIGHTED_RIPS_MAX_RADIUS),
}

MAX_RADIUS_BY_FILTRATION = {
    "rips": RIPS_MAX_EDGE_LENGTH,
    "delaunaycech": DELAUNAYCECH_MAX_RADIUS,
    "alpha": ALPHA_MAX_RADIUS,
    "weighted_rips": WEIGHTED_RIPS_MAX_RADIUS,
}
