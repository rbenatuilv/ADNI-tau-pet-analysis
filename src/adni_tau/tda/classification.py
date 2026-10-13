"""AD / CN classification from topological features.

1. **Nearest-template**: a subject is assigned to the class whose average
   template is topologically closer (bottleneck or Wasserstein distance,
   summed over H0, H1 and views). No training beyond building the templates.
2. **Multi-view logistic regression**: landscapes, Betti curves, lifetime
   curves and barcode statistics of all views are concatenated into one
   feature vector per subject and fed to an L2 (or L1) regularised model.

Label convention throughout: ``0 = AD``, ``1 = CN``.
"""

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .templates import collect_template_distance_features

FEATURE_SETS = ("full", "landscape_betti")
BARCODE_STATS = ["n_features", "total_persistence", "max_persistence", "std_persistence"]


def make_logistic_regression(C: float = 1.0, penalty: str = "l2", solver: str = "liblinear", max_iter: int = 2000):
    """L1/L2 logistic regression across scikit-learn versions (``penalty`` is
    deprecated from 1.8 in favour of ``l1_ratio``)."""
    if penalty not in ("l1", "l2"):
        raise ValueError("penalty must be 'l1' or 'l2'.")
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=1.0 if penalty == "l1" else 0.0, solver=solver, max_iter=max_iter)
    return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)


def eval_predictions(y_true, y_pred, name: str, split: str = "", verbose: bool = True) -> dict:
    """Accuracy, balanced accuracy and confusion matrix (optionally printed)."""
    metrics = {
        "name": name,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
    if verbose:
        print(f"\n==== {name}" + (f" ({split})" if split else "") + " ====")
        print(metrics["confusion_matrix"])
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["AD", "CN"], zero_division=0))
    return metrics


def summarize_results(results: list) -> pd.DataFrame:
    """Compact model / split / accuracy table."""
    return pd.DataFrame(
        [
            {"model": r["name"], "split": r.get("split", ""), "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"]}
            for r in results
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Nearest template
# ─────────────────────────────────────────────────────────────────────────────

def run_nearest_template_classification(tda_by_view, template_tda_by_view, y, views, metric="bottleneck", split="test", verbose=True):
    """Nearest-template predictions on ``split``; templates always come from
    the training set. Returns ``(distance DataFrame, metrics)``."""
    dist_df = collect_template_distance_features(tda_by_view, template_tda_by_view, y, views, split=split, metric=metric)
    metrics = eval_predictions(y, dist_df["pred"].to_numpy(), name=f"Nearest template ({metric})", split=split, verbose=verbose)
    return dist_df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-view feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def barcode_summary_stats(diag: np.ndarray) -> np.ndarray:
    """Number of features and total / max / std of the persistences: coarse
    information complementary to the landscape, in only four columns."""
    if diag is None or diag.shape[0] == 0:
        return np.zeros(4, dtype=np.float64)
    pers = diag[:, 1] - diag[:, 0]
    return np.array([diag.shape[0], pers.sum(), pers.max(), pers.std()], dtype=np.float64)


def build_multiview_feature_matrix(tda_by_view: dict, views: list, split: str, feature_set: str = "full"):
    """Concatenate, for every view, the vectorised diagrams of each subject.

    Per view: landscape H0, landscape H1, Betti H0, Betti H1, lifetime H0,
    lifetime H1 and, for ``feature_set="full"``, barcode statistics H0 and H1.
    Use ``"landscape_betti"`` (no barcode statistics) for the 3D filtration.

    Returns ``(X, feature_groups)`` where ``feature_groups[view]`` is the
    ``(start, end)`` column range of that view.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}.")

    blocks, groups, start = [], {}, 0
    for view in views:
        tda = tda_by_view[view][split]
        view_blocks = [np.vstack(tda[key]) for key in ("land_h0", "land_h1", "betti_h0", "betti_h1", "lifetime_h0", "lifetime_h1")]
        if feature_set == "full":
            view_blocks += [np.vstack([barcode_summary_stats(d) for d in tda[key]]) for key in ("diag_h0", "diag_h1")]
        block = np.hstack(view_blocks)
        blocks.append(block)
        groups[view] = (start, start + block.shape[1])
        start += block.shape[1]
    return np.hstack(blocks), groups


def multiview_feature_names(views: list, feature_set: str = "full") -> list:
    """Column names of :func:`build_multiview_feature_matrix`, same order."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}.")
    names = []
    for view in views:
        for h in ("h0", "h1"):
            names += [f"{view}_landscape_{h}_layer{l}_pt{p}" for l in range(config.N_LANDSCAPE_LAYERS) for p in range(config.N_LANDSCAPE_POINTS)]
        for h in ("h0", "h1"):
            names += [f"{view}_betti_{h}_pt{p}" for p in range(config.N_BETTI_POINTS)]
        for h in ("h0", "h1"):
            names += [f"{view}_lifetime_{h}_pt{p}" for p in range(config.N_LIFETIME_POINTS)]
        if feature_set == "full":
            for h in ("h0", "h1"):
                names += [f"{view}_barcode_{h}_{s}" for s in BARCODE_STATS]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-view logistic regression
# ─────────────────────────────────────────────────────────────────────────────

def fit_eval_logistic_multiview(
    tda_by_view: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    views: list,
    C: float = 0.1,
    penalty: str = "l2",
    verbose: bool = True,
    feature_set: str = "full",
    name: str = "Multi-view logistic regression",
):
    """Standardise (fit on train), train a regularised logistic regression on
    the multi-view features and evaluate it on train and test.

    The number of topological features (~14,000) is far larger than the
    number of training subjects (~190), hence the strong default
    regularisation (``C = 0.1``). Comparing train and test metrics shows how
    much the model overfits.
    """
    X_train, groups = build_multiview_feature_matrix(tda_by_view, views, "train", feature_set=feature_set)
    X_test, _ = build_multiview_feature_matrix(tda_by_view, views, "test", feature_set=feature_set)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = make_logistic_regression(C=C, penalty=penalty)
    model.fit(X_train_s, y_train)

    metrics_train = eval_predictions(y_train, model.predict(X_train_s), name=name, split="train", verbose=verbose)
    metrics_test = eval_predictions(y_test, model.predict(X_test_s), name=name, split="test", verbose=verbose)

    return {
        "model": model,
        "scaler": scaler,
        "feature_groups": groups,
        "feature_set": feature_set,
        "metrics_train": metrics_train,
        "metrics_test": metrics_test,
    }


def cross_validate_logistic_multiview(
    tda_by_view: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    views: list,
    C: float = 0.1,
    penalty: str = "l2",
    feature_set: str = "full",
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 0,
) -> dict:
    """Repeated stratified K-fold accuracy of the multi-view model on all
    subjects (train + test pooled).

    The fixed test set has only 20 subjects, so a single test accuracy moves
    in steps of 5 points; this gives a less noisy estimate (mean +/- std over
    ``n_splits * n_repeats`` folds). Scaling is refit inside every fold.
    """
    X = np.vstack([build_multiview_feature_matrix(tda_by_view, views, s, feature_set=feature_set)[0] for s in ("train", "test")])
    y = np.concatenate([y_train, y_test])
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    acc, bal = [], []
    for tr, te in cv.split(X, y):
        model = make_pipeline(StandardScaler(), make_logistic_regression(C=C, penalty=penalty))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        acc.append(accuracy_score(y[te], pred))
        bal.append(balanced_accuracy_score(y[te], pred))
    return {
        "cv_accuracy_mean": float(np.mean(acc)),
        "cv_accuracy_std": float(np.std(acc)),
        "cv_balanced_accuracy_mean": float(np.mean(bal)),
        "cv_balanced_accuracy_std": float(np.std(bal)),
    }


def top_logistic_coefficients(logit_result: dict, views: list, top_n: int = 20) -> pd.DataFrame:
    """Features ranked by |coefficient|. Features are standardised, so
    coefficients are comparable. ``coef > 0`` pushes towards CN, ``coef < 0``
    towards AD."""
    coefs = np.ravel(logit_result["model"].coef_)
    names = multiview_feature_names(views, feature_set=logit_result.get("feature_set", "full"))
    assert len(names) == len(coefs), "Feature names do not match the model (different views order or config?)."

    df = pd.DataFrame({"feature": names, "view": [n.split("_", 1)[0] for n in names], "coef": coefs})
    df["abs_coef"] = df["coef"].abs()
    df["favours"] = np.where(df["coef"] > 0, "CN", "AD")
    return df.sort_values("abs_coef", ascending=False).head(top_n).reset_index(drop=True)


def coefficient_importance_by_view(logit_result: dict) -> pd.DataFrame:
    """Sum of |coefficient| per view. ``n_nonzero`` is informative only with
    an L1 penalty (with L2 almost every coefficient is non-zero)."""
    coefs = np.ravel(logit_result["model"].coef_)
    rows = [
        {"view": view, "sum_abs_coef": np.abs(coefs[s:e]).sum(), "n_nonzero": int(np.sum(coefs[s:e] != 0)), "n_features": e - s}
        for view, (s, e) in logit_result["feature_groups"].items()
    ]
    return pd.DataFrame(rows).sort_values("sum_abs_coef", ascending=False).reset_index(drop=True)


def descriptor_importance(logit_result: dict, views: list) -> pd.DataFrame:
    """Sum of |coefficient| per (view, descriptor, homology dimension), e.g.
    ``axial / landscape / h1``: a coarser reading than single coefficients."""
    coefs = np.ravel(logit_result["model"].coef_)
    names = multiview_feature_names(views, feature_set=logit_result.get("feature_set", "full"))
    parts = [n.split("_") for n in names]
    df = pd.DataFrame({"view": [p[0] for p in parts], "descriptor": [p[1] for p in parts], "homology": [p[2] for p in parts], "abs_coef": np.abs(coefs)})
    return df.groupby(["view", "descriptor", "homology"], as_index=False)["abs_coef"].sum().sort_values("abs_coef", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Coefficient stability (bootstrap)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_logistic_coefficients(tda_by_view, y_train, views, C=0.1, penalty="l2", n_boot=500, random_state=0, verbose=True, feature_set="full"):
    """Refit the model on ``n_boot`` class-stratified bootstrap resamples of
    the training set and return the ``(n_boot, n_features)`` coefficients.

    With p >> n the unregularised logistic MLE does not exist (the classes
    are separable) and the regularised fit invalidates classical Wald
    p-values, so coefficient reliability is assessed empirically instead.
    """
    X_train, _ = build_multiview_feature_matrix(tda_by_view, views, "train", feature_set=feature_set)
    y_train = np.asarray(y_train)
    idx_ad, idx_cn = np.where(y_train == 0)[0], np.where(y_train == 1)[0]
    rng = np.random.default_rng(random_state)

    boot = np.zeros((n_boot, X_train.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(idx_ad, len(idx_ad)), rng.choice(idx_cn, len(idx_cn))])
        model = make_pipeline(StandardScaler(), make_logistic_regression(C=C, penalty=penalty))
        model.fit(X_train[idx], y_train[idx])
        boot[b] = np.ravel(model[-1].coef_)
        if verbose and (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")
    return boot


def top_logistic_coefficients_with_significance(logit_result, boot_coefs, views, top_n=20) -> pd.DataFrame:
    """:func:`top_logistic_coefficients` plus bootstrap statistics: mean,
    std, empirical 95% interval, fraction of resamples with the same sign
    (the most honest reliability measure) and an approximate z / p-value
    (indicative only, not a formal test)."""
    feature_set = logit_result.get("feature_set", "full")
    all_names = multiview_feature_names(views, feature_set=feature_set)
    base = top_logistic_coefficients(logit_result, views, top_n=len(all_names))
    cols = base["feature"].map({n: i for i, n in enumerate(all_names)}).to_numpy()
    sub = boot_coefs[:, cols]

    mean, std = sub.mean(axis=0), sub.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, mean / std, np.nan)
    base = base.assign(
        boot_mean=mean,
        boot_std=std,
        ci_low=np.percentile(sub, 2.5, axis=0),
        ci_high=np.percentile(sub, 97.5, axis=0),
        pct_same_sign=(np.sign(sub) == np.sign(base["coef"].to_numpy())[None, :]).mean(axis=0),
        z_approx=z,
        p_approx=2 * (1 - norm.cdf(np.abs(z))),
    )
    return base.head(top_n).reset_index(drop=True)
