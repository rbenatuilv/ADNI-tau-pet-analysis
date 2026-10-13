"""PCA on the multi-view topological feature matrix.

Even with ~14,000 columns the PCA is cheap: with n << p the SVD of the
``(n, p)`` data matrix costs O(n^2 p) and at most ``n_train - 1`` components
exist. Reducing to a few components (``k << n_train``) also makes an
*unregularised* logistic regression identifiable, so genuine Wald z-scores
and p-values can be computed per component (:func:`pca_logistic_with_pvalues`),
which is impossible on the raw features (p >> n).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .classification import build_multiview_feature_matrix, eval_predictions, make_logistic_regression, multiview_feature_names


def fit_pca_multiview(tda_by_view: dict, views: list, n_components=0.95, random_state: int = 0, verbose: bool = True) -> dict:
    """Standardise (fit on train) and fit a PCA on the multi-view features.

    ``n_components`` is either a fraction of variance to explain (float) or a
    fixed number of components (int).
    """
    X_train, groups = build_multiview_feature_matrix(tda_by_view, views, "train")
    X_test, _ = build_multiview_feature_matrix(tda_by_view, views, "test")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    Z_train = pca.fit_transform(X_train_s)
    Z_test = pca.transform(X_test_s)

    if verbose:
        print(
            f"PCA: {pca.n_components_} components explain {pca.explained_variance_ratio_.sum():.1%} of the variance "
            f"({X_train.shape[1]} features, {X_train.shape[0]} training subjects)."
        )
    return {"pca": pca, "scaler": scaler, "feature_groups": groups, "Z_train": Z_train, "Z_test": Z_test, "views": views}


def explained_variance_table(pca_result: dict) -> pd.DataFrame:
    ev = pca_result["pca"].explained_variance_ratio_
    return pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(ev))], "explained_variance_ratio": ev, "cumulative_variance": np.cumsum(ev)})


def fit_eval_pca_logistic(pca_result: dict, y_train, y_test, C: float = 1.0, penalty: str = "l2", verbose: bool = True, name: str = "PCA + logistic regression") -> dict:
    """Logistic regression on the principal components (mild regularisation
    is enough once the dimension is reduced)."""
    model = make_logistic_regression(C=C, penalty=penalty, solver="lbfgs" if penalty == "l2" else "liblinear")
    model.fit(pca_result["Z_train"], y_train)
    return {
        "model": model,
        "metrics_train": eval_predictions(y_train, model.predict(pca_result["Z_train"]), name=name, split="train", verbose=verbose),
        "metrics_test": eval_predictions(y_test, model.predict(pca_result["Z_test"]), name=name, split="test", verbose=verbose),
    }


def pca_logistic_with_pvalues(pca_result: dict, y_train, max_components=None) -> pd.DataFrame:
    """Unregularised ``statsmodels.Logit`` on the first ``max_components``
    PCs, returning coefficient, standard error, z and Wald p-value per
    component (sorted by p-value).

    Raises ``RuntimeError`` on perfect separation or non-convergence (too
    many components relative to the number of subjects) instead of returning
    meaningless numbers.
    """
    import statsmodels.api as sm
    from statsmodels.tools import sm_exceptions

    separation = tuple(
        t for t in (getattr(sm_exceptions, "PerfectSeparationWarning", None), getattr(sm_exceptions, "PerfectSeparationError", None)) if t
    )

    Z = pca_result["Z_train"]
    k = Z.shape[1] if max_components is None else min(int(max_components), Z.shape[1])
    model = sm.Logit(np.asarray(y_train), sm.add_constant(Z[:, :k]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in separation:
            if issubclass(t, Warning):
                warnings.simplefilter("error", t)
        try:
            result = model.fit(disp=0, maxiter=200)
        except separation as exc:
            raise RuntimeError(f"Perfect separation with {k} components: reduce the number of components.") from exc

    if not result.mle_retvals.get("converged", True):
        raise RuntimeError(f"Unregularised logistic regression with {k} components did not converge.")

    df = pd.DataFrame(
        {"component": ["const"] + [f"PC{i + 1}" for i in range(k)], "coef": result.params, "std_err": result.bse, "z": result.tvalues, "p_value": result.pvalues}
    )
    return df.sort_values("p_value").reset_index(drop=True)


def top_loadings_for_component(pca_result: dict, component_idx: int, top_n: int = 15) -> pd.DataFrame:
    """Original topological features with the largest |loading| on PC
    ``component_idx`` (0-based). The loading sign gives the direction of the
    correlation with the component, not the favoured class."""
    loadings = pca_result["pca"].components_[component_idx]
    df = pd.DataFrame({"feature": multiview_feature_names(pca_result["views"]), "loading": loadings})
    df["view"] = df["feature"].str.split("_", n=1).str[0]
    df["abs_loading"] = df["loading"].abs()
    return df.sort_values("abs_loading", ascending=False).head(top_n).reset_index(drop=True)
