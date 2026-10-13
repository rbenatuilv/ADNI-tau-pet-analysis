"""Model selection and evaluation helpers shared by the classification notebooks."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def classification_metrics(y_true, y_pred, y_score=None, positive_label=None) -> dict:
    """Accuracy, balanced accuracy, macro-F1 and (if scores are given) ROC-AUC.

    For binary problems ``y_score`` is the score of ``positive_label``; for
    multi-class problems it is the ``(n, n_classes)`` probability matrix and a
    macro one-vs-rest AUC is reported.
    """
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if y_score.ndim == 1:
                out["roc_auc"] = roc_auc_score(np.asarray(y_true) == positive_label, y_score)
            else:
                out["roc_auc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except ValueError:
            out["roc_auc"] = np.nan
    return out


def _positive_scores(model, X, positive_label):
    """Score of ``positive_label`` from ``predict_proba`` or ``decision_function``."""
    classes = list(model.classes_)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, classes.index(positive_label)]
    scores = model.decision_function(X)
    return scores if classes.index(positive_label) == 1 else -scores


def fit_evaluate_binary(
    X_train,
    y_train,
    X_test,
    y_test,
    pipeline: Pipeline,
    param_grid: dict,
    positive_label: str = "AD",
    scoring: str = "balanced_accuracy",
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = 4,
) -> dict:
    """Grid-search ``pipeline`` with stratified K-fold CV on the training set,
    refit on all training data and evaluate once on the held-out test set.

    Returns a dict with the fitted ``model``, ``best_params``, the best
    cross-validated score (``cv_score``), test-set ``metrics`` and the test
    confusion matrix (``cm``, rows = true class, columns = predicted class).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    search.fit(X_train, y_train)

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    scores = _positive_scores(model, X_test, positive_label)
    labels = sorted(np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)])))

    return {
        "model": model,
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "metrics": classification_metrics(y_test, y_pred, scores, positive_label=positive_label),
        "cm": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
    }


def results_table(rows: list[dict]) -> pd.DataFrame:
    """Flatten ``{"...": ..., "metrics": {...}}`` records into a tidy DataFrame."""
    flat = []
    for r in rows:
        base = {k: v for k, v in r.items() if k != "metrics"}
        flat.append({**base, **r.get("metrics", {})})
    return pd.DataFrame(flat)


def plot_confusion_matrices(
    results: dict[str, dict],
    title: Optional[str] = None,
    normalize: bool = False,
    ncols: Optional[int] = None,
) -> plt.Figure:
    """Side-by-side confusion matrices; ``results`` maps a panel title to a
    result of :func:`fit_evaluate_binary` (needs ``cm`` and ``labels``)."""
    n = len(results)
    ncols = ncols or n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)

    for ax, (name, res) in zip(axes.ravel(), results.items()):
        cm = res["cm"]
        if normalize:
            cm = cm / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(cm, display_labels=res["labels"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
        ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
