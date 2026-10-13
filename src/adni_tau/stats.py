"""Group-difference tests on principal-component scores."""

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests


def select_pcs_by_group_test(
    scores: np.ndarray,
    labels: np.ndarray,
    pc_names: list[str] | None = None,
    method: str = "welch",
    alpha: float = 0.05,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """One group-difference test per PC score across diagnostic classes.

    Parameters
    ----------
    scores
        ``(n_subjects, n_pcs)`` PCA scores.
    labels
        ``(n_subjects,)`` group labels (e.g. CN / MCI / AD).
    method
        ``"anova"`` (classical one-way ANOVA) or ``"welch"`` (Welch ANOVA,
        robust to unequal group variances; the default).
    alpha, correction
        Significance level and multiple-testing correction passed to
        ``statsmodels.stats.multitest.multipletests`` (e.g. ``"fdr_bh"``,
        ``"bonferroni"``).

    Returns
    -------
    DataFrame with one row per PC (``pc``, ``test``, ``F``, ``p_value``,
    ``effect_size`` = partial eta squared, ``p_adj``, ``reject_null``), sorted
    by adjusted p-value and then by F.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if scores.ndim != 2 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("Expected scores (n, k) and labels (n,) with matching n.")

    n_pcs = scores.shape[1]
    pc_names = pc_names or [f"PC{k + 1}" for k in range(n_pcs)]
    if len(pc_names) != n_pcs:
        raise ValueError("pc_names must have one entry per PC.")

    groups = np.unique(labels)
    if len(groups) < 2:
        raise ValueError("At least two groups are required.")

    method = method.lower()
    if method not in {"anova", "welch"}:
        raise ValueError("method must be 'anova' or 'welch'.")

    rows = []
    for k in range(n_pcs):
        y = scores[:, k]
        valid = np.isfinite(y)
        y, lab = y[valid], labels[valid]
        grouped = [y[lab == g] for g in groups]

        if min(len(g) for g in grouped) < 2:
            raise ValueError(f"{pc_names[k]}: every group needs at least two valid samples.")

        if method == "anova":
            F, p = f_oneway(*grouped)
            grand_mean = y.mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grouped)
            ss_total = np.sum((y - grand_mean) ** 2)
            effect = ss_between / ss_total if ss_total > 0 else np.nan
        else:
            res = pg.welch_anova(data=pd.DataFrame({"score": y, "label": lab}), dv="score", between="label")
            row = res.iloc[0]
            F = float(row["F"])
            p = float(row["p_unc"] if "p_unc" in res.columns else row["p-unc"])
            effect = float(row["np2"]) if "np2" in res.columns else np.nan

        rows.append({"pc": pc_names[k], "test": method, "F": float(F), "p_value": float(p), "effect_size": effect})

    df = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(df["p_value"].to_numpy(), alpha=alpha, method=correction)
    df["p_adj"] = p_adj
    df["reject_null"] = reject

    return df.sort_values(["p_adj", "F"], ascending=[True, False]).reset_index(drop=True)
