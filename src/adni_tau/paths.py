"""Project directories and helpers to save figures and tables to ``results/``."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _find_project_root() -> Path:
    """Walk up from this file until the folder containing ``pyproject.toml``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not locate the project root (no pyproject.toml found).")


PROJECT_ROOT = _find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
TAU_DATA_DIR = DATA_DIR / "tau_data"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Save ``fig`` as ``results/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Save ``df`` as ``results/tables/<name>.csv`` and return the path."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.4f")
    return path
