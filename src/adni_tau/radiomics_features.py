"""2D radiomics features (pyradiomics) computed inside a binary mask."""

import logging
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm.auto import tqdm

DEFAULT_SETTINGS = {
    "force2D": True,
    "force2Ddimension": 0,
    "normalize": True,
    "normalizeScale": 100,
    "binWidth": 5,
    "label": 1,
    "resampledPixelSpacing": None,
    "voxelArrayShift": 0,
}

# Texture and intensity families; shape features are optional because with a
# fixed (group-level) mask they are identical for every subject.
DEFAULT_FEATURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm")


def silence_pyradiomics() -> None:
    """Hide pyradiomics' per-image log messages and a harmless GLCM warning."""
    warnings.filterwarnings("ignore", message=r".*GLCM is symmetrical.*")
    logging.getLogger("radiomics").setLevel(logging.ERROR)


class RadiomicsExtractor:
    """Thin wrapper around ``RadiomicsFeatureExtractor`` for 2D numpy images."""

    def __init__(
        self,
        settings: Optional[dict] = None,
        feature_classes: Iterable[str] = DEFAULT_FEATURE_CLASSES,
        include_shape: bool = False,
    ):
        params = {**DEFAULT_SETTINGS, **(settings or {})}
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**params)
        self.extractor.disableAllFeatures()
        for name in feature_classes:
            self.extractor.enableFeatureClassByName(name)
        if include_shape:
            self.extractor.enableFeatureClassByName("shape2D")
        silence_pyradiomics()

    @staticmethod
    def _to_sitk(image: np.ndarray, mask: np.ndarray) -> tuple[sitk.Image, sitk.Image]:
        image_sitk = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
        image_sitk.SetSpacing((1.0, 1.0))
        mask_sitk = sitk.GetImageFromArray(np.asarray(mask, dtype=np.uint8))
        mask_sitk.CopyInformation(image_sitk)
        return image_sitk, mask_sitk

    def extract(self, image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Feature name -> value for one image inside ``mask`` (same shape)."""
        if mask.shape != image.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}.")
        result = self.extractor.execute(*self._to_sitk(image, mask))
        return {k: float(v) for k, v in result.items() if not k.startswith("diagnostics_")}

    def extract_many(self, images: Iterable[np.ndarray], mask: np.ndarray, desc: str = "radiomics") -> pd.DataFrame:
        """One row of features per image, all computed inside the same ``mask``."""
        images = list(images)
        rows = [self.extract(img, mask) for img in tqdm(images, desc=desc, leave=False)]
        return pd.DataFrame(rows)


def clean_feature_table(df: pd.DataFrame, max_nan_ratio: float = 0.3, reference: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Drop mostly-missing or constant columns and fill the remaining NaNs.

    If ``reference`` (e.g. the training table) is given, the columns kept and
    the fill values are taken from it, so the same transformation can be
    applied to validation/test tables without leaking information.
    """
    ref = df if reference is None else reference
    keep = [c for c in ref.columns if ref[c].isna().mean() <= max_nan_ratio and ref[c].nunique(dropna=True) > 1]
    medians = ref[keep].median()
    return df[keep].fillna(medians)
