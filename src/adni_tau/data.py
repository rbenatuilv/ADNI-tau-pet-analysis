"""Indexing and loading of the 2D tau-PET slices.

Expected folder layout (see ``data/README.md``)::

    data/tau_data/
        tau_data_axial/{AD,CN,MCI}/<subject_id>_slice<k>.png
        tau_data_coronal/{AD,CN,MCI}/...
        tau_data_sagittal/{AD,CN,MCI}/...

Each subject contributes exactly one slice per view, so within a view one
image corresponds to one subject.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .paths import TAU_DATA_DIR


class ImageLoader:
    """Index of the PNG slices with lazy loading and simple selection helpers.

    Parameters
    ----------
    base_path
        Folder containing the ``tau_data_<view>`` sub-folders.
    views, classes
        Subsets of ``VIEWS`` / ``CLASSES`` to index (default: all).
    image_size
        Optional ``(width, height)`` to resize every image to.
    normalize
        Scale intensities from ``[0, 255]`` to ``[0, 1]``.
    """

    VIEWS = ["sagittal", "coronal", "axial"]
    CLASSES = ["AD", "CN", "MCI"]

    def __init__(
        self,
        base_path: str | Path = TAU_DATA_DIR,
        views: Optional[list[str]] = None,
        classes: Optional[list[str]] = None,
        image_size: Optional[tuple[int, int]] = None,
        normalize: bool = True,
        dtype=np.float32,
    ):
        self.base_path = Path(base_path)
        self.views = list(views) if views is not None else list(self.VIEWS)
        self.classes = list(classes) if classes is not None else list(self.CLASSES)
        self.image_size = image_size
        self.normalize = normalize
        self.dtype = dtype

        self._validate_inputs()
        self.class_to_label = {name: idx for idx, name in enumerate(self.classes)}
        self.label_to_class = {idx: name for idx, name in enumerate(self.classes)}
        self.samples = self._build_index()

    # ------------------------------------------------------------------ setup
    def _validate_inputs(self) -> None:
        if not self.base_path.exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.base_path}. See data/README.md for how to obtain the data."
            )
        invalid_views = [v for v in self.views if v not in self.VIEWS]
        if invalid_views:
            raise ValueError(f"Invalid views: {invalid_views}. Expected a subset of {self.VIEWS}.")
        invalid_classes = [c for c in self.classes if c not in self.CLASSES]
        if invalid_classes:
            raise ValueError(f"Invalid classes: {invalid_classes}. Expected a subset of {self.CLASSES}.")

    def _build_index(self) -> list[dict]:
        samples = []
        for view in self.views:
            for cls in self.classes:
                folder = self.base_path / f"tau_data_{view}" / cls
                for image_path in sorted(folder.glob("*.png")):
                    stem = image_path.stem
                    samples.append(
                        {
                            "path": image_path,
                            "subject_id": stem.split("_slice")[0],
                            "view": view,
                            "class_name": cls,
                            "label": self.class_to_label[cls],
                        }
                    )
        if not samples:
            raise ValueError(f"No PNG images found under {self.base_path}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    # ---------------------------------------------------------------- loading
    def load_image(self, idx: int) -> np.ndarray:
        """Load sample ``idx`` as a 2D grayscale array."""
        with Image.open(self.samples[idx]["path"]) as img:
            img = img.convert("L")
            if self.image_size is not None:
                img = img.resize(self.image_size)
            arr = np.asarray(img, dtype=self.dtype)
        return arr / 255.0 if self.normalize else arr

    def get_sample_info(self, idx: int) -> dict:
        return dict(self.samples[idx])

    # -------------------------------------------------------------- selection
    def get_indices(self, views: Optional[list[str]] = None, classes: Optional[list[str]] = None) -> np.ndarray:
        """Indices of the samples belonging to the given views and classes."""
        views = set(views) if views is not None else set(self.views)
        classes = set(classes) if classes is not None else set(self.classes)
        idx = [i for i, s in enumerate(self.samples) if s["view"] in views and s["class_name"] in classes]
        return np.array(idx, dtype=np.int64)

    def get_shape(self, view: str) -> tuple[int, int]:
        """Shape ``(H, W)`` of the images of ``view``."""
        idx = self.get_indices(views=[view])
        if len(idx) == 0:
            raise ValueError(f"No samples for view '{view}'.")
        return self.load_image(int(idx[0])).shape

    def to_numpy_array(self, indices) -> np.ndarray:
        """Stack the images at ``indices`` as a flattened ``(n, H*W)`` matrix."""
        flat = [self.load_image(int(i)).reshape(-1) for i in indices]
        sizes = {f.size for f in flat}
        if len(sizes) > 1:
            raise ValueError("Images have different sizes: set image_size or select a single view.")
        return np.stack(flat, axis=0)

    def load_view(self, view: str, classes: Optional[list[str]] = None) -> dict:
        """Load every image of one view.

        Returns a dict with the flattened matrix ``X`` (n, H*W), integer
        ``labels``, string ``classes``, ``subject_ids``, the image ``shape``
        and the dataset ``indices``.
        """
        indices = self.get_indices(views=[view], classes=classes)
        return {
            "X": self.to_numpy_array(indices),
            "labels": np.array([self.samples[i]["label"] for i in indices]),
            "classes": np.array([self.samples[i]["class_name"] for i in indices]),
            "subject_ids": np.array([self.samples[i]["subject_id"] for i in indices]),
            "shape": self.get_shape(view),
            "indices": indices,
        }


def split_subjects(subject_ids, classes, test_size: float = 0.3, random_state: int = 42) -> np.ndarray:
    """Stratified subject-level train/test split.

    Returns a boolean array, ``True`` for test subjects. Because every subject
    has one slice per view, applying the same subject split to each view
    keeps all views of a subject on the same side of the split.
    """
    subject_ids, classes = np.asarray(subject_ids), np.asarray(classes)
    _, test_ids = train_test_split(subject_ids, test_size=test_size, stratify=classes, random_state=random_state)
    return np.isin(subject_ids, test_ids)
