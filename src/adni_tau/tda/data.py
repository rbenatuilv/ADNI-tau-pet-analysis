"""Balanced AD/CN train/test split shared by all views.

The same subjects are used in every view, so the multi-view models can
concatenate per-view features subject by subject. Labels: ``0 = AD``,
``1 = CN``.
"""

import numpy as np

from ..data import ImageLoader
from . import config


def select_subjects(seed: int = config.SPLIT_SEED) -> dict:
    """Pick ``N_TRAIN_PER_CLASS + N_TEST_PER_CLASS`` subjects per class at
    random (subjects present in every view) and split them into train/test.

    Returns ``{"train": [...subject ids...], "test": [...]}`` with AD subjects
    first, then CN subjects, in each list.
    """
    loader = ImageLoader(views=config.VIEWS, classes=["AD", "CN"])
    needed = config.N_TRAIN_PER_CLASS + config.N_TEST_PER_CLASS
    rng = np.random.default_rng(seed)

    split = {"train": [], "test": []}
    for cls in ("AD", "CN"):
        per_view = [
            {loader.samples[i]["subject_id"] for i in loader.get_indices(views=[v], classes=[cls])}
            for v in config.VIEWS
        ]
        common = sorted(set.intersection(*per_view))
        if len(common) < needed:
            raise ValueError(f"Need {needed} {cls} subjects present in all views, found {len(common)}.")
        chosen = rng.permutation(common)[:needed]
        split["train"] += list(chosen[: config.N_TRAIN_PER_CLASS])
        split["test"] += list(chosen[config.N_TRAIN_PER_CLASS :])
    return split


def load_one_view(view: str, split: dict, verbose: bool = True) -> dict:
    """Images of ``view`` for the subjects in ``split`` (see
    :func:`select_subjects`), as ``(n, H, W)`` arrays plus labels."""
    W, H = config.VIEW_IMAGE_SIZE[view]
    loader = ImageLoader(views=[view], classes=["AD", "CN"], image_size=(W, H))
    index_by_subject = {s["subject_id"]: i for i, s in enumerate(loader.samples)}

    out = {"view": view}
    for part in ("train", "test"):
        idx = [index_by_subject[sid] for sid in split[part]]
        out[f"{part}_imgs"] = loader.to_numpy_array(idx).reshape(len(idx), H, W)
        out[f"y_{part}"] = np.array([0 if loader.samples[i]["class_name"] == "AD" else 1 for i in idx])

    if verbose:
        print(f"{view:>8}: train {out['train_imgs'].shape}, test {out['test_imgs'].shape}")
    return out


def load_all_views(views=None, seed: int = config.SPLIT_SEED, verbose: bool = True) -> dict:
    """Load every view for one common subject split; labels are checked to
    be identical across views."""
    views = views if views is not None else config.VIEWS
    split = select_subjects(seed)
    data_by_view = {view: load_one_view(view, split, verbose=verbose) for view in views}

    for view in views[1:]:
        for part in ("y_train", "y_test"):
            assert np.array_equal(data_by_view[views[0]][part], data_by_view[view][part])
    return data_by_view
