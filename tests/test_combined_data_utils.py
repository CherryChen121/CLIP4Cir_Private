import json
from pathlib import Path

import pytest
from PIL import Image

from data_utils import (
    FashionIQDataset,
    list_fashioniq_categories,
    resolve_fashioniq_root,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_layout(root: Path) -> None:
    for directory in ("captions", "image_splits", "images"):
        (root / directory).mkdir(parents=True, exist_ok=True)


def _add_category(root: Path, category: str, split: str, triplets=None, image_names=None) -> None:
    _create_layout(root)
    _write_json(
        root / "captions" / f"cap.{category}.{split}.json",
        [] if triplets is None else triplets,
    )
    _write_json(
        root / "image_splits" / f"split.{category}.{split}.json",
        [] if image_names is None else image_names,
    )


def _add_image(root: Path, name: str) -> None:
    Image.new("RGB", (4, 4), color=(20, 40, 60)).save(root / "images" / f"{name}.png")


def test_explicit_root_resolution_ignores_environment_roots(tmp_path, monkeypatch):
    explicit_root = tmp_path / "combined"
    legacy_root = tmp_path / "legacy"
    _add_category(explicit_root, "Internal", "train")
    _add_category(legacy_root, "IDRiD", "train")
    monkeypatch.setenv("CLIP4CIR_FASHIONIQ_ROOT", str(legacy_root))

    resolved = resolve_fashioniq_root("Internal", "train", explicit_root)

    assert resolved == explicit_root.resolve()


def test_explicit_category_listing_is_isolated_and_sorted(tmp_path, monkeypatch):
    explicit_root = tmp_path / "combined"
    legacy_root = tmp_path / "legacy"
    for category in ("ODIR5K", "Internal", "GRAPE"):
        _add_category(explicit_root, category, "test")
    _add_category(legacy_root, "dress", "test")
    monkeypatch.setenv("CLIP4CIR_FASHIONIQ_ROOT", str(legacy_root))

    categories = list_fashioniq_categories("test", explicit_root)

    assert categories == ["GRAPE", "Internal", "ODIR5K"]


def test_labeled_test_returns_names_and_keeps_legacy_test_tuple(tmp_path):
    root = tmp_path / "combined"
    triplets = [{"candidate": "reference", "target": "target", "captions": ["change"]}]
    _add_category(root, "Internal", "test", triplets, ["reference", "target"])
    _add_image(root, "reference")
    _add_image(root, "target")
    preprocess = lambda image: image.copy()

    legacy_dataset = FashionIQDataset(
        "test", ["Internal"], "relative", preprocess, dataset_root=root
    )
    labeled_dataset = FashionIQDataset(
        "test",
        ["Internal"],
        "relative",
        preprocess,
        dataset_root=root,
        return_target=True,
    )

    legacy_item = legacy_dataset[0]
    assert legacy_item[0] == "reference"
    assert isinstance(legacy_item[1], Image.Image)
    assert legacy_item[2] == ["change"]
    assert labeled_dataset[0] == ("reference", "target", ["change"])


def test_labeled_test_rejects_missing_target_key(tmp_path):
    root = tmp_path / "combined"
    triplets = [{"candidate": "reference", "captions": ["change"]}]
    _add_category(root, "Internal", "test", triplets, ["reference"])
    _add_image(root, "reference")

    with pytest.raises(ValueError, match=r"Internal.*test.*target"):
        FashionIQDataset(
            "test",
            ["Internal"],
            "relative",
            lambda image: image,
            dataset_root=root,
            return_target=True,
        )


def test_labeled_test_rejects_target_missing_from_gallery(tmp_path):
    root = tmp_path / "combined"
    triplets = [{"candidate": "reference", "target": "target", "captions": ["change"]}]
    _add_category(root, "Internal", "test", triplets, ["reference"])
    _add_image(root, "reference")

    with pytest.raises(ValueError, match=r"Internal.*test.*target"):
        FashionIQDataset(
            "test",
            ["Internal"],
            "relative",
            lambda image: image,
            dataset_root=root,
            return_target=True,
        )
