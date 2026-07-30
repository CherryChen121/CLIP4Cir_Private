import json
from datetime import datetime

import pytest

from output_paths import (
    build_run_id,
    create_run_layout,
    resolve_output_root,
    slugify_component,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FashionIQ", "fashioniq"),
        ("ViT-B/32", "vit-b-32"),
        ("  BLIP  ITM_Base  ", "blip-itm-base"),
        ("RN50x4", "rn50x4"),
    ],
)
def test_slugify_component(raw, expected):
    assert slugify_component(raw) == expected


def test_slugify_component_rejects_empty_result():
    with pytest.raises(ValueError, match="empty slug"):
        slugify_component("///")


def test_relative_output_root_is_resolved_from_project(tmp_path):
    assert resolve_output_root(tmp_path, "artifacts") == tmp_path / "artifacts"
    assert resolve_output_root(tmp_path, None) == tmp_path / "outputs"


def test_run_id_is_sortable_and_contains_pid():
    started = datetime(2026, 7, 30, 9, 50, 58, 123456)
    assert build_run_id(started, 2384293) == "20260730-095058-123456-p2384293"


def test_create_run_layout_writes_manifest_and_sanitizes_model(tmp_path):
    layout = create_run_layout(
        project_root=tmp_path,
        output_root=None,
        dataset="FashionIQ",
        stage="combiner",
        model_name="ViT-B/32",
        started_at=datetime(2026, 7, 30, 9, 50, 58, 123456),
        pid=2384293,
    )
    assert layout.root == (
        tmp_path
        / "outputs/fashioniq/combiner/vit-b-32/"
        "20260730-095058-123456-p2384293"
    )
    assert layout.checkpoints == layout.root / "checkpoints"
    payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert payload["model_name"] == "ViT-B/32"
    assert payload["model_slug"] == "vit-b-32"
    assert payload["checkpoint_dir"] == "checkpoints"


def test_create_run_layout_never_reuses_existing_run(tmp_path):
    kwargs = {
        "project_root": tmp_path,
        "output_root": None,
        "dataset": "CIRR",
        "stage": "clip-finetune",
        "model_name": "RN50x4",
        "started_at": datetime(2026, 7, 30, 9, 50, 58, 123456),
        "pid": 123,
    }
    create_run_layout(**kwargs)
    with pytest.raises(FileExistsError):
        create_run_layout(**kwargs)


def test_create_run_layout_rejects_unknown_stage(tmp_path):
    with pytest.raises(ValueError, match="training stage"):
        create_run_layout(
            project_root=tmp_path,
            output_root=None,
            dataset="FashionIQ",
            stage="unknown",
            model_name="RN50x4",
        )
