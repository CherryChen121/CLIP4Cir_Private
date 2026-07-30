import csv
import json
from pathlib import Path

import pytest

from output_dataset_reclassification import (
    ReclassificationBlockedError,
    build_reclassification_plan,
    classify_run,
)


UWF_METRIC_HEADERS = tuple(
    f"{category}_recall_at1"
    for category in ("CH", "CO", "NM", "RB", "RCH", "UM")
)


def make_run(
    output_root: Path,
    stage: str,
    run_id: str,
    *,
    model_slug: str = "vitb32",
    hyperparameters=None,
    validation_headers=(),
) -> Path:
    run = output_root / "fashioniq" / stage / model_slug / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/model.pt").write_bytes(run_id.encode())
    (run / "train_metrics.csv").write_text(
        "epoch,loss\n1,0.1\n", encoding="utf-8"
    )
    with (run / "validation_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", *validation_headers))
        writer.writerow((1, *(0.5 for _ in validation_headers)))
    if hyperparameters is not None:
        (run / "training_hyperparameters.json").write_text(
            json.dumps(hyperparameters), encoding="utf-8"
        )
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "fashioniq",
                "dataset_slug": "fashioniq",
                "training_stage": stage,
                "model_slug": model_slug,
                "run_id": run_id,
            }
        ),
        encoding="utf-8",
    )
    return run


@pytest.mark.parametrize(
    ("hyperparameters", "headers", "expected"),
    [
        (
            {"train_dress_types": ["IDRiD"]},
            ("IDRiD_recall_at1",),
            "idrid",
        ),
        (
            {
                "train_dress_types": [
                    "CH",
                    "CO",
                    "NM",
                    "RB",
                    "RCH",
                    "UM",
                ]
            },
            UWF_METRIC_HEADERS,
            "uwf",
        ),
        (
            {
                "train_dress_types": ["Internal"],
                "fashioniq_root": (
                    "/data0/qrchen/datasets/"
                    "Combined_Fundus_CIR_Dataset"
                ),
            },
            ("Internal_recall_at1",),
            "combined-fundus-cir",
        ),
        (
            {"train_dress_types": ["dress", "shirt", "toptee"]},
            ("dress_recall_at1", "shirt_recall_at1", "toptee_recall_at1"),
            "fashioniq",
        ),
    ],
)
def test_classifies_combiner_from_hyperparameters_and_metrics(
    tmp_path, hyperparameters, headers, expected
):
    run = make_run(
        tmp_path,
        "combiner",
        expected,
        hyperparameters=hyperparameters,
        validation_headers=headers,
    )

    result = classify_run(run)

    assert result.dataset_slug == expected
    assert result.dataset_format == "fashioniq"
    assert any(
        evidence.startswith("training_hyperparameters:train_dress_types=")
        for evidence in result.evidence
    )


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (("IDRiD_recall_at1",), "idrid"),
        (UWF_METRIC_HEADERS, "uwf"),
        (("Internal_recall_at1",), "combined-fundus-cir"),
        (
            ("dress_recall_at1", "shirt_recall_at1", "toptee_recall_at1"),
            "fashioniq",
        ),
    ],
)
def test_classifies_clip_finetune_from_validation_headers(
    tmp_path, headers, expected
):
    run = make_run(
        tmp_path,
        "clip-finetune",
        expected,
        validation_headers=headers,
    )

    result = classify_run(run)

    assert result.dataset_slug == expected
    assert any(
        evidence.startswith("validation_metrics:")
        for evidence in result.evidence
    )


def test_classification_rejects_conflicting_evidence(tmp_path):
    run = make_run(
        tmp_path,
        "combiner",
        "conflict",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=UWF_METRIC_HEADERS,
    )

    with pytest.raises(ReclassificationBlockedError, match="conflicting"):
        classify_run(run)


def test_plan_blocks_unknown_evidence(tmp_path):
    make_run(
        tmp_path,
        "clip-finetune",
        "unknown",
        validation_headers=("loss",),
    )

    with pytest.raises(ReclassificationBlockedError, match="unresolved"):
        build_reclassification_plan(tmp_path)


def test_plan_builds_exact_destination_and_snapshots(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "idrid-run",
        validation_headers=("IDRiD_recall_at1",),
    )

    plan = build_reclassification_plan(tmp_path)

    assert plan.dataset_counts == {"idrid": 1}
    assert len(plan.actions) == 1
    action = plan.actions[0]
    assert action.source == run
    assert action.destination == (
        tmp_path / "idrid/clip-finetune/vitb32/idrid-run"
    )
    assert action.staging == (
        tmp_path
        / ".dataset-reclassify-staging/"
        "idrid/clip-finetune/vitb32/idrid-run"
    )
    assert {snapshot.relative_path.as_posix() for snapshot in action.files} == {
        "checkpoints/model.pt",
        "run_manifest.json",
        "train_metrics.csv",
        "validation_metrics.csv",
    }
    assert all(len(snapshot.sha256) == 64 for snapshot in action.files)


def test_plan_blocks_target_collision(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "idrid-run",
        validation_headers=("IDRiD_recall_at1",),
    )
    target = tmp_path / "idrid/clip-finetune/vitb32" / run.name
    target.mkdir(parents=True)

    with pytest.raises(ReclassificationBlockedError, match="collision"):
        build_reclassification_plan(tmp_path)


def test_plan_rejects_symlinks_inside_run(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "linked",
        validation_headers=("IDRiD_recall_at1",),
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "outside-link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ReclassificationBlockedError, match="symlink"):
        build_reclassification_plan(tmp_path)


def test_plan_rejects_cross_filesystem_action(tmp_path, monkeypatch):
    make_run(
        tmp_path,
        "clip-finetune",
        "cross-device",
        validation_headers=("IDRiD_recall_at1",),
    )
    monkeypatch.setattr(
        "output_dataset_reclassification._device_id",
        lambda path: 2 if "fashioniq" in path.parts else 1,
    )

    with pytest.raises(ReclassificationBlockedError, match="filesystem"):
        build_reclassification_plan(tmp_path)


def test_dry_plan_does_not_modify_files(tmp_path):
    make_run(
        tmp_path,
        "clip-finetune",
        "stable",
        validation_headers=("IDRiD_recall_at1",),
    )
    before = sorted(
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")
    )

    plan = build_reclassification_plan(tmp_path)

    after = sorted(
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")
    )
    assert len(plan.actions) == 1
    assert before == after


def test_missing_fashioniq_source_returns_empty_plan(tmp_path):
    plan = build_reclassification_plan(tmp_path)

    assert plan.actions == ()
    assert plan.dataset_counts == {}
