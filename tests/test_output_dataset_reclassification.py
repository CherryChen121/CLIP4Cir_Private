import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from model_output_migration import SourceChangedError
from output_dataset_reclassification import (
    ReclassificationBlockedError,
    TransactionError,
    apply_reclassification,
    build_reclassification_plan,
    classify_run,
    find_output_writer_pids,
    verify_reclassification,
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


AUDIT_FIELDS = (
    "old_path",
    "new_path",
    "dataset",
    "stage",
    "model_slug",
    "run_id",
    "status",
    "size",
    "sha256",
    "duplicate_group",
    "canonical",
    "reason",
)


def write_audit_fixture(output_root: Path, runs) -> None:
    rows = []
    for run in runs:
        stage, model_slug, run_id = run.relative_to(
            output_root / "fashioniq"
        ).parts
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.name == "run_manifest.json":
                continue
            rows.append(
                {
                    "old_path": str(
                        output_root.parent
                        / "models"
                        / run_id
                        / path.relative_to(run)
                    ),
                    "new_path": str(path),
                    "dataset": "fashioniq",
                    "stage": stage,
                    "model_slug": model_slug,
                    "run_id": run_id,
                    "status": "moved",
                    "size": str(path.stat().st_size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "duplicate_group": "",
                    "canonical": "",
                    "reason": "",
                }
            )
    with (output_root / "migration_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "migration_report.json").write_text(
        json.dumps(
            {
                "valid_runs": len(runs),
                "failed_runs": 0,
                "status_counts": {"moved": len(rows)},
            }
        )
        + "\n",
        encoding="utf-8",
    )


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


def test_apply_blocks_active_training_writer(tmp_path, monkeypatch):
    run = make_run(
        tmp_path,
        "combiner",
        "active",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [run])
    plan = build_reclassification_plan(tmp_path)
    monkeypatch.setattr(
        "output_dataset_reclassification.find_output_writer_pids",
        lambda *_args, **_kwargs: (1234,),
    )

    with pytest.raises(ReclassificationBlockedError, match="1234"):
        apply_reclassification(plan)

    assert run.exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()


def test_apply_blocks_file_changed_after_snapshot(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "changed",
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [run])
    plan = build_reclassification_plan(tmp_path)
    (run / "train_metrics.csv").write_text(
        "epoch,loss\n1,9.9\n", encoding="utf-8"
    )

    with pytest.raises(SourceChangedError, match="train_metrics.csv"):
        apply_reclassification(plan)

    assert run.exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()


def test_apply_blocks_missing_audit_before_moving_runs(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "missing-audit",
        validation_headers=("IDRiD_recall_at1",),
    )
    plan = build_reclassification_plan(tmp_path)

    with pytest.raises(ReclassificationBlockedError, match="audit file"):
        apply_reclassification(plan)

    assert run.exists()
    assert not (tmp_path / "idrid").exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()


def test_apply_moves_run_and_updates_run_and_top_level_audits(tmp_path):
    run = make_run(
        tmp_path,
        "combiner",
        "idrid",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [run])
    plan = build_reclassification_plan(tmp_path)

    result = apply_reclassification(plan)

    target = tmp_path / "idrid/combiner/vitb32/idrid"
    assert result.ok
    assert not run.exists()
    assert target.is_dir()
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"] == "idrid"
    assert manifest["dataset_slug"] == "idrid"
    assert manifest["dataset_format"] == "fashioniq"
    assert manifest["dataset_classification_evidence"]
    rows = list(
        csv.DictReader(
            (tmp_path / "migration_manifest.csv").open(encoding="utf-8")
        )
    )
    assert {row["dataset"] for row in rows} == {"idrid"}
    assert all("/idrid/combiner/vitb32/idrid/" in row["new_path"] for row in rows)
    report = json.loads(
        (tmp_path / "migration_report.json").read_text(encoding="utf-8")
    )
    assert report["actual_dataset_run_counts"] == {"idrid": 1}
    assert report["reclassification_state"] == "applied"


def test_apply_rolls_back_all_runs_when_second_final_move_fails(
    tmp_path, monkeypatch
):
    first = make_run(
        tmp_path,
        "clip-finetune",
        "first",
        validation_headers=("IDRiD_recall_at1",),
    )
    second = make_run(
        tmp_path,
        "clip-finetune",
        "second",
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [first, second])
    plan = build_reclassification_plan(tmp_path)
    failing_action = plan.actions[1]
    real_replace = Path.replace

    def fail_second_final_move(path, target):
        if path == failing_action.staging and target == failing_action.destination:
            raise OSError("injected move failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_final_move)

    with pytest.raises(TransactionError, match="rolled back"):
        apply_reclassification(plan)

    assert first.exists()
    assert second.exists()
    assert not (tmp_path / "idrid").exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()


def test_apply_restores_runs_and_metadata_when_audit_replace_fails(
    tmp_path, monkeypatch
):
    run = make_run(
        tmp_path,
        "combiner",
        "audit-failure",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [run])
    original_manifest = (run / "run_manifest.json").read_bytes()
    original_audit = (tmp_path / "migration_manifest.csv").read_bytes()
    plan = build_reclassification_plan(tmp_path)
    real_replace_file = (
        __import__("output_dataset_reclassification")._replace_file
    )
    failure_injected = False

    def fail_manifest_audit(source, destination):
        nonlocal failure_injected
        if (
            destination.name == "migration_manifest.csv"
            and not failure_injected
        ):
            failure_injected = True
            raise OSError("injected audit failure")
        return real_replace_file(source, destination)

    monkeypatch.setattr(
        "output_dataset_reclassification._replace_file",
        fail_manifest_audit,
    )

    with pytest.raises(TransactionError, match="rolled back"):
        apply_reclassification(plan)

    assert run.exists()
    assert (run / "run_manifest.json").read_bytes() == original_manifest
    assert (tmp_path / "migration_manifest.csv").read_bytes() == original_audit
    assert not (tmp_path / "idrid").exists()


def test_verify_reclassification_detects_destination_hash_change(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "verify",
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, [run])
    plan = build_reclassification_plan(tmp_path)
    apply_reclassification(plan)
    checkpoint = (
        tmp_path
        / "idrid/clip-finetune/vitb32/verify/checkpoints/model.pt"
    )
    checkpoint.write_bytes(b"xxxxxx")

    result = verify_reclassification(tmp_path, expected_plan=plan)

    assert not result.ok
    assert any("sha256 mismatch" in error for error in result.errors)


def test_find_output_writer_pids_filters_cirr_and_other_output_roots(tmp_path):
    project = tmp_path / "project"
    output = project / "outputs"
    other_output = project / "other-outputs"
    (project / "src").mkdir(parents=True)
    (project / "src/clip_fine_tune.py").write_text("", encoding="utf-8")
    proc = tmp_path / "proc"
    for pid, arguments in (
        (
            101,
            [
                "python",
                "src/clip_fine_tune.py",
                "--dataset",
                "FashionIQ",
            ],
        ),
        (
            102,
            [
                "python",
                "src/clip_fine_tune.py",
                "--dataset",
                "CIRR",
            ],
        ),
        (
            103,
            [
                "python",
                "src/clip_fine_tune.py",
                "--dataset=FashionIQ",
                f"--output-root={other_output}",
            ],
        ),
    ):
        process = proc / str(pid)
        process.mkdir(parents=True)
        (process / "cmdline").write_bytes(
            b"\0".join(os.fsencode(value) for value in arguments) + b"\0"
        )
        (process / "cwd").symlink_to(project, target_is_directory=True)

    assert find_output_writer_pids(project, output, proc_root=proc) == (101,)
