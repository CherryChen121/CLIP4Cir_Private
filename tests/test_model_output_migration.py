from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

import model_output_migration as migration
from model_output_migration import (
    MigrationBlockedError,
    SourceChangedError,
    VerificationResult,
    apply_migration,
    build_migration_plan,
    finalize_source,
    find_legacy_writer_pids,
    scan_legacy_outputs,
    sha256_file,
    verify_migration,
    write_migration_reports,
)


NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _checkpoint(path: Path, payload: bytes) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(
            "archive/data.pkl",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        archive.writestr(info, payload)
    return _write(path, buffer.getvalue())


def _age_tree(path: Path, age: timedelta) -> None:
    timestamp = (NOW - age).timestamp()
    for item in [path, *path.rglob("*")]:
        os.utime(item, (timestamp, timestamp))


def _scan(source: Path, output: Path, *, live_pids=(), writer_pids=()):
    return scan_legacy_outputs(
        source,
        output,
        now=NOW,
        pid_is_alive=lambda pid: pid in live_pids,
        legacy_writer_pids=tuple(writer_pids),
    )


def test_scanner_reconstructs_model_name_split_by_slash(tmp_path):
    source = tmp_path / "models"
    run = (
        source
        / "combiner_trained_on_fiq_ViT-B"
        / "32_2026-03-30_13:06:57_pid1462419"
    )
    _checkpoint(run / "saved_models/combiner.pt", b"weights")
    _write(run / "validation_metrics.csv", b"epoch,recall\n1,0.5\n")

    result = _scan(source, tmp_path / "outputs")

    record = result.runs[0]
    assert record.model_name == "ViT-B/32"
    assert record.model_slug == "vit-b-32"
    assert record.stage == "combiner"
    assert record.dataset_slug == "fashioniq"
    assert record.destination.parts[-4:-1] == (
        "fashioniq",
        "combiner",
        "vit-b-32",
    )
    assert record.classification == "valid"


def test_failed_run_requires_all_four_conditions(tmp_path):
    source = tmp_path / "models"
    run = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid7"
    _write(run / "training_hyperparameters.json", b"{}")
    (run / "saved_models").mkdir()
    _age_tree(run, timedelta(hours=25))

    result = _scan(source, tmp_path / "outputs")

    assert result.runs[0].classification == "failed"
    assert result.runs[0].reasons == (
        "no-nonempty-checkpoint",
        "no-nonempty-metrics",
        "pid-not-alive",
        "older-than-24-hours",
    )


def test_recent_empty_run_is_not_failed(tmp_path):
    source = tmp_path / "models"
    run = source / "combiner_trained_on_fiq_RN50x4_2026-07-30_11:00:00_pid8"
    _write(run / "training_hyperparameters.json", b"{}")
    (run / "saved_models").mkdir()

    result = _scan(source, tmp_path / "outputs")

    assert result.runs[0].classification == "unresolved"
    assert "modified-within-24-hours" in result.runs[0].reasons


def test_live_pid_and_nonempty_metrics_each_prevent_failure(tmp_path):
    source = tmp_path / "models"
    live = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid9"
    metrics = source / "combiner_trained_on_fiq_RN50x4_2026-01-02_00:00:00_pid10"
    _write(live / "training_hyperparameters.json", b"{}")
    _write(metrics / "validation_metrics.csv", b"epoch,recall\n1,0.1\n")
    (live / "saved_models").mkdir()
    (metrics / "saved_models").mkdir()
    _age_tree(source, timedelta(hours=25))

    result = _scan(source, tmp_path / "outputs", live_pids=(9,))

    by_pid = {run.pid: run for run in result.runs}
    assert by_pid[9].classification == "active"
    assert by_pid[10].classification == "unresolved"
    assert "nonempty-metrics-without-checkpoint" in by_pid[10].reasons


def test_any_legacy_writer_marks_empty_runs_active(tmp_path):
    source = tmp_path / "models"
    run = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid11"
    _write(run / "training_hyperparameters.json", b"{}")
    (run / "saved_models").mkdir()
    _age_tree(run, timedelta(hours=25))

    result = _scan(source, tmp_path / "outputs", writer_pids=(200,))

    assert result.runs[0].classification == "active"
    assert "legacy-writer-active" in result.runs[0].reasons


def test_unknown_top_level_file_blocks_clean_scan(tmp_path):
    source = tmp_path / "models"
    mystery = _write(source / "mystery.bin", b"unknown")

    result = _scan(source, tmp_path / "outputs")

    assert result.unknown_paths == (mystery,)


def test_xlsx_outside_run_is_classified_as_legacy_report(tmp_path):
    source = tmp_path / "models"
    report = _write(source / "validation_metrics_summary.xlsx", b"xlsx")

    result = _scan(source, tmp_path / "outputs")

    assert result.reports == (report,)
    assert result.unknown_paths == ()


def test_nonempty_corrupt_checkpoint_is_unresolved(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    _write(run / "saved_models/tuned.pt", b"not-a-checkpoint")
    _age_tree(run, timedelta(hours=25))

    result = _scan(source, tmp_path / "outputs")

    assert result.runs[0].classification == "unresolved"
    assert "checkpoint-format-invalid" in result.runs[0].reasons


def test_sha256_file_matches_hashlib(tmp_path):
    checkpoint = _write(tmp_path / "model.pt", b"abc" * 1000)

    assert sha256_file(checkpoint) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_sha256_file_rejects_changed_snapshot(tmp_path, monkeypatch):
    checkpoint = _write(tmp_path / "model.pt", b"original")
    original_stat = Path.stat
    calls = 0

    def changing_stat(path):
        nonlocal calls
        result = original_stat(path)
        if path == checkpoint:
            calls += 1
            if calls == 2:
                checkpoint.write_bytes(b"changed-size")
                result = original_stat(path)
        return result

    monkeypatch.setattr(Path, "stat", changing_stat)

    try:
        sha256_file(checkpoint)
    except SourceChangedError as error:
        assert str(checkpoint) in str(error)
    else:
        raise AssertionError("changed file must reject its snapshot")


def test_plan_groups_only_byte_identical_checkpoints(tmp_path):
    source = tmp_path / "models"
    first = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    second = source / "clip_finetuned_on_fiq_RN50x4_2026-01-02_00:00:00_000002"
    third = source / "clip_finetuned_on_fiq_RN50x4_2026-01-03_00:00:00_000003"
    _checkpoint(first / "saved_models/tuned.pt", b"same")
    _checkpoint(second / "saved_models/tuned.pt", b"same")
    _checkpoint(third / "saved_models/tuned.pt", b"diff")

    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))

    assert len(plan.duplicate_groups) == 1
    group = next(iter(plan.duplicate_groups.values()))
    assert len(group) == 2
    assert all(path.name == "tuned.pt" for path in group)
    deduplicated = [
        action for action in plan.actions
        if action.status == "planned-deduplicate"
    ]
    assert len(deduplicated) == 1
    assert deduplicated[0].canonical


def test_same_size_different_hash_is_not_duplicate(tmp_path):
    source = tmp_path / "models"
    first = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    second = source / "clip_finetuned_on_fiq_RN50x4_2026-01-02_00:00:00_000002"
    _checkpoint(first / "saved_models/tuned.pt", b"aaaa")
    _checkpoint(second / "saved_models/tuned.pt", b"bbbb")

    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))

    assert plan.duplicate_groups == {}


def test_reports_contain_path_hash_status_and_canonical(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))

    csv_path, json_path = write_migration_reports(plan, tmp_path / "outputs")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["old_path"].endswith("saved_models/tuned.pt")
    assert rows[0]["new_path"].endswith("checkpoints/tuned.pt")
    assert rows[0]["sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["valid_runs"] == 1
    assert report["logical_bytes"] >= checkpoint.stat().st_size
    assert report["status_counts"] == {"planned-move": 1}


def test_active_unknown_and_collision_are_plan_blockers(tmp_path):
    source = tmp_path / "models"
    active = source / "combiner_trained_on_fiq_RN50x4_2026-01-01_00:00:00_pid7"
    _checkpoint(active / "saved_models/combiner.pt", b"weights")
    _write(source / "mystery.bin", b"unknown")

    scan = _scan(source, tmp_path / "outputs", live_pids=(7,))
    plan = build_migration_plan(scan)

    assert plan.has_blockers
    assert {action.status for action in plan.actions} >= {
        "skipped-active",
        "unresolved",
    }


def test_apply_deduplicates_verifies_and_finalize_removes_source(tmp_path):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    first = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    second = source / "clip_finetuned_on_fiq_RN50x4_2026-01-02_00:00:00_000002"
    failed = source / "combiner_trained_on_fiq_RN50x4_2026-01-03_00:00:00_pid999"
    _checkpoint(first / "saved_models/tuned.pt", b"same weights")
    _write(first / "validation_metrics.csv", b"epoch,r\n1,0.5\n")
    _checkpoint(second / "saved_models/tuned.pt", b"same weights")
    _write(second / "validation_metrics.csv", b"epoch,r\n1,0.5\n")
    _write(failed / "training_hyperparameters.json", b"{}")
    (failed / "saved_models").mkdir()
    _write(source / "validation_metrics_summary.xlsx", b"xlsx")
    _age_tree(failed, timedelta(hours=25))
    plan = build_migration_plan(_scan(source, output))

    applied = apply_migration(plan)
    manifest = output / "migration_manifest.csv"
    verification = verify_migration(manifest)

    checkpoint_paths = sorted(
        output.glob(
            "fashioniq/clip-finetune/rn50x4/*/checkpoints/tuned.pt"
        )
    )
    assert len(checkpoint_paths) == 2
    assert checkpoint_paths[0].stat().st_ino == checkpoint_paths[1].stat().st_ino
    assert not failed.exists()
    assert (output / "reports/legacy/validation_metrics_summary.xlsx").exists()
    assert source.exists()
    assert verification.ok
    assert verification.checked_files == 5
    assert not applied.has_blockers
    assert {action.status for action in applied.actions} == {
        "moved",
        "deduplicated",
        "deleted-failed",
    }

    finalize_source(source, verification)

    assert not source.exists()


@pytest.mark.parametrize("blocker", ["active", "unknown", "collision"])
def test_apply_blockers_change_no_files(tmp_path, blocker):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    live_pids = ()
    if blocker == "active":
        renamed = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_pid7"
        run.rename(renamed)
        run = renamed
        checkpoint = run / "saved_models/tuned.pt"
        live_pids = (7,)
    elif blocker == "unknown":
        _write(source / "mystery.bin", b"unknown")

    scan = _scan(source, output, live_pids=live_pids)
    if blocker == "collision":
        scan.runs[0].destination.mkdir(parents=True)
    plan = build_migration_plan(scan)

    with pytest.raises(MigrationBlockedError):
        apply_migration(plan)

    assert checkpoint.exists()
    if blocker != "collision":
        assert not output.exists()


def test_apply_rejects_cross_filesystem_plan(tmp_path, monkeypatch):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))
    monkeypatch.setattr(migration, "_same_filesystem", lambda left, right: False)

    with pytest.raises(MigrationBlockedError, match="same filesystem"):
        apply_migration(plan)

    assert checkpoint.exists()


def test_apply_rejects_source_changed_after_plan(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    checkpoint = _checkpoint(run / "saved_models/tuned.pt", b"weights")
    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))
    checkpoint.write_bytes(b"changed")

    with pytest.raises(SourceChangedError):
        apply_migration(plan)

    assert checkpoint.exists()


def test_apply_rejects_symlink_inside_source(tmp_path):
    source = tmp_path / "models"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    _checkpoint(run / "saved_models/tuned.pt", b"weights")
    target = _write(tmp_path / "outside.txt", b"outside")
    (run / "outside-link").symlink_to(target)
    plan = build_migration_plan(_scan(source, tmp_path / "outputs"))

    with pytest.raises(MigrationBlockedError, match="symbolic link"):
        apply_migration(plan)

    assert (run / "outside-link").is_symlink()


def test_verify_detects_tampered_destination(tmp_path):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    _checkpoint(run / "saved_models/tuned.pt", b"weights")
    apply_migration(build_migration_plan(_scan(source, output)))
    migrated = next(output.glob("**/checkpoints/tuned.pt"))
    tampered = bytearray(migrated.read_bytes())
    tampered[-1] ^= 1
    migrated.write_bytes(tampered)

    verification = verify_migration(output / "migration_manifest.csv")

    assert not verification.ok
    assert any("sha256 mismatch" in error for error in verification.errors)


def test_finalize_rejects_failed_verification_and_nonempty_source(tmp_path):
    source = tmp_path / "models"
    source.mkdir()
    failed = VerificationResult(
        ok=False,
        checked_files=0,
        checked_bytes=0,
        errors=("bad hash",),
    )
    with pytest.raises(MigrationBlockedError, match="verification"):
        finalize_source(source, failed)

    _write(source / "unexpected.txt", b"unexpected")
    passed = VerificationResult(
        ok=True,
        checked_files=0,
        checked_bytes=0,
        errors=(),
    )
    with pytest.raises(MigrationBlockedError, match="not empty"):
        finalize_source(source, passed)
    assert source.exists()


def test_find_legacy_writer_pids_resolves_relative_script(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    script = _write(project / "src/combiner_train.py", b"")
    proc = tmp_path / "proc"
    process = proc / "123"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0src/combiner_train.py\0")
    (process / "cwd").symlink_to(project, target_is_directory=True)

    assert find_legacy_writer_pids(project, proc_root=proc) == (123,)
    assert script.exists()
