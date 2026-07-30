from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import zipfile

from model_output_migration import scan_legacy_outputs


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
