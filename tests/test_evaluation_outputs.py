import csv
import json
from datetime import datetime
import sys

import pytest

from dataset_identity import DatasetIdentity
from evaluation_outputs import (
    create_evaluation_layout,
    discard_evaluation_metrics,
    finalize_evaluation,
    publish_evaluation_metrics,
    tee_evaluation_output,
    validate_metrics_csv_filename,
)


def _identity(dataset="idrid"):
    return DatasetIdentity(
        dataset_slug=dataset,
        dataset_format="fashioniq",
        root_requested="/datasets/link",
        root_resolved="/datasets/IDRiD_CIR_Dataset_cold",
        classification_evidence=("resolved-root:idrid",),
    )


def _make_layout(tmp_path):
    return create_evaluation_layout(
        project_root=tmp_path,
        output_root=None,
        identity=_identity(),
        evaluation_script="src/validate.py",
        evaluation_name="idrid-val",
        model_name="ViT-B/32",
        split="val",
        categories=["IDRiD"],
        cli_args={"clip_model_path": tmp_path / "clip.pt"},
        input_paths={"clip_model_path": tmp_path / "clip.pt"},
        started_at=datetime(2026, 7, 30, 16, 5, 6, 123456),
        pid=42,
    )


def test_create_evaluation_layout_writes_running_manifest(tmp_path):
    layout = _make_layout(tmp_path)

    assert layout.root == (
        tmp_path
        / "outputs/idrid/evaluation/vit-b-32/"
        "20260730-160506-123456-p42"
    )
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    assert manifest["dataset"] == "idrid"
    assert manifest["dataset_format"] == "fashioniq"
    assert manifest["dataset_root_requested"] == "/datasets/link"
    assert (
        manifest["dataset_root_resolved"]
        == "/datasets/IDRiD_CIR_Dataset_cold"
    )
    assert manifest["dataset_classification_evidence"] == [
        "resolved-root:idrid"
    ]
    assert manifest["evaluation_name"] == "idrid-val"
    assert manifest["model_slug"] == "vit-b-32"
    assert manifest["categories"] == ["IDRiD"]
    assert manifest["cli_args"]["clip_model_path"] == str(
        tmp_path / "clip.pt"
    )
    assert manifest["input_paths"]["clip_model_path"] == str(
        tmp_path / "clip.pt"
    )
    assert manifest["metric_files"] == {
        "json": "evaluation_metrics.json",
        "csv": "evaluation_metrics.csv",
    }
    assert layout.log.exists()


def test_create_evaluation_layout_never_reuses_run(tmp_path):
    _make_layout(tmp_path)

    with pytest.raises(FileExistsError):
        _make_layout(tmp_path)


@pytest.mark.parametrize(
    "value",
    ["/tmp/results.csv", "../results.csv", "sub/results.csv"],
)
def test_metrics_csv_filename_rejects_paths(value):
    with pytest.raises(ValueError, match="filename"):
        validate_metrics_csv_filename(value)


def test_metrics_csv_filename_accepts_basename():
    assert validate_metrics_csv_filename("retizero.csv") == "retizero.csv"


def test_publish_metrics_and_finalize_success(tmp_path):
    layout = _make_layout(tmp_path)

    publish_evaluation_metrics(
        layout,
        {
            "schema_version": 1,
            "results": [{"epoch": 3, "average_recall": 8.5}],
        },
        [{"epoch": 3, "average_recall": 8.5}],
        ["epoch", "average_recall"],
    )
    finalize_evaluation(
        layout,
        "succeeded",
        completed_at=datetime(2026, 7, 30, 16, 6, 7),
    )

    document = json.loads(layout.metrics_json.read_text(encoding="utf-8"))
    assert document["results"][0]["epoch"] == 3
    with layout.metrics_csv.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == [
            {"epoch": "3", "average_recall": "8.5"}
        ]
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["completed_at"] == "2026-07-30T16:06:07"
    assert "error" not in manifest
    assert not list(layout.root.glob(".*.tmp"))


def test_failed_run_discards_metrics_and_records_error(tmp_path):
    layout = _make_layout(tmp_path)
    publish_evaluation_metrics(layout, {"results": []}, [], ["epoch"])

    discard_evaluation_metrics(layout)
    error = RuntimeError("checkpoint mismatch")
    finalize_evaluation(layout, "failed", error=error)

    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "checkpoint mismatch",
    }
    assert not layout.metrics_json.exists()
    assert not layout.metrics_csv.exists()


def test_publish_failure_removes_partial_metric_files(tmp_path):
    layout = _make_layout(tmp_path)

    with pytest.raises(ValueError, match="fields not in fieldnames"):
        publish_evaluation_metrics(
            layout,
            {"results": [{"unexpected": 1}]},
            [{"unexpected": 1}],
            ["epoch"],
        )

    assert not layout.metrics_json.exists()
    assert not layout.metrics_csv.exists()
    assert not list(layout.root.glob(".*.tmp"))


def test_tee_copies_stdout_and_stderr(tmp_path, capsys):
    layout = _make_layout(tmp_path)

    with tee_evaluation_output(layout.log):
        print("metric line")
        print("warning line", file=sys.stderr)

    captured = capsys.readouterr()
    assert "metric line" in captured.out
    assert "warning line" in captured.err
    log = layout.log.read_text(encoding="utf-8")
    assert "metric line" in log
    assert "warning line" in log


def test_finalize_rejects_unknown_status(tmp_path):
    layout = _make_layout(tmp_path)

    with pytest.raises(ValueError, match="status"):
        finalize_evaluation(layout, "running")
