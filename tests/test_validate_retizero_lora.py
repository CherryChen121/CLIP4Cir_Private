import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import validate_retizero_lora as retizero_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "validate_retizero_lora.py"


def _run(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_exposes_dataset_and_output_options():
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    for option in (
        "--fashioniq-root",
        "--dress-types",
        "--fashioniq-split",
        "--output-root",
        "--output-dataset",
        "--evaluation-name",
        "--output-csv",
    ):
        assert option in result.stdout


def test_cli_rejects_output_csv_outside_run():
    result = _run(
        "--model-paths",
        "/checkpoints/model.pth",
        "--base-weight-path",
        "/weights/RetiZero.pth",
        "--output-csv",
        "../outside.csv",
    )

    assert result.returncode == 2
    assert "filename inside the evaluation run" in result.stderr


def test_validate_single_model_passes_split_and_root_to_datasets(
    monkeypatch,
):
    dataset_calls = []

    def fake_dataset(
        split,
        dress_types,
        mode,
        preprocess,
        dataset_root=None,
        return_target=False,
    ):
        dataset_calls.append(
            {
                "split": split,
                "dress_types": dress_types,
                "mode": mode,
                "preprocess": preprocess,
                "dataset_root": dataset_root,
                "return_target": return_target,
            }
        )
        return object()

    class FakeModel:
        def eval(self):
            return self

        def float(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(
        retizero_validation,
        "FashionIQDataset",
        fake_dataset,
    )
    monkeypatch.setattr(
        retizero_validation,
        "extract_index_features",
        lambda _dataset, _model: ("features", ["image"]),
    )
    monkeypatch.setattr(
        retizero_validation,
        "compute_fiq_val_metrics",
        lambda *_args: (1.0, 5.0, 10.0),
    )

    retizero_validation.validate_single_model(
        FakeModel(),
        ["Internal"],
        "preprocess",
        "combine",
        split="test",
        dataset_root="/datasets/combined",
    )

    assert dataset_calls == [
        {
            "split": "test",
            "dress_types": ["Internal"],
            "mode": "classic",
            "preprocess": "preprocess",
            "dataset_root": "/datasets/combined",
            "return_target": False,
        },
        {
            "split": "test",
            "dress_types": ["Internal"],
            "mode": "relative",
            "preprocess": "preprocess",
            "dataset_root": "/datasets/combined",
            "return_target": True,
        },
    ]


def _make_fashioniq_root(tmp_path):
    root = tmp_path / "Combined_Fundus_CIR_Dataset"
    (root / "captions").mkdir(parents=True)
    (root / "image_splits").mkdir()
    (root / "images").mkdir()
    (root / "captions/cap.Internal.val.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    (root / "image_splits/split.Internal.val.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    return root


def _canonical_results():
    return [
        {
            "model_path": "/checkpoints/a.pth",
            "epoch": 4,
            "classification_val_accuracy": 0.80,
            "per_category": {
                "Internal": {
                    "recall_at1": 11.0,
                    "recall_at5": 21.0,
                    "recall_at10": 31.0,
                }
            },
            "aggregate": {
                "average_recall_at1": 11.0,
                "average_recall_at5": 21.0,
                "average_recall_at10": 31.0,
                "average_recall": 21.0,
            },
        },
        {
            "model_path": "/checkpoints/b.pth",
            "epoch": 9,
            "classification_val_accuracy": 0.84,
            "per_category": {
                "Internal": {
                    "recall_at1": 12.0,
                    "recall_at5": 22.0,
                    "recall_at10": 32.0,
                }
            },
            "aggregate": {
                "average_recall_at1": 12.0,
                "average_recall_at5": 22.0,
                "average_recall_at10": 32.0,
                "average_recall": 22.0,
            },
        },
    ]


def test_canonical_checkpoint_result_preserves_model_and_category_metrics():
    result = retizero_validation._canonical_checkpoint_result(
        model_path="/checkpoints/a.pth",
        sequence=1,
        checkpoint_epoch=4,
        checkpoint_accuracy=0.8,
        categories=["Internal"],
        metrics={
            "Internal_recall_at1": 11.0,
            "Internal_recall_at5": 21.0,
            "Internal_recall_at10": 31.0,
            "average_recall_at1": 11.0,
            "average_recall_at5": 21.0,
            "average_recall_at10": 31.0,
            "average_recall": 21.0,
        },
    )

    assert result == _canonical_results()[0]


def _main_arguments(dataset_root, output_root):
    return [
        "--model-paths",
        "/checkpoints/a.pth",
        "/checkpoints/b.pth",
        "--base-weight-path",
        "/weights/RetiZero.pth",
        "--fashioniq-root",
        str(dataset_root),
        "--dress-types",
        "Internal",
        "--fashioniq-split",
        "val",
        "--output-root",
        str(output_root),
        "--output-dataset",
        "combined-fundus-cir",
        "--evaluation-name",
        "retizero-internal-val",
    ]


def test_main_publishes_multi_checkpoint_results(tmp_path, monkeypatch):
    dataset_root = _make_fashioniq_root(tmp_path)
    output_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        retizero_validation,
        "_run_evaluation",
        lambda _args, _categories: _canonical_results(),
        raising=False,
    )
    monkeypatch.setattr(
        retizero_validation,
        "load_retizero_with_lora",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("checkpoint loaded outside _run_evaluation")
        ),
    )

    retizero_validation.main(
        _main_arguments(dataset_root, output_root)
    )

    model_root = (
        output_root
        / "combined-fundus-cir/evaluation/retizero-lora"
    )
    runs = list(model_root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    manifest = json.loads(
        (run / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "succeeded"
    assert manifest["evaluation_name"] == "retizero-internal-val"
    assert manifest["input_paths"] == {
        "base_weight_path": "/weights/RetiZero.pth",
        "model_paths": [
            "/checkpoints/a.pth",
            "/checkpoints/b.pth",
        ],
    }
    metrics = json.loads(
        (run / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["results"] == _canonical_results()
    with (run / "evaluation_metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0] == {
        "model_path": "/checkpoints/a.pth",
        "epoch": "4",
        "classification_val_accuracy": "0.8",
        "Internal_recall_at1": "11.0",
        "Internal_recall_at5": "21.0",
        "Internal_recall_at10": "31.0",
        "average_recall_at1": "11.0",
        "average_recall_at5": "21.0",
        "average_recall_at10": "31.0",
        "average_recall": "21.0",
    }
    assert (run / "evaluation.log").exists()


def test_main_records_failed_run_without_metrics(tmp_path, monkeypatch):
    dataset_root = _make_fashioniq_root(tmp_path)
    output_root = tmp_path / "artifacts"

    def fail_evaluation(_args, _categories):
        raise RuntimeError("LoRA load failed")

    monkeypatch.setattr(
        retizero_validation,
        "_run_evaluation",
        fail_evaluation,
        raising=False,
    )
    monkeypatch.setattr(
        retizero_validation,
        "load_retizero_with_lora",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("checkpoint loaded outside _run_evaluation")
        ),
    )

    with pytest.raises(RuntimeError, match="LoRA load failed"):
        retizero_validation.main(
            _main_arguments(dataset_root, output_root)
        )

    run = next(
        (
            output_root
            / "combined-fundus-cir/evaluation/retizero-lora"
        ).iterdir()
    )
    manifest = json.loads(
        (run / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "LoRA load failed",
    }
    assert "LoRA load failed" in (
        run / "evaluation.log"
    ).read_text(encoding="utf-8")
    assert not (run / "evaluation_metrics.json").exists()
    assert not (run / "evaluation_metrics.csv").exists()
