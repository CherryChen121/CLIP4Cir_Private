import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import validate as validate_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SCRIPT = PROJECT_ROOT / "src" / "validate.py"


def _run_validate(*arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_validate_cli_exposes_combined_dataset_options():
    result = _run_validate("--help")

    assert result.returncode == 0, result.stderr
    for option in (
        "--fashioniq-root",
        "--dress-types",
        "--fashioniq-split",
        "--output-root",
        "--output-dataset",
        "--evaluation-name",
    ):
        assert option in result.stdout


def test_validate_cli_rejects_unsupported_fashioniq_split_before_model_loading():
    result = _run_validate(
        "--dataset",
        "FashionIQ",
        "--combining-function",
        "sum",
        "--fashioniq-split",
        "invalid",
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_fashioniq_csv_projection_flattens_category_and_averages():
    results = [
        {
            "model_path": None,
            "epoch": None,
            "per_category": {
                "Internal": {
                    "recall_at1": 10.0,
                    "recall_at5": 20.0,
                    "recall_at10": 30.0,
                }
            },
            "aggregate": {
                "average_recall_at1": 10.0,
                "average_recall_at5": 20.0,
                "average_recall_at10": 30.0,
            },
        }
    ]

    rows, fieldnames = validate_module._csv_projection(
        SimpleNamespace(dataset="FashionIQ"),
        ["Internal"],
        results,
    )

    assert fieldnames == [
        "model_path",
        "epoch",
        "Internal_recall_at1",
        "Internal_recall_at5",
        "Internal_recall_at10",
        "average_recall_at1",
        "average_recall_at5",
        "average_recall_at10",
    ]
    assert rows == [
        {
            "model_path": None,
            "epoch": None,
            "Internal_recall_at1": 10.0,
            "Internal_recall_at5": 20.0,
            "Internal_recall_at10": 30.0,
            "average_recall_at1": 10.0,
            "average_recall_at5": 20.0,
            "average_recall_at10": 30.0,
        }
    ]


def test_cirr_csv_projection_uses_explicit_group_and_global_names():
    aggregate = {
        "group_recall_at1": 1.0,
        "group_recall_at2": 2.0,
        "group_recall_at3": 3.0,
        "global_recall_at1": 4.0,
        "global_recall_at5": 5.0,
        "global_recall_at10": 6.0,
        "global_recall_at50": 7.0,
    }
    results = [
        {
            "model_path": "/checkpoints/clip.pt",
            "epoch": None,
            "per_category": {},
            "aggregate": aggregate,
        }
    ]

    rows, fieldnames = validate_module._csv_projection(
        SimpleNamespace(dataset="CIRR"),
        [],
        results,
    )

    assert fieldnames == [
        "model_path",
        "epoch",
        *aggregate.keys(),
    ]
    assert rows == [
        {
            "model_path": "/checkpoints/clip.pt",
            "epoch": None,
            **aggregate,
        }
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


def _single_fashioniq_result():
    return [
        {
            "model_path": None,
            "epoch": None,
            "per_category": {
                "Internal": {
                    "recall_at1": 10.0,
                    "recall_at5": 20.0,
                    "recall_at10": 30.0,
                }
            },
            "aggregate": {
                "average_recall_at1": 10.0,
                "average_recall_at5": 20.0,
                "average_recall_at10": 30.0,
            },
        }
    ]


def _fashioniq_main_arguments(dataset_root, output_root):
    return [
        "--dataset",
        "FashionIQ",
        "--fashioniq-root",
        str(dataset_root),
        "--dress-types",
        "Internal",
        "--fashioniq-split",
        "val",
        "--combining-function",
        "sum",
        "--clip-model-name",
        "ViT-B/32",
        "--output-root",
        str(output_root),
        "--output-dataset",
        "combined-fundus-cir",
        "--evaluation-name",
        "internal-val",
    ]


def test_validate_main_publishes_structured_success_run(tmp_path, monkeypatch):
    dataset_root = _make_fashioniq_root(tmp_path)
    output_root = tmp_path / "artifacts"
    monkeypatch.setattr(
        validate_module,
        "_run_evaluation",
        lambda _args, _categories: _single_fashioniq_result(),
        raising=False,
    )
    monkeypatch.setattr(
        validate_module.clip,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model loaded outside _run_evaluation")
        ),
    )

    validate_module.main(
        _fashioniq_main_arguments(dataset_root, output_root)
    )

    model_root = (
        output_root
        / "combined-fundus-cir/evaluation/vit-b-32"
    )
    runs = list(model_root.iterdir())
    assert len(runs) == 1
    run = runs[0]
    manifest = json.loads(
        (run / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "succeeded"
    assert manifest["evaluation_name"] == "internal-val"
    assert manifest["dataset"] == "combined-fundus-cir"
    assert manifest["dataset_root_resolved"] == str(dataset_root.resolve())
    assert manifest["cli_args"]["output_dataset"] == "combined-fundus-cir"
    metrics = json.loads(
        (run / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["results"] == _single_fashioniq_result()
    with (run / "evaluation_metrics.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "model_path": "",
            "epoch": "",
            "Internal_recall_at1": "10.0",
            "Internal_recall_at5": "20.0",
            "Internal_recall_at10": "30.0",
            "average_recall_at1": "10.0",
            "average_recall_at5": "20.0",
            "average_recall_at10": "30.0",
        }
    ]
    assert (run / "evaluation.log").exists()


def test_validate_main_records_failed_run_without_metrics(
    tmp_path,
    monkeypatch,
):
    dataset_root = _make_fashioniq_root(tmp_path)
    output_root = tmp_path / "artifacts"

    def fail_evaluation(_args, _categories):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(
        validate_module,
        "_run_evaluation",
        fail_evaluation,
        raising=False,
    )
    monkeypatch.setattr(
        validate_module.clip,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model loaded outside _run_evaluation")
        ),
    )

    with pytest.raises(RuntimeError, match="shape mismatch"):
        validate_module.main(
            _fashioniq_main_arguments(dataset_root, output_root)
        )

    run = next(
        (
            output_root
            / "combined-fundus-cir/evaluation/vit-b-32"
        ).iterdir()
    )
    manifest = json.loads(
        (run / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "type": "RuntimeError",
        "message": "shape mismatch",
    }
    assert "shape mismatch" in (
        run / "evaluation.log"
    ).read_text(encoding="utf-8")
    assert not (run / "evaluation_metrics.json").exists()
    assert not (run / "evaluation_metrics.csv").exists()
