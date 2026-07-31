import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from dataset_identity import DatasetIdentity
from utils import save_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_save_model_uses_checkpoints_directory(tmp_path):
    model = nn.Linear(2, 1)
    save_model("tiny", 3, model, tmp_path)
    payload = torch.load(
        tmp_path / "checkpoints/tiny.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["epoch"] == 3
    assert not (tmp_path / "saved_models").exists()


def test_training_sources_do_not_construct_legacy_output_root():
    for relative in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert 'f"models/' not in source
        assert "/saved_models" not in source


def test_training_clis_expose_output_root():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    for script in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--output-root" in result.stdout
        assert "--output-dataset" in result.stdout


@pytest.mark.parametrize(
    ("module_name", "function_name", "stage"),
    [
        ("clip_fine_tune", "clip_finetune_fiq", "clip-finetune"),
        ("combiner_train", "combiner_training_fiq", "combiner"),
    ],
)
def test_fashioniq_entrypoint_routes_identity_before_model_loading(
    monkeypatch, module_name, function_name, stage
):
    module = importlib.import_module(module_name)
    identity = DatasetIdentity(
        dataset_slug="idrid",
        dataset_format="fashioniq",
        root_requested="/datasets/idrid",
        root_resolved="/datasets/idrid",
        classification_evidence=("test",),
    )
    captured = {}

    class LayoutCaptured(RuntimeError):
        pass

    monkeypatch.setattr(
        module,
        "resolve_fashioniq_training_identity",
        lambda **_kwargs: identity,
    )

    def capture_layout(**kwargs):
        captured.update(kwargs)
        raise LayoutCaptured

    monkeypatch.setattr(module, "create_run_layout", capture_layout)
    common = {
        "train_dress_types": ["IDRiD"],
        "val_dress_types": ["IDRiD"],
        "num_epochs": 1,
        "clip_model_name": "ViT-B/32",
        "batch_size": 1,
        "validation_frequency": 1,
        "transform": "clip",
        "save_training": False,
        "save_best": False,
        "output_root": None,
        "fashioniq_root": "/datasets/idrid",
        "output_dataset": None,
    }
    if stage == "clip-finetune":
        common.update(learning_rate=1e-6, encoder="both")
    else:
        common.update(
            projection_dim=64,
            hidden_dim=128,
            combiner_lr=1e-5,
            clip_bs=1,
        )

    with pytest.raises(LayoutCaptured):
        getattr(module, function_name)(**common)

    assert captured["dataset"] == "idrid"
    assert captured["dataset_format"] == "fashioniq"
    assert captured["dataset_root_resolved"] == "/datasets/idrid"
    assert captured["stage"] == stage
