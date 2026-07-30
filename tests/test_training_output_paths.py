import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

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
