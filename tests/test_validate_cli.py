import os
import subprocess
import sys
from pathlib import Path


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
    assert "--fashioniq-root" in result.stdout
    assert "--dress-types" in result.stdout
    assert "--fashioniq-split" in result.stdout


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
