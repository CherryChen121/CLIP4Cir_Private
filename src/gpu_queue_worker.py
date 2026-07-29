from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


ALLOWED_ENV_OVERRIDES = {"CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE"}


class WorkerError(RuntimeError):
    pass


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkerError(f"invalid manifest: {exc}") from exc
    required = {"argv", "cwd", "env_overrides", "log_path", "result_path"}
    if set(manifest) != required:
        raise WorkerError("manifest keys do not match the worker schema")
    if not isinstance(manifest["argv"], list) or not manifest["argv"]:
        raise WorkerError("manifest argv must be a non-empty list")
    if not all(isinstance(item, str) and item for item in manifest["argv"]):
        raise WorkerError("manifest argv contains an invalid value")
    overrides = manifest["env_overrides"]
    if not isinstance(overrides, dict) or not set(overrides).issubset(ALLOWED_ENV_OVERRIDES):
        raise WorkerError("manifest contains an unsafe environment override")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in overrides.items()):
        raise WorkerError("manifest environment overrides must be strings")
    return manifest


def run_manifest(manifest_path: Path) -> int:
    manifest = _load_manifest(Path(manifest_path))
    cwd = Path(manifest["cwd"])
    log_path = Path(manifest["log_path"])
    result_path = Path(manifest["result_path"])
    if not cwd.is_dir():
        raise WorkerError(f"working directory does not exist: {cwd}")
    if log_path.exists():
        raise WorkerError(f"log already exists: {log_path}")
    if result_path.exists():
        raise WorkerError(f"result already exists: {result_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(manifest["env_overrides"])
    with log_path.open("x", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            manifest["argv"],
            cwd=str(cwd),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            check=False,
        )
    return_code = int(completed.returncode)
    _atomic_json(result_path, {"return_code": return_code})
    return return_code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_manifest(args.manifest)
    except WorkerError as exc:
        print(f"worker error: {exc}")
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
