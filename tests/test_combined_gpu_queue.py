import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from combined_gpu_queue import (
    NvidiaSmiProbe,
    OwnedProcessLauncher,
    ProcessIdentity,
)
from gpu_queue_core import GpuSnapshot, ProbeError, TaskSpec
from gpu_queue_worker import WorkerError, run_manifest


def _gpu_rows():
    return [f"{index}, GPU-{index}, 0, 0" for index in range(8)]


class FakeRunner:
    def __init__(self, gpu_rows=None, process_rows=None, returncode=0):
        self.gpu_rows = _gpu_rows() if gpu_rows is None else gpu_rows
        self.process_rows = [] if process_rows is None else process_rows
        self.returncode = returncode
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        rows = self.gpu_rows if "--query-gpu=" in argv[1] else self.process_rows
        return CompletedProcess(argv, self.returncode, "\n".join(rows), "probe failed")


def test_probe_keeps_zero_utilization_gpu_busy_when_compute_pid_exists():
    runner = FakeRunner(
        process_rows=["GPU-0, 1790625, ray::WorkerDict, 14898"]
    )

    gpu = NvidiaSmiProbe(runner=runner).snapshot()[0]

    assert gpu.utilization_percent == 0
    assert gpu.compute_pids == (1790625,)
    assert all(call[1]["shell"] is False for call in runner.calls)


def test_probe_accepts_blank_compute_process_output():
    snapshots = NvidiaSmiProbe(runner=FakeRunner()).snapshot()

    assert len(snapshots) == 8
    assert all(gpu.compute_pids == () for gpu in snapshots)


def test_probe_rejects_unknown_process_uuid():
    runner = FakeRunner(process_rows=["GPU-unknown, 12, python, 100"])

    with pytest.raises(ProbeError, match="unknown GPU UUID"):
        NvidiaSmiProbe(runner=runner).snapshot()


def test_probe_rejects_nonzero_nvidia_smi_exit():
    with pytest.raises(ProbeError, match="probe failed"):
        NvidiaSmiProbe(runner=FakeRunner(returncode=1)).snapshot()


def test_probe_rejects_missing_gpu_index():
    with pytest.raises(ProbeError, match="indices 0..7"):
        NvidiaSmiProbe(runner=FakeRunner(gpu_rows=_gpu_rows()[:-1])).snapshot()


def test_worker_records_child_exit_code_without_shell(tmp_path):
    log_path = tmp_path / "task.log"
    result_path = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", "raise SystemExit(7)"],
                "cwd": str(tmp_path),
                "env_overrides": {
                    "CUDA_VISIBLE_DEVICES": "6",
                    "NCCL_P2P_DISABLE": "1",
                },
                "log_path": str(log_path),
                "result_path": str(result_path),
            }
        ),
        encoding="utf-8",
    )

    assert run_manifest(manifest) == 7
    assert json.loads(result_path.read_text()) == {"return_code": 7}
    assert log_path.exists()


def test_worker_rejects_existing_log_without_starting_child(tmp_path):
    marker = tmp_path / "child-started"
    log_path = tmp_path / "task.log"
    log_path.write_text("existing", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
                "cwd": str(tmp_path),
                "env_overrides": {"CUDA_VISIBLE_DEVICES": "0"},
                "log_path": str(log_path),
                "result_path": str(tmp_path / "result.json"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkerError, match="already exists"):
        run_manifest(manifest)
    assert not marker.exists()


class FakePopen:
    def __init__(self):
        self.calls = []
        self.pid = 4321

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return type("Process", (), {"pid": self.pid})()


class FakeInspector:
    def __init__(self, identity):
        self.current = identity

    def identity(self, pid):
        return self.current if self.current and self.current.pid == pid else None


def _task():
    return TaskSpec(
        task_id="A01",
        phase="A",
        ordinal=1,
        argv=(sys.executable, "-c", "raise SystemExit(0)"),
        env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
        log_name="a01.log",
        source="fixture",
    )


def test_launcher_uses_worker_new_session_and_persists_only_safe_env(tmp_path):
    identity = ProcessIdentity(4321, 4321, 99, "worker-digest")
    popen = FakePopen()
    launcher = OwnedProcessLauncher(
        project_root=tmp_path,
        python_executable=Path(sys.executable),
        worker_script=tmp_path / "gpu_queue_worker.py",
        proc_inspector=FakeInspector(identity),
        popen_factory=popen,
    )

    launch = launcher.start(
        _task(),
        GpuSnapshot(6, "GPU-6", 0, 0, ()),
        tmp_path / "A01",
    )

    argv, kwargs = popen.calls[-1]
    assert kwargs["shell"] is False
    assert kwargs["start_new_session"] is True
    assert argv[0] == sys.executable
    manifest = json.loads(launch.manifest_path.read_text())
    assert manifest["env_overrides"] == {
        "CUDA_VISIBLE_DEVICES": "6",
        "NCCL_P2P_DISABLE": "1",
    }
    assert "PATH" not in manifest["env_overrides"]


def test_terminate_refuses_signal_when_process_identity_changed(tmp_path):
    original = ProcessIdentity(4321, 4321, 99, "original")
    changed = ProcessIdentity(4321, 4321, 100, "different")
    signals = []
    launcher = OwnedProcessLauncher(
        project_root=tmp_path,
        python_executable=Path(sys.executable),
        worker_script=tmp_path / "gpu_queue_worker.py",
        proc_inspector=FakeInspector(changed),
        killpg=lambda pgid, signal_number: signals.append((pgid, signal_number)),
    )

    assert launcher.terminate(original, grace_seconds=0) is False
    assert signals == []
