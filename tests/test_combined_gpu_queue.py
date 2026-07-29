from subprocess import CompletedProcess

import pytest

from combined_gpu_queue import NvidiaSmiProbe
from gpu_queue_core import ProbeError


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
