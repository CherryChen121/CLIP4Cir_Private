from __future__ import annotations

import subprocess
from typing import Callable, Tuple

try:
    from gpu_queue_core import GpuSnapshot, ProbeError
except ImportError:
    from src.gpu_queue_core import GpuSnapshot, ProbeError


class NvidiaSmiProbe:
    GPU_QUERY = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    PROCESS_QUERY = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]

    def __init__(self, runner: Callable = subprocess.run):
        self.runner = runner

    def _run(self, argv) -> str:
        result = self.runner(
            argv,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
            raise ProbeError(detail)
        return result.stdout.strip()

    @staticmethod
    def _integer(value: str, field: str) -> int:
        value = value.strip()
        if not value.isdigit():
            raise ProbeError(f"invalid {field}: {value}")
        return int(value)

    def snapshot(self) -> Tuple[GpuSnapshot, ...]:
        gpu_output = self._run(self.GPU_QUERY)
        process_output = self._run(self.PROCESS_QUERY)
        gpu_rows = {}
        uuid_to_index = {}
        for line in gpu_output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                raise ProbeError(f"malformed GPU row: {line}")
            index = self._integer(parts[0], "GPU index")
            uuid = parts[1]
            if index in gpu_rows or uuid in uuid_to_index or not uuid:
                raise ProbeError("duplicate GPU index or UUID")
            memory = self._integer(parts[2], "memory.used")
            utilization = self._integer(parts[3], "utilization.gpu")
            gpu_rows[index] = (uuid, memory, utilization)
            uuid_to_index[uuid] = index

        if tuple(sorted(gpu_rows)) != tuple(range(8)):
            raise ProbeError(f"expected GPU indices 0..7, found {tuple(sorted(gpu_rows))}")

        pids = {index: [] for index in gpu_rows}
        if process_output:
            for line in process_output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 4:
                    raise ProbeError(f"malformed compute process row: {line}")
                uuid = parts[0]
                if uuid not in uuid_to_index:
                    raise ProbeError(f"compute process references unknown GPU UUID: {uuid}")
                pid = self._integer(parts[1], "compute PID")
                pids[uuid_to_index[uuid]].append(pid)

        return tuple(
            GpuSnapshot(
                index=index,
                uuid=gpu_rows[index][0],
                memory_used_mib=gpu_rows[index][1],
                utilization_percent=gpu_rows[index][2],
                compute_pids=tuple(sorted(set(pids[index]))),
            )
            for index in range(8)
        )
