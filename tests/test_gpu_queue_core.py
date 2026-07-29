import json
from pathlib import Path

import pytest

from gpu_queue_core import (
    PreflightError,
    QueueSpec,
    TaskSpec,
    parse_combined_queue,
    preflight_queue,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "命令.sh"
PYTHON = Path("/data0/qrchen/miniconda3/envs/clip4cir/bin/python")


def _line(log_name="job.log", extra=""):
    return (
        "CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 nohup python "
        "src/combiner_train.py --dataset FashionIQ "
        "--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset "
        f"{extra}> {log_name} 2>&1 &"
    )


def _command_file(tmp_path, phase_a=None, phase_b=None):
    phase_a = phase_a or [_line(f"a{i}.log") for i in range(10)]
    phase_b = phase_b or [_line(f"b{i}.log") for i in range(10)]
    path = tmp_path / "commands.sh"
    path.write_text(
        "\n".join(
            [
                "# COMBINED_COMMANDS_BEGIN",
                "# Phase A:",
                *phase_a,
                "# Phase B:",
                *phase_b,
                "# Phase C:",
                _line("c.log"),
                "# COMBINED_COMMANDS_END",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _task(argv):
    return TaskSpec(
        task_id="A01",
        phase="A",
        ordinal=1,
        argv=tuple(argv),
        env={"CUDA_VISIBLE_DEVICES": "0", "NCCL_P2P_DISABLE": "1"},
        log_name="job.log",
        source="fixture",
    )


def _queue(task):
    return QueueSpec(tasks=(task,), command_sha256="fixture")


def test_real_combined_queue_contains_only_ordered_phase_a_and_b():
    queue = parse_combined_queue(COMMAND_FILE, PROJECT_ROOT, PYTHON)

    assert len(queue.tasks) == 20
    assert [task.task_id for task in queue.tasks[:2]] == ["A01", "A02"]
    assert [task.phase for task in queue.tasks] == ["A"] * 10 + ["B"] * 10
    assert len({task.log_name for task in queue.tasks}) == 20
    assert all(task.argv[0] == str(PYTHON) for task in queue.tasks)
    assert all("Phase C" not in task.source for task in queue.tasks)


def test_parser_rejects_shell_operators_outside_fixed_redirection(tmp_path):
    unsafe = _line("unsafe.log", extra="; touch /tmp/unsafe ")
    path = _command_file(tmp_path, phase_a=[unsafe] + [_line(f"a{i}.log") for i in range(1, 10)])

    with pytest.raises(PreflightError, match="unsupported shell token"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_parser_rejects_wrong_phase_count(tmp_path):
    path = _command_file(tmp_path, phase_a=[_line(f"a{i}.log") for i in range(9)])

    with pytest.raises(PreflightError, match="Phase A.*10"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_parser_rejects_duplicate_log_names(tmp_path):
    phase_a = [_line("duplicate.log")] + [_line(f"a{i}.log") for i in range(1, 10)]
    phase_b = [_line("duplicate.log")] + [_line(f"b{i}.log") for i in range(1, 10)]
    path = _command_file(tmp_path, phase_a=phase_a, phase_b=phase_b)

    with pytest.raises(PreflightError, match="duplicate log"):
        parse_combined_queue(path, tmp_path, PYTHON)


def test_preflight_rejects_missing_absolute_path():
    task = _task(
        (
            str(PYTHON),
            "src/combiner_train.py",
            "--dataset",
            "FashionIQ",
            "--fashioniq-root",
            "/definitely/missing/dataset",
        )
    )

    with pytest.raises(PreflightError, match="/definitely/missing/dataset"):
        preflight_queue(_queue(task), PROJECT_ROOT)


def test_preflight_rejects_unknown_cli_option():
    task = _task(
        (
            str(PYTHON),
            "src/combiner_train.py",
            "--dataset",
            "FashionIQ",
            "--definitely-unknown",
            "1",
        )
    )

    with pytest.raises(PreflightError, match="--definitely-unknown"):
        preflight_queue(_queue(task), PROJECT_ROOT)
