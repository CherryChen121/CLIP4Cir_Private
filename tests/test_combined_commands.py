import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_FILE = PROJECT_ROOT / "命令.sh"
BEGIN = "# COMBINED_COMMANDS_BEGIN"
END = "# COMBINED_COMMANDS_END"
LEGACY_SUFFIX_SHA256 = "0b96411202f8eb89b33323162c3e2eaa1b4d5dbdc8947d3a217d9fea87f808ae"
COMBINED_ROOT_ARGUMENT = (
    "--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset"
)


def _command_parts():
    text = COMMAND_FILE.read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    combined = text.split(BEGIN, 1)[1].split(END, 1)[0]
    legacy = text[text.index("# 1) UWF 数据集"):]
    commands = [
        line
        for line in combined.splitlines()
        if line.startswith("CUDA_VISIBLE_DEVICES=")
        and " nohup python src/" in line
        and not line.lstrip().startswith("#")
    ]
    return combined, legacy, commands


def test_combined_training_matrix_has_one_command_per_configuration():
    combined, _, commands = _command_parts()

    assert len(commands) == 29
    assert sum("src/combiner_train.py" in line for line in commands) == 19
    assert sum("src/clip_fine_tune.py" in line for line in commands) == 10
    assert combined.count("# Phase A:") == 1
    assert combined.count("# Phase B:") == 1
    assert combined.count("# Phase C:") == 1


def test_combined_training_commands_are_isolated_single_gpu_nohup_jobs():
    _, _, commands = _command_parts()

    assert all(line.startswith("CUDA_VISIBLE_DEVICES=0 ") for line in commands)
    assert all(" nohup python " in line for line in commands)
    assert all(COMBINED_ROOT_ARGUMENT in line for line in commands)
    assert all("--dress-types Internal" in line for line in commands)
    assert all(line.endswith(" 2>&1 &") for line in commands)
    assert not any(re.search(r"run[2-9]_combined_", line) for line in commands)


def test_combined_training_commands_use_unique_log_files():
    _, _, commands = _command_parts()

    log_names = [re.search(r">\s+(\S+\.log)\s+2>&1", line).group(1) for line in commands]

    assert len(log_names) == 29
    assert len(set(log_names)) == 29
    assert all(name.startswith("run1_combined_") for name in log_names)


def test_existing_uwf_and_idrid_commands_remain_byte_for_byte_unchanged():
    _, legacy, _ = _command_parts()

    assert hashlib.sha256(legacy.encode()).hexdigest() == LEGACY_SUFFIX_SHA256
