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


def _phase_commands(combined):
    phase_sections = {}
    for phase, next_phase in (("A", "B"), ("B", "C"), ("C", None)):
        start = combined.index(f"# Phase {phase}:")
        end = combined.index(f"# Phase {next_phase}:") if next_phase else len(combined)
        phase_sections[phase] = [
            line
            for line in combined[start:end].splitlines()
            if line.startswith("CUDA_VISIBLE_DEVICES=")
            and " nohup python src/" in line
        ]
    return phase_sections


def test_combined_training_matrix_has_one_command_per_configuration():
    combined, _, commands = _command_parts()
    phase_commands = _phase_commands(combined)

    assert len(commands) == 30
    assert sum("src/combiner_train.py" in line for line in commands) == 20
    assert sum("src/clip_fine_tune.py" in line for line in commands) == 10
    assert {phase: len(lines) for phase, lines in phase_commands.items()} == {
        "A": 10,
        "B": 10,
        "C": 10,
    }
    assert combined.count("# Phase A:") == 1
    assert combined.count("# Phase B:") == 1
    assert combined.count("# Phase C:") == 1
    assert not any("RETFound" in line for line in commands)


def test_eyeclip_and_retizero_appear_once_in_every_combined_phase():
    combined, _, _ = _command_parts()
    phase_commands = _phase_commands(combined)

    for model_tag in ("eyeclip", "retizero"):
        assert all(
            sum(model_tag in line.lower() for line in phase_commands[phase]) == 1
            for phase in ("A", "B", "C")
        )

    phase_a_eyeclip = next(
        line for line in phase_commands["A"] if "eyeclip" in line.lower()
    )
    assert "pretrained_models/eyeclip_clip4cir_vitb32.pt" in phase_a_eyeclip

    phase_b_retizero = next(
        line for line in phase_commands["B"] if "retizero" in line.lower()
    )
    assert "src/clip_fine_tune.py" in phase_b_retizero
    assert "--clip-model-name RetiZero" in phase_b_retizero
    assert "--retizero-base-path " in phase_b_retizero


def test_combined_preflight_uses_runnable_retrieval_checkpoints():
    combined, _, _ = _command_parts()
    phase_commands = _phase_commands(combined)

    assert "blip2-flan-t5-xxl" not in combined.lower()
    assert "clip_RN50x4_noft.pt" not in combined

    for phase in ("A", "B"):
        noft_command = next(
            line
            for line in phase_commands[phase]
            if "rn50x4_noft" in line.lower()
        )
        assert "--clip-model-name RN50x4" in noft_command
        assert "--clip-model-path" not in noft_command

        blip_large_command = next(
            line
            for line in phase_commands[phase]
            if "run1_combined_blip_" in line
        )
        assert (
            "--clip-model-path "
            "/data0/qrchen/projects/CLIP4Cir/pretrained_models/"
            "blip_itm_large_coco/pytorch_model.bin"
        ) in blip_large_command

    for phase in ("A", "B", "C"):
        base_commands = [
            line
            for line in phase_commands[phase]
            if "blip_itm_base_coco" in line.lower()
        ]
        assert len(base_commands) == 1
        assert "--clip-model-name BLIP" in base_commands[0]
        assert "--blip-projection-dim 256" in base_commands[0]
        assert "--blip-input-resolution 384" in base_commands[0]

    phase_c_base = next(
        line
        for line in phase_commands["C"]
        if "blip_itm_base_coco" in line.lower()
    )
    assert (
        "pretrained_models/Combined/tuned_blip_itm_base_coco_best.pt"
        in phase_c_base
    )


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

    assert len(log_names) == 30
    assert len(set(log_names)) == 30
    assert all(name.startswith("run1_combined_") for name in log_names)


def test_existing_uwf_and_idrid_commands_remain_byte_for_byte_unchanged():
    _, legacy, _ = _command_parts()

    assert hashlib.sha256(legacy.encode()).hexdigest() == LEGACY_SUFFIX_SHA256


def test_combined_validation_templates_route_outputs_by_actual_dataset():
    combined, _, _ = _command_parts()
    validation_commands = [
        line
        for line in combined.splitlines()
        if line.startswith("# CUDA_VISIBLE_DEVICES=")
        and "python src/validate.py " in line
    ]

    assert len(validation_commands) == 3
    assert all(
        "--output-dataset combined-fundus-cir" in line
        for line in validation_commands
    )
    assert {
        re.search(
            r"--evaluation-name\s+(\S+)",
            line,
        ).group(1)
        for line in validation_commands
    } == {"internal-test", "odir5k-test", "grape-test"}
    assert all(
        "eval_combined_" not in line
        for line in validation_commands
    )
    assert all(
        line.endswith("> /dev/null 2>&1 &")
        for line in validation_commands
    )
