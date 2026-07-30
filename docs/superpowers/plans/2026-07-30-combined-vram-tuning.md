# Combined Internal VRAM Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adjust all 30 Combined Internal training commands for stable, higher-throughput use of the machine's 49,140 MiB RTX 4090 GPUs.

**Architecture:** Keep the change confined to the marker-delimited Combined section of `命令.sh`. Add a per-process DataLoader worker limit to every training command, tune frozen-backbone `clip-bs` separately from end-to-end fine-tuning `batch-size`, and preserve all existing optimizer, model, checkpoint, logging, and GPU-assignment settings.

**Tech Stack:** Bash, GNU awk/sed, Git.

## Global Constraints

- Modify only the section from `# COMBINED_COMMANDS_BEGIN` through `# COMBINED_COMMANDS_END` in `命令.sh`.
- Preserve the user's uncommitted GPU 1, 2, and 3 assignments.
- Treat 38–42 GiB as a safe peak ceiling with 15–20% headroom, not a mandatory occupancy target.
- Add `CLIP4CIR_NUM_WORKERS=8` to all 30 active training commands.
- Do not change learning rates, model dimensions, epochs, transforms, validation frequencies, checkpoints, save flags, or log filenames.
- Keep all Combined-external UWF and IDRiD content byte-for-byte unchanged.

---

### Task 1: Apply the approved parameter matrix

**Files:**
- Modify: `命令.sh:25-148`
- Reference: `docs/superpowers/specs/2026-07-30-combined-vram-tuning-design.md`

**Interfaces:**
- Consumes: The existing marker-delimited Combined command block and its 30 active `nohup python` commands.
- Produces: A marker-delimited block with the same commands, GPU assignments, and output logs, plus the approved batch matrix and worker cap.

- [ ] **Step 1: Capture the pre-change boundaries and confirm the expected failing state**

Run:

```bash
awk '
  /# COMBINED_COMMANDS_BEGIN/ {inside=1}
  inside && /nohup python src\\/(combiner_train|clip_fine_tune)\\.py/ {
    commands++
    if ($0 ~ /CLIP4CIR_NUM_WORKERS=8/) workers++
  }
  /# COMBINED_COMMANDS_END/ {inside=0}
  END {
    printf "commands=%d workers=%d\n", commands, workers
    exit !(commands == 30 && workers != 30)
  }
' 命令.sh
```

Expected: `commands=30 workers=0`, proving the worker-cap requirement is not yet implemented.

- [ ] **Step 2: Update the Combined header and all command prefixes**

Use `apply_patch` to replace the stale `GPU：全部使用 GPU 0` comment with:

```text
# GPU：GPU 0–7 均为 49,140 MiB；实际卡号以每条命令为准
```

Prefix every active Combined training command with:

```text
CLIP4CIR_NUM_WORKERS=8
```

The resulting prefix format must be:

```text
CLIP4CIR_NUM_WORKERS=8 CUDA_VISIBLE_DEVICES=N NCCL_P2P_DISABLE=1 nohup python ...
```

- [ ] **Step 3: Apply the Phase A frozen-backbone matrix**

Use `apply_patch` so Phase A has these exact `(batch-size, clip-bs)` pairs in command order:

```text
ViT-B/32                 (128, 128)
ViT-L/14                 (128, 64)
BMC_CLIP_CF              (128, 64)
RN50x4 Full FT           (128, 64)
RN50x4 No FT             (128, 64)
EyeCLIP                  (128, 128)
RetiZero                 (256, 128)
BLIP ITM Large COCO      (256, 32)
BLIP2 ITM ViT-G          (256, 16)
BLIP ITM Base COCO       (256, 64)
```

- [ ] **Step 4: Apply the Phase B fine-tuning matrix**

Use `apply_patch` so Phase B has these exact `batch-size` values in command order:

```text
ViT-B/32                 128
ViT-L/14                  64
BMC_CLIP_CF               64
RN50x4 Full FT             64
RN50x4 No FT               64
EyeCLIP                   128
RetiZero LoRA              32
BLIP ITM Large COCO        16
BLIP2 ITM ViT-G             4
BLIP ITM Base COCO         32
```

- [ ] **Step 5: Apply the Phase C frozen-backbone matrix**

Use `apply_patch` so Phase C has these exact `(batch-size, clip-bs)` pairs in command order:

```text
ViT-B/32 Fine-tuned       (128, 128)
ViT-L/14 Fine-tuned       (128, 64)
BMC_CLIP_CF Fine-tuned    (128, 64)
RN50x4 Full FT Fine-tuned (128, 64)
RN50x4 No FT Fine-tuned   (128, 64)
RetiZero LoRA Fine-tuned  (256, 128)
EyeCLIP Fine-tuned        (128, 128)
BLIP Fine-tuned           (256, 32)
BLIP2 ViT-G Fine-tuned    (256, 16)
BLIP Base Fine-tuned      (256, 64)
```

### Task 2: Verify syntax, scope, and exact parameters

**Files:**
- Verify: `命令.sh`

**Interfaces:**
- Consumes: The modified marker-delimited Combined block.
- Produces: Static evidence that the script is syntactically valid and matches the approved matrix without disturbing unrelated content.

- [ ] **Step 1: Check Bash syntax**

Run:

```bash
bash -n 命令.sh
```

Expected: exit status 0 with no output.

- [ ] **Step 2: Verify command and worker counts**

Run:

```bash
awk '
  /# COMBINED_COMMANDS_BEGIN/ {inside=1}
  inside && /nohup python src\\/(combiner_train|clip_fine_tune)\\.py/ {
    commands++
    if ($0 ~ /^CLIP4CIR_NUM_WORKERS=8 CUDA_VISIBLE_DEVICES=[0-7] NCCL_P2P_DISABLE=1 nohup python /) workers++
  }
  /# COMBINED_COMMANDS_END/ {inside=0}
  END {
    printf "commands=%d correctly_prefixed=%d\n", commands, workers
    exit !(commands == 30 && workers == 30)
  }
' 命令.sh
```

Expected: `commands=30 correctly_prefixed=30`.

- [ ] **Step 3: Print and compare the exact phase matrices**

Run:

```bash
awk '
  /# Phase A:/ {phase="A"; next}
  /# Phase B:/ {phase="B"; next}
  /# Phase C:/ {phase="C"; next}
  /# Combined 带真值/ {phase=""}
  phase && /^CLIP4CIR_NUM_WORKERS=8 / {
    batch=""; clip="n/a"
    for (i=1; i<=NF; i++) {
      if ($i=="--batch-size") batch=$(i+1)
      if ($i=="--clip-bs") clip=$(i+1)
    }
    printf "%s %s %s\n", phase, batch, clip
  }
' 命令.sh
```

Expected:

```text
A 128 128
A 128 64
A 128 64
A 128 64
A 128 64
A 128 128
A 256 128
A 256 32
A 256 16
A 256 64
B 128 n/a
B 64 n/a
B 64 n/a
B 64 n/a
B 64 n/a
B 128 n/a
B 32 n/a
B 16 n/a
B 4 n/a
B 32 n/a
C 128 128
C 128 64
C 128 64
C 128 64
C 128 64
C 256 128
C 128 128
C 256 32
C 256 16
C 256 64
```

- [ ] **Step 4: Verify scope and preservation of user changes**

Run:

```bash
git diff --check -- 命令.sh
git diff -- 命令.sh
```

Expected:

- No whitespace errors.
- Diff changes within lines 25–148 only.
- GPU assignments for Phase A ViT-L/14, BMC_CLIP_CF, and RN50x4 Full FT remain 1, 2, and 3.
- No UWF or IDRiD command changes appear.

- [ ] **Step 5: Report completion without committing the user's command file**

Do not commit `命令.sh` unless the user explicitly requests a commit. Report the modified file, exact validation results, and the retained user-owned GPU assignments.
