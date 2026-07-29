# Combined Command Preflight Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all Combined Phase A and Phase B commands runnable and keep the Phase C model matrix aligned at 10×3.

**Architecture:** Keep the change command-driven: correct invalid checkpoint arguments and replace the generative-only FLAN entry with a verified retrieval model. Preserve the existing adapters, training code, legacy command suffix, process-launch convention, and dataset configuration.

**Tech Stack:** Bash command matrix, Python 3.9, pytest, Transformers BLIP adapter, PyTorch

## Global Constraints

- Combined 每种配置只运行 1 次。
- 全部命令使用 GPU 0、`nohup`、独立日志文件。
- 数据集固定为 `Combined_Fundus_CIR_Dataset` 的 `Internal` 类别。
- 原有 UWF 与 IDRiD 命令逐字节不变。
- 不自动启动任何训练任务。

---

### Task 1: Encode the corrected command contract

**Files:**
- Modify: `tests/test_combined_commands.py`
- Test: `tests/test_combined_commands.py`

**Interfaces:**
- Consumes: Combined block delimited by `COMBINED_COMMANDS_BEGIN` and `COMBINED_COMMANDS_END`.
- Produces: Regression assertions for RN50x4 No FT, BLIP ITM Large, and BLIP ITM Base.

- [ ] **Step 1: Write a failing regression test**

Add assertions that Phase A/B No FT commands omit `--clip-model-path`, Phase A/B BLIP Large commands use `blip_itm_large_coco/pytorch_model.bin`, FLAN is absent, and `blip_itm_base_coco` occurs once per phase.

- [ ] **Step 2: Run the focused test and verify failure**

Run: `PYTHONPATH=src python -m pytest tests/test_combined_commands.py -q`

Expected: FAIL because the current command block still uses the invalid No FT checkpoint and FLAN entry.

- [ ] **Step 3: Commit the failing test**

Run: `git add tests/test_combined_commands.py && git commit -m "test: cover Combined command preflight requirements"`

### Task 2: Correct the 10×3 command matrix

**Files:**
- Modify: `命令.sh`
- Test: `tests/test_combined_commands.py`

**Interfaces:**
- Consumes: Stable BLIP model directory `/data0/qrchen/projects/CLIP4Cir/pretrained_models/blip_itm_base_coco`.
- Produces: Thirty Combined commands with ten valid retrieval backbones in every phase.

- [ ] **Step 1: Apply the minimal command changes**

Remove the No FT checkpoint arguments in Phase A/B, change BLIP Large Phase A/B checkpoint arguments to `pytorch_model.bin`, and replace all three FLAN entries with BLIP ITM Base commands.

- [ ] **Step 2: Run the focused tests**

Run: `PYTHONPATH=src python -m pytest tests/test_combined_commands.py -q`

Expected: `6 passed`.

- [ ] **Step 3: Run shell syntax and CLI parser checks**

Run: `bash -n 命令.sh`

Parse every Phase A/B command with the corresponding Python entry point using `--help`-compatible argument extraction without starting training.

- [ ] **Step 4: Commit the implementation**

Run: `git add 命令.sh && git commit -m "fix: make Combined training commands runnable"`

### Task 3: Provision and verify BLIP ITM Base

**Files:**
- Create ignored symlink: `pretrained_models/blip_itm_base_coco`
- Verify: `src/blip_adapter.py`

**Interfaces:**
- Consumes: Local Hugging Face snapshot `models--Salesforce--blip-itm-base-coco`.
- Produces: Stable model path used by all three BLIP ITM Base commands.

- [ ] **Step 1: Create the stable local model link**

Create `pretrained_models/blip_itm_base_coco` pointing to the complete local snapshot. Do not add this ignored model artifact to Git.

- [ ] **Step 2: Verify offline model loading**

Instantiate `BLIPAdapter` from the stable path with `local_files_only` behavior and verify one image and one text produce finite `[1, 256]` features with pretrained projection heads.

- [ ] **Step 3: Run the complete test suite**

Run: `PYTHONPATH=src python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Commit documentation**

Run: `git add docs/superpowers/specs/2026-07-30-combined-preflight-fixes-design.md docs/superpowers/plans/2026-07-30-combined-preflight-fixes.md && git commit -m "docs: record Combined preflight fixes"`

### Task 4: Integrate into main

**Files:**
- Verify: repository state

**Interfaces:**
- Consumes: Green `fix/combined-preflight` branch.
- Produces: Locally merged `main` with preserved user files.

- [ ] **Step 1: Review the branch diff and run final verification**

Confirm only the command tests, Combined command block, design, and plan changed. Re-run the full suite from a clean branch state.

- [ ] **Step 2: Merge locally**

Merge `fix/combined-preflight` into `main` without pulling, because local `main` intentionally contains unpublished commits.

- [ ] **Step 3: Verify the merged result**

Run the full suite on `main`, confirm the stable BLIP link resolves, and confirm the two pre-existing untracked user files remain untouched.
