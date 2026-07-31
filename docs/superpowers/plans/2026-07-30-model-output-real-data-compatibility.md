# Model Output Real-Data Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve metrics-only legacy runs and migrate same-named legacy reports without collisions so the guarded `models/` migration can proceed once all 24-hour failure windows expire.

**Architecture:** Keep the existing scanner, planner, and transactional mover. Extend classification so nonempty metrics are a retained, migratable run state, and derive report destinations from their paths relative to the legacy source root instead of flattening them to a basename.

**Tech Stack:** Python 3.9, `pathlib`, immutable dataclasses, pytest, existing `model_output_migration` CLI.

## Global Constraints

- A run is deletable only when it has no nonempty checkpoint, no nonempty metrics CSV, no live writer, and its newest modification is at least 24 hours old.
- Metrics-only runs must retain every ordinary file and use the normal dataset/stage/model/run target hierarchy.
- Legacy reports must retain their source-relative hierarchy below `outputs/reports/legacy/`.
- No destination may be overwritten or silently renamed.
- Dry-run must not create, move, link, or delete files.
- `--apply` remains all-or-nothing when active, unresolved, unknown, changed, or colliding inputs exist.

---

### Task 1: Retain metrics-only legacy runs

**Files:**
- Modify: `src/model_output_migration.py`
- Test: `tests/test_model_output_migration.py`

**Interfaces:**
- Consumes: `_classify_run(run: LegacyRun, *, now: datetime, pid_is_alive: Callable[[int], bool], legacy_writer_pids: Tuple[int, ...]) -> LegacyRun`
- Produces: metrics-only runs with `classification="valid"` and `reasons=("metrics-only-retained",)`; existing `build_migration_plan()` then creates normal file actions for every file in the run.

- [ ] **Step 1: Write the failing classification test**

Add a test that creates a legacy run with nonempty `train_metrics.csv`, a hyperparameter file, and no checkpoint:

```python
def test_metrics_only_run_is_retained_as_valid(tmp_path):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    run = source / "clip_finetuned_on_fiq_RN50x4_2026-01-01_00:00:00_000001"
    _write(run / "training_hyperparameters.json", b"{}")
    metrics = _write(run / "train_metrics.csv", b"epoch,loss\n1,0.5\n")

    scan = _scan(source, output)

    assert scan.runs[0].classification == "valid"
    assert scan.runs[0].reasons == ("metrics-only-retained",)
    plan = build_migration_plan(scan)
    assert any(action.source == metrics for action in plan.actions)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py::test_metrics_only_run_is_retained_as_valid -q
```

Expected: FAIL because the current classifier returns `unresolved` with `nonempty-metrics-without-checkpoint`.

- [ ] **Step 3: Implement the retained classification**

Replace the current nonempty-metrics branch in `_classify_run()` with:

```python
if nonempty_metrics:
    return replace(
        run,
        classification="valid",
        reasons=("metrics-only-retained",),
    )
```

Do not change the active-writer, corrupt-checkpoint, recent-run, or strict-failure branches.

- [ ] **Step 4: Verify the focused behavior**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
```

Expected: all migration tests pass, including the new metrics-only retention test and the existing strict-failure tests.

- [ ] **Step 5: Commit the classification change**

```bash
git add src/model_output_migration.py tests/test_model_output_migration.py
git commit -m "fix: retain metrics-only legacy runs"
```

### Task 2: Preserve relative paths for legacy reports

**Files:**
- Modify: `src/model_output_migration.py`
- Test: `tests/test_model_output_migration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `ScanResult.source_root`, `ScanResult.output_root`, and each report `Path`.
- Produces: `_legacy_report_destination(scan: ScanResult, report: Path) -> Path`, returning `scan.output_root / "reports" / "legacy" / report.relative_to(scan.source_root)`.

- [ ] **Step 1: Write the failing collision test**

Create two same-named Excel reports in distinct recognized legacy parent paths:

```python
def test_same_named_legacy_reports_keep_relative_paths(tmp_path):
    source = tmp_path / "models"
    output = tmp_path / "outputs"
    clip_report = _write(
        source / "clip_finetuned_on_fiq_ViT-B/summary.xlsx",
        b"clip",
    )
    combiner_report = _write(
        source / "combiner_trained_on_fiq_ViT-B/summary.xlsx",
        b"combiner",
    )

    plan = build_migration_plan(_scan(source, output))
    report_actions = {
        action.source: action
        for action in plan.actions
        if action.kind == "report"
    }

    assert report_actions[clip_report].destination == (
        output / "reports/legacy/clip_finetuned_on_fiq_ViT-B/summary.xlsx"
    )
    assert report_actions[combiner_report].destination == (
        output / "reports/legacy/combiner_trained_on_fiq_ViT-B/summary.xlsx"
    )
    assert all(action.status == "planned-move" for action in report_actions.values())
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py::test_same_named_legacy_reports_keep_relative_paths -q
```

Expected: FAIL because both reports currently flatten to `outputs/reports/legacy/summary.xlsx` and become collisions.

- [ ] **Step 3: Implement collision-free report destinations**

Add:

```python
def _legacy_report_destination(scan: ScanResult, report: Path) -> Path:
    relative = report.relative_to(scan.source_root)
    return scan.output_root / "reports" / "legacy" / relative
```

Use this helper in `build_migration_plan()` instead of `report.name`. Preserve the existing destination-exists check and collision counter.

- [ ] **Step 4: Update the README contract**

Change the migration description to state that recognized legacy reports are moved below `outputs/reports/legacy/` with their source-relative directory structure retained, so same-named reports are never overwritten.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest tests/test_model_output_migration.py -q
git diff --check
```

Expected: all migration tests pass and the whitespace check exits zero.

Commit:

```bash
git add src/model_output_migration.py tests/test_model_output_migration.py README.md
git commit -m "fix: preserve legacy report paths"
```

### Task 3: Verify code and repeat the real dry-run

**Files:**
- Verify: `src/model_output_migration.py`
- Verify: `scripts/organize_model_outputs.py`
- Verify: `tests/test_model_output_migration.py`

**Interfaces:**
- Consumes: the corrected classifier and report destination helper.
- Produces: a new read-only audit with zero report collisions, fourteen fewer unresolved runs, and no data mutation.

- [ ] **Step 1: Run task-focused tests and compilation**

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m pytest \
  tests/test_output_paths.py \
  tests/test_training_output_paths.py \
  tests/test_model_output_migration.py \
  tests/test_training_cli_dataset_root.py -q
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  -m compileall -q src scripts tests
git diff --check
```

Expected: all focused tests pass, compilation exits zero, and `git diff --check` reports nothing.

- [ ] **Step 2: Record the known unrelated full-suite result**

Run:

```bash
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python -m pytest -q
```

Expected while the user's uncommitted `命令.sh` GPU allocation remains present: only `tests/test_combined_commands.py::test_combined_training_commands_are_isolated_single_gpu_nohup_jobs` fails because it asserts every command uses GPU 0. Do not modify or revert `命令.sh`.

- [ ] **Step 3: Recheck live writers and run the real dry-run**

```bash
pgrep -af '(^|/)(python|python3).*src/(clip_fine_tune|combiner_train)\.py' || true
PYTHONPATH=src /data0/qrchen/miniconda3/envs/clip4cir/bin/python \
  scripts/organize_model_outputs.py
```

Expected before 2026-07-31 09:18:45+08:00:

- `active_runs: 0`
- `collisions: 0`
- `errors: 0`
- `valid_runs: 215` (201 checkpoint runs plus 14 metrics-only runs)
- `failed_runs: 2`
- `unresolved_runs: 2` for the two recent empty runs
- no `outputs/` directory is created

- [ ] **Step 4: Respect the remaining time gate**

If the two recent empty runs are still younger than 24 hours, stop before `--apply`; report their exact paths and eligibility time. Do not delete, rename, or modify them.

At or after 2026-07-31 09:18:45+08:00, rerun dry-run from a fresh source snapshot. Proceed to `--apply`, `--verify`, and `--finalize` only when `active_runs`, `unresolved_runs`, `collisions`, `errors`, and `unknown_paths` are all zero.

- [ ] **Step 5: Commit any verification-only documentation updates**

If no source or documentation changes were required during verification, do not create an empty commit. Otherwise stage only files owned by this task and commit with:

```bash
git commit -m "docs: record model migration compatibility audit"
```
