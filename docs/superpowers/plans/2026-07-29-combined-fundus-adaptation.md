# Combined Fundus CIR Dataset Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add isolated, explicit support for `Combined_Fundus_CIR_Dataset`, including Internal training/validation, labeled Internal/ODIR5K/GRAPE test evaluation, and a 29-command single-run GPU-0 experiment section at the top of `命令.sh`.

**Architecture:** Extend `FashionIQDataset` with an optional explicit root and opt-in labeled-test return mode while preserving all legacy defaults. Thread the explicit root through both training CLIs and the validation CLI, and reuse the existing FashionIQ retrieval metric path for labeled test splits. Add lightweight tests around filesystem resolution, dataset tuples, CLI plumbing, retrieval metrics, and the shell command matrix without loading large model weights.

**Tech Stack:** Python 3, PyTorch, Pillow, pytest, argparse, Bash, existing CLIP4Cir modules.

## Global Constraints

- Combined root is exactly `/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset`.
- Train and validation category is exactly `Internal`.
- Labeled test categories are exactly `Internal`, `ODIR5K`, and `GRAPE`.
- Existing FashionIQ, UWF, and IDRiD default behavior must remain available.
- Existing UWF and IDRiD command counts, parameters, and GPU assignments must not change.
- The new Combined section contains exactly 29 training commands: 8 pretrained-backbone Combiner, 10 backbone fine-tune, and 11 fine-tuned-backbone Combiner.
- Each Combined configuration appears once, uses `CUDA_VISIBLE_DEVICES=0`, and retains `nohup ... > log 2>&1 &`.
- Dependent fine-tuned Combiner commands are documented as a later stage and must not be presented as safe to launch before checkpoint preparation.

---

### Task 1: Explicit FashionIQ root and labeled-test dataset mode

**Files:**
- Modify: `src/data_utils.py`
- Create: `tests/test_combined_data_utils.py`

**Interfaces:**
- Consumes: Existing FashionIQ-style `captions/`, `image_splits/`, and `images/` layout.
- Produces: `resolve_fashioniq_root(category=None, required_split=None, dataset_root=None) -> Path`.
- Produces: `list_fashioniq_categories(split="train", dataset_root=None) -> List[str]`.
- Produces: `FashionIQDataset(..., dataset_root=None, return_target=False)`.

- [ ] **Step 1: Write failing tests for explicit-root isolation**

Create temporary roots containing disjoint category files and assert that:

```python
assert resolve_fashioniq_root("Internal", "train", explicit_root) == explicit_root
assert list_fashioniq_categories("test", explicit_root) == ["GRAPE", "Internal", "ODIR5K"]
```

Also set legacy root environment variables to other temporary roots so the test proves an explicit root cannot leak categories from them.

- [ ] **Step 2: Run the explicit-root tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_combined_data_utils.py -k explicit_root
```

Expected: failure because the functions do not yet accept `dataset_root`.

- [ ] **Step 3: Implement explicit-root resolution**

Add an optional `dataset_root` argument. If present, normalize it with `Path(...).expanduser().resolve()`, validate required directories and requested split files, and use only that root. Keep the existing environment/default candidate search when it is absent.

- [ ] **Step 4: Run the explicit-root tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_combined_data_utils.py -k explicit_root
```

Expected: all selected tests pass.

- [ ] **Step 5: Write failing tests for labeled test tuples**

Build a minimal test category with two real RGB images, one classic split, and one caption record containing `candidate`, `target`, and `captions`. Assert:

```python
legacy_item = FashionIQDataset("test", ["Internal"], "relative", transform,
                               dataset_root=root)[0]
labeled_item = FashionIQDataset("test", ["Internal"], "relative", transform,
                                dataset_root=root, return_target=True)[0]
assert len(legacy_item) == 3
assert labeled_item == ("reference", "target", "change")
```

Add failure cases for a missing `target` key and a target absent from the physical image gallery.

- [ ] **Step 6: Run labeled-test tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_combined_data_utils.py -k labeled
```

Expected: failure because `return_target` does not exist.

- [ ] **Step 7: Implement labeled-test behavior**

Pass `dataset_root` into root resolution for every category. Preserve the current test tuple when `return_target=False`; when `split=="test"` and `return_target=True`, require a valid physical target and return `(reference_name, target_name, caption)`, matching val query semantics.

- [ ] **Step 8: Run the complete data utility test file**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_combined_data_utils.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/data_utils.py tests/test_combined_data_utils.py
git commit -m "feat: support isolated labeled FashionIQ-style datasets"
```

### Task 2: Thread the explicit root through both training entry points

**Files:**
- Modify: `src/clip_fine_tune.py`
- Modify: `src/combiner_train.py`
- Create: `tests/test_training_cli_dataset_root.py`

**Interfaces:**
- Consumes: `FashionIQDataset(..., dataset_root=...)` from Task 1.
- Produces: `--fashioniq-root PATH` in both training CLIs.
- Produces: `fashioniq_root: Optional[str]` in the corresponding FashionIQ training functions.

- [ ] **Step 1: Write failing CLI help tests**

Use subprocesses with the project environment to run:

```python
for script in ("src/clip_fine_tune.py", "src/combiner_train.py"):
    result = subprocess.run([sys.executable, script, "--help"], ...)
    assert result.returncode == 0
    assert "--fashioniq-root" in result.stdout
```

- [ ] **Step 2: Run the CLI tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_training_cli_dataset_root.py
```

Expected: failure because the option is absent.

- [ ] **Step 3: Add and propagate `--fashioniq-root`**

Add:

```python
parser.add_argument(
    "--fashioniq-root",
    type=str,
    default=None,
    help="Explicit root for a FashionIQ-style dataset; overrides root auto-discovery",
)
```

Include it in each `training_hyper_params` dictionary and pass it to every FashionIQ train/val dataset construction. Do not apply it to CIRR.

- [ ] **Step 4: Run CLI and syntax checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_training_cli_dataset_root.py
python -m py_compile src/data_utils.py src/clip_fine_tune.py src/combiner_train.py
```

Expected: tests pass and compilation exits zero.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/clip_fine_tune.py src/combiner_train.py tests/test_training_cli_dataset_root.py
git commit -m "feat: expose FashionIQ dataset root in training CLIs"
```

### Task 3: Labeled val/test retrieval evaluation

**Files:**
- Modify: `src/validate.py`
- Create: `src/fashioniq_evaluation.py`
- Create: `tests/test_fashioniq_evaluation.py`
- Create: `tests/test_validate_cli.py`

**Interfaces:**
- Consumes: labeled query tuples `(reference_name, target_name, caption)` and gallery index names/features.
- Produces: `compute_recall_at_k(predicted_features, target_names, index_features, index_names, ks=(1, 5, 10)) -> Tuple[float, ...]`.
- Produces: validation CLI options `--fashioniq-root`, `--dress-types`, and `--fashioniq-split {val,test}`.

- [ ] **Step 1: Write failing pure recall tests**

Use a three-item orthogonal gallery and query vectors with known ranks:

```python
recalls = compute_recall_at_k(
    predicted_features=torch.tensor([[1., 0.], [1., 0.]]),
    target_names=["a", "b"],
    index_features=torch.tensor([[1., 0.], [.8, .2], [0., 1.]]),
    index_names=["a", "b", "c"],
    ks=(1, 2, 3),
)
assert recalls == (50.0, 100.0, 100.0)
```

Add assertions for duplicate/missing gallery names and mismatched query/target lengths.

- [ ] **Step 2: Run recall tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_fashioniq_evaluation.py
```

Expected: import failure because the helper module does not exist.

- [ ] **Step 3: Implement the pure metric helper**

Normalize both feature matrices, compute cosine distance, sort gallery indices, map every target name to exactly one gallery index, and return percentage recall for each requested `k`. Raise `ValueError` with the offending name/count for invalid inputs.

- [ ] **Step 4: Run recall tests and verify GREEN**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_fashioniq_evaluation.py
```

Expected: all tests pass.

- [ ] **Step 5: Write failing validation CLI tests**

Assert `python src/validate.py --help` advertises:

```text
--fashioniq-root
--dress-types
--fashioniq-split
```

Also assert the parser accepts `--fashioniq-split test` and rejects unsupported split values through a subprocess invocation that stops before model loading by using `--help`/parser-level coverage.

- [ ] **Step 6: Run validation CLI tests and verify RED**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_validate_cli.py
```

Expected: failure because the arguments are absent.

- [ ] **Step 7: Generalize FashionIQ validation**

Update the FashionIQ evaluation path so:

- `val` constructs normal relative datasets;
- `test` constructs `FashionIQDataset(..., return_target=True)`;
- classic gallery and relative query use the selected split;
- explicit `dress_types` bypass global category discovery;
- omitted `dress_types` preserves current val category discovery;
- metrics are produced per category and averaged exactly as R@1/R@5/R@10.

Refactor existing distance/rank calculation to call `compute_recall_at_k` without changing its existing percentage semantics.

- [ ] **Step 8: Run validation tests and syntax checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_fashioniq_evaluation.py tests/test_validate_cli.py
python -m py_compile src/validate.py src/fashioniq_evaluation.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/validate.py src/fashioniq_evaluation.py tests/test_fashioniq_evaluation.py tests/test_validate_cli.py
git commit -m "feat: evaluate labeled FashionIQ-style test splits"
```

### Task 4: Add the single-run Combined experiment matrix

**Files:**
- Modify: `命令.sh`
- Create: `tests/test_combined_commands.py`

**Interfaces:**
- Consumes: `--fashioniq-root` from Tasks 2 and 3.
- Produces: a delimited shell section between `COMBINED_COMMANDS_BEGIN` and `COMBINED_COMMANDS_END`.

- [ ] **Step 1: Write failing command-matrix tests**

Parse only the delimited Combined section and assert:

```python
assert len(training_commands) == 29
assert sum("src/combiner_train.py" in line for line in training_commands) == 19
assert sum("src/clip_fine_tune.py" in line for line in training_commands) == 10
assert all(line.startswith("CUDA_VISIBLE_DEVICES=0 ") for line in training_commands)
assert all(" nohup python " in line for line in training_commands)
assert all("--fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset" in line
           for line in training_commands)
assert all("--dress-types Internal" in line for line in training_commands)
assert len({extract_log_name(line) for line in training_commands}) == 29
```

Snapshot the existing UWF/IDRiD section before editing and assert it is byte-for-byte unchanged after the Combined delimiter.

- [ ] **Step 2: Run command tests and verify RED**

Run:

```bash
pytest -q tests/test_combined_commands.py
```

Expected: failure because the delimiters and Combined section do not exist.

- [ ] **Step 3: Add the Combined section at the top of `命令.sh`**

Place it after environment preparation and before UWF. Use the same model arguments as the corresponding IDRiD configuration, replacing:

- category with `Internal`;
- root with the Combined path;
- checkpoint namespace with `/data0/qrchen/projects/CLIP4Cir/pretrained_models/Combined/`;
- log prefix with `run1_combined_`;
- GPU with `0`;
- repetitions with exactly one command.

Organize the section as:

```text
0.1 pretrained backbone + Combiner (8)
0.2 backbone fine-tune (10)
0.3 fine-tuned backbone + Combiner (11)
0.4 labeled test evaluation examples (Internal/ODIR5K/GRAPE)
```

Keep evaluation examples commented so executing selected training commands does not unexpectedly launch evaluation.

- [ ] **Step 4: Run command tests and shell syntax validation**

Run:

```bash
pytest -q tests/test_combined_commands.py
bash -n 命令.sh
```

Expected: tests pass and Bash syntax validation exits zero.

- [ ] **Step 5: Commit Task 4**

```bash
git add 命令.sh tests/test_combined_commands.py
git commit -m "feat: add single-run Combined experiment commands"
```

### Task 5: Verify against the real Combined dataset and full regression suite

**Files:**
- Modify only if verification exposes a defect in files already covered above.

**Interfaces:**
- Consumes: Real Combined dataset and all preceding changes.
- Produces: Evidence that the adaptation meets every design criterion without launching training.

- [ ] **Step 1: Run the full lightweight test suite**

```bash
PYTHONPATH=src pytest -q tests
```

Expected: zero failures.

- [ ] **Step 2: Run Python and shell syntax checks**

```bash
python -m py_compile src/data_utils.py src/fashioniq_evaluation.py src/validate.py src/clip_fine_tune.py src/combiner_train.py
bash -n 命令.sh
```

Expected: both commands exit zero.

- [ ] **Step 3: Instantiate every real Combined split**

Run a small Python check using an identity/PIL transform and the explicit root. Assert:

```text
Internal train relative: 19393
Internal train classic: 3920
Internal val relative: 436
Internal val classic: 311
Internal test relative: 1278
Internal test classic: 741
ODIR5K test relative: 3958
ODIR5K test classic: 3368
GRAPE test relative: 74
GRAPE test classic: 82
```

For every test relative dataset, instantiate it with `return_target=True`, sample first/last items, and verify both referenced files exist.

- [ ] **Step 4: Audit the exact Combined command matrix**

Print and check:

```text
29 training commands
19 combiner commands
10 fine-tune commands
29 unique log files
29 GPU-0 assignments
29 explicit Combined roots
29 Internal training categories
```

- [ ] **Step 5: Inspect final diff and user-owned files**

Run:

```bash
git status --short
git diff --check
git diff HEAD~4 -- src/data_utils.py src/fashioniq_evaluation.py src/validate.py src/clip_fine_tune.py src/combiner_train.py tests 命令.sh
```

Confirm unrelated untracked user files remain untouched.

- [ ] **Step 6: Commit verification-only fixes if necessary**

If verification required an implementation correction, commit only the affected project/test files:

```bash
git add <affected-project-files>
git commit -m "fix: complete Combined dataset verification"
```

If no correction was needed, do not create an empty commit.
