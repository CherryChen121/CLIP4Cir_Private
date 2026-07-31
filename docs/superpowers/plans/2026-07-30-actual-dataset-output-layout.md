# Actual Dataset Output Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate existing and future FashionIQ-format training outputs by their actual IDRiD, UWF, Combined Fundus CIR, or native FashionIQ dataset identity.

**Architecture:** A focused `dataset_identity` module resolves the actual dataset before any run directory is created, while `output_paths` remains responsible only for path and manifest creation. A separate transactional reclassification module scans historical evidence, snapshots every file, moves complete run directories through a same-filesystem staging area, updates manifests and audit records recoverably, and exposes dry-run/apply/verify/finalize operations through a small CLI.

**Tech Stack:** Python 3, standard-library `argparse`/`csv`/`dataclasses`/`hashlib`/`json`/`pathlib`, existing PyTorch training entrypoints, and pytest.

## Global Constraints

- Actual dataset slugs are exactly `idrid`, `uwf`, `combined-fundus-cir`, and `fashioniq`; FashionIQ-format data records `dataset_format` as `fashioniq`.
- Dataset resolution priority is explicit `--output-dataset`, then the resolved FashionIQ root, then dress types.
- Without an explicit override, conflicting or unknown evidence must raise before a run directory is created; there is no silent `fashioniq` fallback.
- CIRR training remains under `outputs/cirr/` with `dataset_format` equal to `cirr`.
- Historical classification uses retained hyperparameters and validation CSV headers only; dates, model names, and directory names are not classification evidence.
- Existing unresolved classifications, symlinks inside run directories, target collisions, active writers, cross-filesystem moves, or changed source snapshots block the entire apply.
- Moves use same-filesystem atomic renames through `outputs/.dataset-reclassify-staging`; no target is overwritten.
- Ordinary files are verified by size and SHA-256 before and after movement.
- The five approved legacy Excel files are deleted only during finalize after a successful applied-state verification, using their five exact audited paths.
- Directory cleanup uses empty-directory removal only; recursive forced deletion is forbidden.
- Preserve `outputs/migration_manifest.csv` and `outputs/migration_report.json`, including old paths and original checksums.
- Do not modify or stage the user's existing `代码修改说明.md`, `命令.sh`, or `IDRiD平均召回率汇总.xlsx` worktree changes.

---

## File Structure

- Create `src/dataset_identity.py`: immutable dataset identity model and future-training evidence resolution.
- Modify `src/output_paths.py`: accept identity metadata and write it into `run_manifest.json`.
- Modify `src/clip_fine_tune.py`: expose `--output-dataset`, resolve the actual FashionIQ-format root/identity, and preserve CIRR behavior.
- Modify `src/combiner_train.py`: expose `--output-dataset`, resolve the actual FashionIQ-format root/identity, and preserve CIRR behavior.
- Create `src/output_dataset_reclassification.py`: historical evidence classification, snapshots, planning, transactional movement, audit updates, rollback, verification, and exact cleanup.
- Create `scripts/reclassify_output_datasets.py`: dry-run/apply/verify/finalize command-line interface.
- Create `tests/test_dataset_identity.py`: identity priority, root, category, conflict, unknown, and CIRR tests.
- Modify `tests/test_output_paths.py`: manifest identity-field tests.
- Modify `tests/test_training_output_paths.py`: both training CLIs and source integration checks.
- Modify `tests/test_training_cli_dataset_root.py`: explicit output dataset/root help coverage.
- Create `tests/test_output_dataset_reclassification.py`: classification, safety gate, transaction, rollback, audit, verification, and finalize tests.
- Modify `README.md`: actual-dataset layout, new training options, and reclassification operations.

### Task 1: Resolve Actual Dataset Identity Before Output Creation

**Files:**
- Create: `src/dataset_identity.py`
- Create: `tests/test_dataset_identity.py`

**Interfaces:**
- Consumes: `output_paths.slugify_component(value: str) -> str`.
- Produces: `DatasetIdentity`, `DatasetIdentityError`, and `resolve_dataset_identity(*, dataset_format, dress_types, dataset_root_requested, dataset_root_resolved, output_dataset) -> DatasetIdentity`.

- [ ] **Step 1: Write failing tests for every supported identity and resolution priority**

```python
# tests/test_dataset_identity.py
from pathlib import Path

import pytest

from dataset_identity import (
    DatasetIdentityError,
    resolve_dataset_identity,
    resolve_fashioniq_training_identity,
)


@pytest.mark.parametrize(
    ("root_name", "dress_types", "expected"),
    [
        ("IDRiD_CIR_Dataset_cold", ["IDRiD"], "idrid"),
        ("UWF_CIR_Dataset_cold", ["CH", "CO", "NM", "RB", "RCH", "UM"], "uwf"),
        ("Combined_Fundus_CIR_Dataset", ["Internal"], "combined-fundus-cir"),
        ("fashionIQ_dataset", ["dress", "shirt", "toptee"], "fashioniq"),
    ],
)
def test_resolves_supported_fashioniq_format_datasets(
    tmp_path, root_name, dress_types, expected
):
    root = tmp_path / root_name
    root.mkdir()
    identity = resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=dress_types,
        dataset_root_requested=str(root),
        dataset_root_resolved=root,
        output_dataset=None,
    )
    assert identity.dataset_slug == expected
    assert identity.dataset_format == "fashioniq"


def test_explicit_output_dataset_wins_and_is_slugified(tmp_path):
    root = tmp_path / "unrecognized"
    root.mkdir()
    identity = resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=["unknown"],
        dataset_root_requested="unrecognized",
        dataset_root_resolved=root,
        output_dataset=" My Study / Phase 1 ",
    )
    assert identity.dataset_slug == "my-study-phase-1"
    assert "explicit-output-dataset" in identity.classification_evidence


def test_conflicting_root_and_categories_are_rejected(tmp_path):
    root = tmp_path / "IDRiD_CIR_Dataset_cold"
    root.mkdir()
    with pytest.raises(DatasetIdentityError, match="conflicting"):
        resolve_dataset_identity(
            dataset_format="fashioniq",
            dress_types=["CH", "CO", "NM", "RB", "RCH", "UM"],
            dataset_root_requested=str(root),
            dataset_root_resolved=root,
            output_dataset=None,
        )


def test_unknown_automatic_evidence_is_rejected_before_output(tmp_path):
    root = tmp_path / "mystery"
    root.mkdir()
    with pytest.raises(DatasetIdentityError, match="could not identify"):
        resolve_dataset_identity(
            dataset_format="fashioniq",
            dress_types=["unknown"],
            dataset_root_requested=None,
            dataset_root_resolved=root,
            output_dataset=None,
        )
    assert not (tmp_path / "outputs").exists()


def test_cirr_identity_does_not_require_fashioniq_evidence():
    identity = resolve_dataset_identity(
        dataset_format="cirr",
        dress_types=(),
        dataset_root_requested=None,
        dataset_root_resolved=None,
        output_dataset=None,
    )
    assert identity.dataset_slug == "cirr"
    assert identity.dataset_format == "cirr"


def test_training_identity_records_requested_symlink_and_resolved_target(
    tmp_path
):
    target = tmp_path / "IDRiD_CIR_Dataset_cold"
    target.mkdir()
    link = tmp_path / "fashionIQ_dataset"
    link.symlink_to(target, target_is_directory=True)
    identity = resolve_fashioniq_training_identity(
        project_root=tmp_path,
        dress_types=["IDRiD"],
        dataset_root=None,
        output_dataset=None,
        root_resolver=lambda *_args: link,
    )
    assert identity.dataset_slug == "idrid"
    assert identity.root_requested == str(link)
    assert identity.root_resolved == str(target.resolve())
```

- [ ] **Step 2: Run the identity tests and confirm the new module is missing**

Run: `PYTHONPATH=src pytest -q tests/test_dataset_identity.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'dataset_identity'`.

- [ ] **Step 3: Implement the immutable identity model and deterministic evidence resolver**

```python
# src/dataset_identity.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from output_paths import slugify_component


IDRID_TYPES = frozenset({"idrid"})
UWF_TYPES = frozenset({"ch", "co", "nm", "rb", "rch", "um"})
FASHIONIQ_TYPES = frozenset({"dress", "shirt", "toptee"})


class DatasetIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_slug: str
    dataset_format: str
    root_requested: Optional[str]
    root_resolved: Optional[str]
    classification_evidence: Tuple[str, ...]

    def layout_fields(self) -> dict:
        return {
            "dataset": self.dataset_slug,
            "dataset_format": self.dataset_format,
            "dataset_root_requested": self.root_requested,
            "dataset_root_resolved": self.root_resolved,
            "dataset_classification_evidence": list(
                self.classification_evidence
            ),
        }


def resolve_dataset_identity(
    *,
    dataset_format: str,
    dress_types: Sequence[str],
    dataset_root_requested: Optional[Union[str, Path]],
    dataset_root_resolved: Optional[Union[str, Path]],
    output_dataset: Optional[str],
) -> DatasetIdentity:
    normalized_format = slugify_component(dataset_format)
    requested = (
        str(Path(dataset_root_requested).expanduser())
        if dataset_root_requested is not None
        else None
    )
    resolved = (
        str(Path(dataset_root_resolved).expanduser().resolve())
        if dataset_root_resolved is not None
        else None
    )
    if output_dataset is not None:
        return DatasetIdentity(
            slugify_component(output_dataset),
            normalized_format,
            requested,
            resolved,
            ("explicit-output-dataset",),
        )
    if normalized_format == "cirr":
        return DatasetIdentity(
            "cirr", "cirr", requested, resolved, ("dataset-format:cirr",)
        )

    category_set = frozenset(value.casefold() for value in dress_types)
    category_slug = _classify_categories(category_set)
    root_slug = _classify_root(resolved)
    evidence = tuple(
        value
        for value in (
            f"resolved-root:{resolved}" if root_slug else None,
            f"dress-types:{','.join(sorted(category_set))}"
            if category_slug
            else None,
        )
        if value is not None
    )
    candidates = {value for value in (root_slug, category_slug) if value}
    if len(candidates) > 1:
        raise DatasetIdentityError(
            f"conflicting dataset evidence: root={root_slug}, "
            f"dress_types={category_slug}"
        )
    if not candidates:
        raise DatasetIdentityError(
            "could not identify actual dataset from root or dress types"
        )
    return DatasetIdentity(
        candidates.pop(), "fashioniq", requested, resolved, evidence
    )
```

Implement `_classify_categories` so only the exact normalized sets above map to their slugs, except `{"internal"}` maps to `combined-fundus-cir`. Implement `_classify_root` using case-insensitive path-component/name matching for `IDRiD_CIR_Dataset_cold`, `UWF_CIR_Dataset_cold`, `Combined_Fundus_CIR_Dataset`, and native `fashionIQ_dataset`; do not use substring matches that classify an unrelated parent directory.

- [ ] **Step 4: Run focused identity tests**

Run: `PYTHONPATH=src pytest -q tests/test_dataset_identity.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the identity resolver**

```bash
git add src/dataset_identity.py tests/test_dataset_identity.py
git commit -m "feat: resolve actual training dataset identity"
```

### Task 2: Record Identity and Integrate Both Training Entrypoints

**Files:**
- Modify: `src/output_paths.py:45-91`
- Modify: `src/clip_fine_tune.py:369-405,650-680,875-1015`
- Modify: `src/combiner_train.py:240-263,767-800,1033-1169`
- Modify: `tests/test_output_paths.py:39-89`
- Modify: `tests/test_training_output_paths.py:27-49`
- Modify: `tests/test_training_cli_dataset_root.py:12-33`

**Interfaces:**
- Consumes: `DatasetIdentity.layout_fields()` and `resolve_dataset_identity(*, dataset_format, dress_types, dataset_root_requested, dataset_root_resolved, output_dataset)`.
- Produces: identity-aware `create_run_layout` and `resolve_fashioniq_training_identity` calls in both training paths.

- [ ] **Step 1: Extend output-layout tests with the required manifest metadata**

```python
# tests/test_output_paths.py
def test_create_run_layout_records_actual_dataset_identity(tmp_path):
    layout = create_run_layout(
        project_root=tmp_path,
        output_root=None,
        dataset="combined-fundus-cir",
        dataset_format="fashioniq",
        dataset_root_requested="/datasets/combined",
        dataset_root_resolved="/datasets/combined",
        dataset_classification_evidence=("resolved-root:combined",),
        stage="combiner",
        model_name="ViT-L/14",
        started_at=datetime(2026, 7, 30, 9, 51, 41),
        pid=2393301,
    )
    payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert layout.root.parts[-5] == "combined-fundus-cir"
    assert payload["dataset"] == "combined-fundus-cir"
    assert payload["dataset_format"] == "fashioniq"
    assert payload["dataset_root_requested"] == "/datasets/combined"
    assert payload["dataset_root_resolved"] == "/datasets/combined"
    assert payload["dataset_classification_evidence"] == [
        "resolved-root:combined"
    ]
```

Also update `test_create_run_layout_writes_manifest_and_sanitizes_model` to pass `dataset_format="fashioniq"` and assert that `dataset` and `dataset_slug` are both the normalized actual slug.

- [ ] **Step 2: Extend CLI and source-integration tests**

```python
# tests/test_training_output_paths.py
def test_training_clis_expose_actual_dataset_controls():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    for script in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "--output-root" in result.stdout
        assert "--output-dataset" in result.stdout


def test_fashioniq_training_paths_do_not_hardcode_fashioniq_dataset():
    for relative in ("src/clip_fine_tune.py", "src/combiner_train.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert 'dataset="fashioniq"' not in source
        assert "resolve_dataset_identity(" in source
```

Add a help assertion for `--output-dataset` to each parameterized case in `tests/test_training_cli_dataset_root.py`.

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run: `PYTHONPATH=src pytest -q tests/test_output_paths.py tests/test_training_output_paths.py tests/test_training_cli_dataset_root.py`

Expected: failures report unsupported identity keyword arguments and missing `--output-dataset`.

- [ ] **Step 4: Extend `create_run_layout` without changing its path-safety rules**

```python
# src/output_paths.py
def create_run_layout(
    *,
    project_root: Path,
    output_root: Optional[Union[str, Path]],
    dataset: str,
    dataset_format: str,
    stage: str,
    model_name: str,
    dataset_root_requested: Optional[str] = None,
    dataset_root_resolved: Optional[str] = None,
    dataset_classification_evidence: tuple[str, ...] = (),
    started_at: Optional[datetime] = None,
    pid: Optional[int] = None,
) -> RunLayout:
    if stage not in VALID_STAGES:
        raise ValueError(
            f"invalid training stage {stage!r}; expected one of "
            f"{sorted(VALID_STAGES)}"
        )
    started_at = started_at or datetime.now()
    pid = os.getpid() if pid is None else pid
    dataset_slug = slugify_component(dataset)
    model_slug = slugify_component(model_name)
    run_id = build_run_id(started_at, pid)
    root = (
        resolve_output_root(project_root, output_root)
        / dataset_slug / stage / model_slug / run_id
    )
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    manifest = root / "run_manifest.json"
    temporary_manifest = root / ".run_manifest.json.tmp"
    payload = {
        "dataset": dataset_slug,
        "dataset_slug": dataset_slug,
        "dataset_format": slugify_component(dataset_format),
        "dataset_root_requested": dataset_root_requested,
        "dataset_root_resolved": dataset_root_resolved,
        "dataset_classification_evidence": list(
            dataset_classification_evidence
        ),
        "training_stage": stage,
        "model_name": model_name,
        "model_slug": model_slug,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "pid": pid,
        "checkpoint_dir": checkpoints.name,
    }
    temporary_manifest.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest)
    return RunLayout(root, checkpoints, manifest, run_id)
```

Update every call site, including CIRR, to pass `dataset_format`; CIRR passes `dataset="cirr"` and `dataset_format="cirr"` with no root evidence.

- [ ] **Step 5: Resolve the same effective root used by the FashionIQ data loader**

Before either FashionIQ training function calls `create_run_layout`, resolve each selected category with the existing loader. Put this exact shared helper in `dataset_identity.py`:

```python
def resolve_fashioniq_training_identity(
    *,
    project_root: Path,
    dress_types: Sequence[str],
    dataset_root: Optional[Union[str, Path]],
    output_dataset: Optional[str],
    root_resolver: Callable[
        [str, str, Optional[Union[str, Path]]], Path
    ],
) -> DatasetIdentity:
    resolved_roots = {
        root_resolver(category, "train", dataset_root).resolve()
        for category in dress_types
    }
    if len(resolved_roots) != 1:
        raise DatasetIdentityError(
            "FashionIQ categories resolved to multiple dataset roots: "
            + ", ".join(sorted(str(path) for path in resolved_roots))
        )
    resolved_root = resolved_roots.pop()
    requested_root = dataset_root
    project_link = project_root / "fashionIQ_dataset"
    if requested_root is None and (
        project_link.exists() or project_link.is_symlink()
    ):
        if project_link.resolve() == resolved_root:
            requested_root = str(project_link)
    return resolve_dataset_identity(
        dataset_format="fashioniq",
        dress_types=dress_types,
        dataset_root_requested=requested_root,
        dataset_root_resolved=resolved_root,
        output_dataset=output_dataset,
    )
```

Import `Callable` in `dataset_identity.py` and `resolve_fashioniq_root` in each training entrypoint.

After resolution, put `identity.root_resolved` back into `training_hyper_params["fashioniq_root"]`; this guarantees manifest classification and dataset loading use the same directory.

- [ ] **Step 6: Add `--output-dataset` and pass identity fields before directory creation**

```python
parser.add_argument(
    "--output-dataset",
    type=str,
    default=None,
    help=(
        "Actual dataset name for output paths; overrides root/category "
        "auto-detection"
    ),
)
```

Store `output_dataset` in `training_hyper_params`. In both FashionIQ functions, resolve identity first and then create the layout:

```python
identity = resolve_fashioniq_training_identity(
    project_root=base_path,
    dress_types=train_dress_types,
    dataset_root=kwargs.get("fashioniq_root"),
    output_dataset=kwargs.get("output_dataset"),
    root_resolver=resolve_fashioniq_root,
)
layout = create_run_layout(
    project_root=base_path,
    output_root=kwargs.get("output_root"),
    stage="combiner",  # "clip-finetune" in clip_fine_tune.py
    model_name=clip_model_name,
    **identity.layout_fields(),
)
```

Resolve identity before `create_run_layout`; no directory may exist if resolution raises. Keep the existing CIRR branch independent of FashionIQ root/category resolution.

- [ ] **Step 7: Add entrypoint smoke tests proving both stages route the resolved identity**

```python
# tests/test_training_output_paths.py
@pytest.mark.parametrize(
    ("module_name", "function_name", "stage"),
    [
        ("clip_fine_tune", "clip_finetune_fiq", "clip-finetune"),
        ("combiner_train", "combiner_training_fiq", "combiner"),
    ],
)
def test_fashioniq_entrypoint_routes_identity_before_model_loading(
    monkeypatch, module_name, function_name, stage
):
    module = importlib.import_module(module_name)
    identity = DatasetIdentity(
        dataset_slug="idrid",
        dataset_format="fashioniq",
        root_requested="/datasets/idrid",
        root_resolved="/datasets/idrid",
        classification_evidence=("test",),
    )
    captured = {}

    class LayoutCaptured(RuntimeError):
        pass

    monkeypatch.setattr(
        module, "resolve_fashioniq_training_identity",
        lambda **_kwargs: identity,
    )

    def capture_layout(**kwargs):
        captured.update(kwargs)
        raise LayoutCaptured

    monkeypatch.setattr(module, "create_run_layout", capture_layout)
    common = {
        "train_dress_types": ["IDRiD"],
        "val_dress_types": ["IDRiD"],
        "num_epochs": 1,
        "clip_model_name": "ViT-B/32",
        "batch_size": 1,
        "validation_frequency": 1,
        "transform": "clip",
        "save_training": False,
        "save_best": False,
        "output_root": None,
        "fashioniq_root": "/datasets/idrid",
        "output_dataset": None,
    }
    if stage == "clip-finetune":
        common.update(learning_rate=1e-6, encoder="both")
    else:
        common.update(
            projection_dim=64, hidden_dim=128, combiner_lr=1e-5, clip_bs=1
        )
    with pytest.raises(LayoutCaptured):
        getattr(module, function_name)(**common)
    assert captured["dataset"] == "idrid"
    assert captured["dataset_format"] == "fashioniq"
    assert captured["stage"] == stage
```

- [ ] **Step 8: Run the focused suite**

Run: `PYTHONPATH=src pytest -q tests/test_dataset_identity.py tests/test_output_paths.py tests/test_training_output_paths.py tests/test_training_cli_dataset_root.py`

Expected: all tests pass, and both `python src/clip_fine_tune.py --help` and `python src/combiner_train.py --help` list `--output-dataset`.

- [ ] **Step 9: Commit future-output integration**

```bash
git add src/dataset_identity.py src/output_paths.py \
  src/clip_fine_tune.py src/combiner_train.py \
  tests/test_dataset_identity.py tests/test_output_paths.py \
  tests/test_training_output_paths.py tests/test_training_cli_dataset_root.py
git commit -m "feat: route training outputs by actual dataset"
```

### Task 3: Classify and Plan Historical Run Reclassification

**Files:**
- Create: `src/output_dataset_reclassification.py`
- Create: `tests/test_output_dataset_reclassification.py`

**Interfaces:**
- Consumes: `DatasetIdentity`, `DatasetIdentityError`, and `model_output_migration.sha256_file`.
- Produces: `RunClassification`, `FileSnapshot`, `ReclassificationAction`, `ReclassificationPlan`, `classify_run(run_root: Path)`, and `build_reclassification_plan(output_root: Path)`.

- [ ] **Step 1: Write fixtures and failing classification tests**

```python
# tests/test_output_dataset_reclassification.py
import csv
import json
from pathlib import Path

import pytest

from output_dataset_reclassification import (
    ReclassificationBlockedError,
    build_reclassification_plan,
    classify_run,
)


def make_run(
    output_root: Path,
    stage: str,
    run_id: str,
    *,
    hyperparameters=None,
    validation_headers=(),
):
    run = output_root / "fashioniq" / stage / "vitb32" / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/model.pt").write_bytes(run_id.encode())
    (run / "train_metrics.csv").write_text("epoch,loss\n1,0.1\n")
    with (run / "validation_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("epoch", *validation_headers))
        writer.writerow((1, *(0.5 for _ in validation_headers)))
    if hyperparameters is not None:
        (run / "training_hyperparameters.json").write_text(
            json.dumps(hyperparameters), encoding="utf-8"
        )
    (run / "run_manifest.json").write_text(
        json.dumps({"dataset": "fashioniq"}), encoding="utf-8"
    )
    return run


def test_classifies_combiner_from_hyperparameters_and_root(tmp_path):
    run = make_run(
        tmp_path,
        "combiner",
        "combined",
        hyperparameters={
            "train_dress_types": ["Internal"],
            "val_dress_types": ["Internal"],
            "fashioniq_root":
                "/data0/qrchen/datasets/Combined_Fundus_CIR_Dataset",
        },
        validation_headers=("Internal_recall_at1",),
    )
    result = classify_run(run)
    assert result.dataset_slug == "combined-fundus-cir"
    assert "training_hyperparameters:train_dress_types=Internal" in (
        result.evidence
    )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("IDRiD_recall_at1", "idrid"),
        ("CH_recall_at1", "uwf"),
        ("Internal_recall_at1", "combined-fundus-cir"),
    ],
)
def test_classifies_clip_finetune_from_validation_header(
    tmp_path, header, expected
):
    run = make_run(
        tmp_path, "clip-finetune", expected, validation_headers=(header,)
    )
    assert classify_run(run).dataset_slug == expected
```

- [ ] **Step 2: Add failing planning safety tests**

```python
def test_plan_blocks_unknown_evidence(tmp_path):
    make_run(tmp_path, "clip-finetune", "unknown", validation_headers=("loss",))
    with pytest.raises(ReclassificationBlockedError, match="unresolved"):
        build_reclassification_plan(tmp_path)


def test_plan_blocks_target_collision(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "idrid-run",
        validation_headers=("IDRiD_recall_at1",),
    )
    target = (
        tmp_path / "idrid" / "clip-finetune" / "vitb32" / run.name
    )
    target.mkdir(parents=True)
    with pytest.raises(ReclassificationBlockedError, match="collision"):
        build_reclassification_plan(tmp_path)


def test_plan_rejects_symlinks_inside_run(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "linked",
        validation_headers=("IDRiD_recall_at1",),
    )
    (run / "outside-link").symlink_to(tmp_path / "outside")
    with pytest.raises(ReclassificationBlockedError, match="symlink"):
        build_reclassification_plan(tmp_path)


def test_plan_rejects_cross_filesystem_action(tmp_path, monkeypatch):
    make_run(
        tmp_path,
        "clip-finetune",
        "cross-device",
        validation_headers=("IDRiD_recall_at1",),
    )
    monkeypatch.setattr(
        "output_dataset_reclassification._device_id",
        lambda path: 2 if "fashioniq" in path.parts else 1,
    )
    with pytest.raises(ReclassificationBlockedError, match="filesystem"):
        build_reclassification_plan(tmp_path)


def test_dry_plan_does_not_modify_files(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "stable",
        validation_headers=("IDRiD_recall_at1",),
    )
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    plan = build_reclassification_plan(tmp_path)
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert len(plan.actions) == 1
    assert before == after
```

- [ ] **Step 3: Run the new tests and confirm the module is missing**

Run: `PYTHONPATH=src pytest -q tests/test_output_dataset_reclassification.py`

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 4: Implement historical evidence parsing with exact accepted sets**

```python
# src/output_dataset_reclassification.py
@dataclass(frozen=True)
class RunClassification:
    dataset_slug: str
    dataset_format: str
    root_requested: Optional[str]
    root_resolved: Optional[str]
    evidence: Tuple[str, ...]


def classify_run(run_root: Path) -> RunClassification:
    hyperparameters_path = run_root / "training_hyperparameters.json"
    validation_path = run_root / "validation_metrics.csv"
    hyperparameters = (
        json.loads(hyperparameters_path.read_text(encoding="utf-8"))
        if hyperparameters_path.exists()
        else {}
    )
    with validation_path.open(newline="", encoding="utf-8-sig") as file:
        headers = next(csv.reader(file))

    category_evidence = _classify_historical_categories(
        hyperparameters.get("train_dress_types")
    )
    metric_evidence = _classify_metric_headers(headers)
    root_evidence = _classify_historical_root(
        hyperparameters.get("fashioniq_root")
    )
    candidates = {
        value
        for value in (category_evidence, metric_evidence, root_evidence)
        if value is not None
    }
    if len(candidates) != 1:
        raise ReclassificationBlockedError(
            f"unresolved or conflicting evidence for {run_root}: "
            f"categories={category_evidence}, metrics={metric_evidence}, "
            f"root={root_evidence}"
        )
    dataset_slug = candidates.pop()
    requested_root = hyperparameters.get("fashioniq_root")
    resolved_root = (
        str(Path(requested_root).expanduser().resolve())
        if requested_root
        else None
    )
    evidence = []
    if hyperparameters.get("train_dress_types"):
        evidence.append(
            "training_hyperparameters:train_dress_types="
            + ",".join(hyperparameters["train_dress_types"])
        )
    evidence.extend(
        f"validation_metrics:{header}"
        for header in headers
        if _metric_header_dataset(header) == dataset_slug
    )
    if requested_root:
        evidence.append(
            f"training_hyperparameters:fashioniq_root={requested_root}"
        )
    return RunClassification(
        dataset_slug=dataset_slug,
        dataset_format="fashioniq",
        root_requested=requested_root,
        root_resolved=resolved_root,
        evidence=tuple(evidence),
    )
```

Define `_classify_historical_categories(values) -> Optional[str]`, `_metric_header_dataset(header) -> Optional[str]`, `_classify_metric_headers(headers) -> Optional[str]`, and `_classify_historical_root(value) -> Optional[str]` in this step. They use the exact sets from Task 1. Require `Internal` to agree with the Combined root when a root is present. IDRiD metrics are columns beginning `IDRiD_recall_`; UWF accepts the six observed medical prefixes and rejects a mixture with another dataset; native FashionIQ accepts `dress`, `shirt`, and `toptee`. Evidence strings must name the source file and exact field/header used.

- [ ] **Step 5: Implement immutable snapshots and an all-or-nothing plan**

```python
@dataclass(frozen=True)
class FileSnapshot:
    relative_path: Path
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class ReclassificationAction:
    source: Path
    staging: Path
    destination: Path
    classification: RunClassification
    files: Tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class ReclassificationPlan:
    output_root: Path
    actions: Tuple[ReclassificationAction, ...]
    unresolved: Tuple[str, ...]
    collisions: Tuple[str, ...]

    @property
    def dataset_counts(self) -> dict[str, int]:
        return dict(Counter(
            action.classification.dataset_slug for action in self.actions
        ))


def build_reclassification_plan(output_root: Path) -> ReclassificationPlan:
    output_root = output_root.resolve()
    source_root = output_root / "fashioniq"
    staging_root = output_root / ".dataset-reclassify-staging"
    if not source_root.exists():
        return ReclassificationPlan(output_root, (), (), ())
    run_roots = _discover_exact_run_roots(source_root)
    actions = []
    unresolved = []
    collisions = []
    for source in run_roots:
        try:
            _reject_symlinks_and_special_files(source)
            classification = classify_run(source)
        except ReclassificationBlockedError as error:
            unresolved.append(str(error))
            continue
        relative = source.relative_to(source_root)
        stage, model_slug, run_id = relative.parts
        destination = (
            output_root / classification.dataset_slug
            / stage / model_slug / run_id
        )
        staging = (
            staging_root / classification.dataset_slug
            / stage / model_slug / run_id
        )
        if destination.exists() or staging.exists():
            collisions.append(str(destination))
            continue
        files = tuple(
            _snapshot_file(path, source)
            for path in sorted(source.rglob("*"))
            if path.is_file()
        )
        actions.append(ReclassificationAction(
            source, staging, destination, classification, files
        ))
    if unresolved or collisions:
        raise ReclassificationBlockedError(
            "reclassification plan blocked; unresolved="
            + repr(unresolved) + "; collisions=" + repr(collisions)
        )
    return ReclassificationPlan(
        output_root, tuple(actions), (), ()
    )
```

Define `_discover_exact_run_roots(source_root) -> Tuple[Path, ...]` to accept only `source_root/<stage>/<model>/<run>` where stage is in `VALID_STAGES`, every run contains `run_manifest.json`, and no unexpected shallower/deeper peer exists. Define `_reject_symlinks_and_special_files(run_root) -> None` to use `lstat`, reject any symlink, and accept only directories or ordinary files. Define `_snapshot_file(path, run_root) -> FileSnapshot` to capture relative path, `st_size`, `st_mtime_ns`, and `sha256_file(path)`. Define `_device_id(path: Path) -> int` as `path.stat().st_dev`, then require `_device_id(source) == _device_id(output_root)` for every action. A missing `outputs/fashioniq` returns an empty plan, enabling the final zero-pending dry-run.

- [ ] **Step 6: Run classification and planning tests**

Run: `PYTHONPATH=src pytest -q tests/test_output_dataset_reclassification.py -k 'classif or plan or symlink or dry'`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the read-only historical planner**

```bash
git add src/output_dataset_reclassification.py \
  tests/test_output_dataset_reclassification.py
git commit -m "feat: plan output dataset reclassification"
```

### Task 4: Apply, Roll Back, Update Audits, and Verify Recoverably

**Files:**
- Modify: `src/output_dataset_reclassification.py`
- Modify: `tests/test_output_dataset_reclassification.py`

**Interfaces:**
- Consumes: `ReclassificationPlan` and its immutable file snapshots.
- Produces: `find_output_writer_pids(project_root, output_root, proc_root)`, `apply_reclassification(plan)`, `verify_reclassification(output_root, expected_plan, finalized)`, and `VerificationResult`.

- [ ] **Step 1: Write failing tests for active writers and source mutation**

```python
def test_apply_blocks_active_training_writer(tmp_path, monkeypatch):
    make_run(
        tmp_path,
        "combiner",
        "active",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=("IDRiD_recall_at1",),
    )
    plan = build_reclassification_plan(tmp_path)
    monkeypatch.setattr(
        "output_dataset_reclassification.find_output_writer_pids",
        lambda *_args, **_kwargs: (1234,),
    )
    with pytest.raises(ReclassificationBlockedError, match="1234"):
        apply_reclassification(plan)
    assert (tmp_path / "fashioniq").exists()


def test_apply_blocks_file_changed_after_snapshot(tmp_path):
    run = make_run(
        tmp_path,
        "clip-finetune",
        "changed",
        validation_headers=("IDRiD_recall_at1",),
    )
    plan = build_reclassification_plan(tmp_path)
    (run / "train_metrics.csv").write_text("epoch,loss\n1,9.9\n")
    with pytest.raises(SourceChangedError, match="train_metrics.csv"):
        apply_reclassification(plan)
```

- [ ] **Step 2: Write failing tests for rollback and atomic metadata updates**

```python
def test_apply_rolls_back_all_runs_when_second_move_fails(
    tmp_path, monkeypatch
):
    first = make_run(
        tmp_path, "clip-finetune", "first",
        validation_headers=("IDRiD_recall_at1",)
    )
    second = make_run(
        tmp_path, "clip-finetune", "second",
        validation_headers=("IDRiD_recall_at1",)
    )
    plan = build_reclassification_plan(tmp_path)
    real_replace = Path.replace
    calls = 0

    def fail_second_final_move(path, target):
        nonlocal calls
        if ".dataset-reclassify-staging" in str(path):
            calls += 1
            if calls == 2:
                raise OSError("injected move failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_final_move)
    with pytest.raises(TransactionError, match="rolled back"):
        apply_reclassification(plan)
    assert first.exists()
    assert second.exists()
    assert not (tmp_path / "idrid").exists()


def test_apply_updates_run_and_top_level_audits(tmp_path):
    run = make_run(
        tmp_path,
        "combiner",
        "idrid",
        hyperparameters={"train_dress_types": ["IDRiD"]},
        validation_headers=("IDRiD_recall_at1",),
    )
    write_audit_fixture(tmp_path, run)
    plan = build_reclassification_plan(tmp_path)
    apply_reclassification(plan)
    target = tmp_path / "idrid/combiner/vitb32/idrid"
    manifest = json.loads(
        (target / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["dataset"] == "idrid"
    assert manifest["dataset_format"] == "fashioniq"
    rows = list(csv.DictReader(
        (tmp_path / "migration_manifest.csv").open()
    ))
    assert rows[0]["dataset"] == "idrid"
    assert "/outputs/idrid/" in rows[0]["new_path"]
```

`write_audit_fixture` must create the exact current CSV columns:
`old_path,new_path,dataset,stage,model_slug,run_id,status,size,sha256,duplicate_group,canonical,reason`
and a JSON report containing the existing counters.

- [ ] **Step 3: Run transactional tests and confirm missing interfaces**

Run: `PYTHONPATH=src pytest -q tests/test_output_dataset_reclassification.py -k 'active or changed or rolls_back or audits'`

Expected: failures report undefined apply/verification interfaces.

- [ ] **Step 4: Implement writer detection and pre-apply snapshot revalidation**

```python
TRAINING_SCRIPTS = frozenset({"clip_fine_tune.py", "combiner_train.py"})


def find_output_writer_pids(
    project_root: Path,
    output_root: Path,
    proc_root: Path = Path("/proc"),
) -> Tuple[int, ...]:
    matches = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        decoded = [item.decode(errors="replace") for item in argv if item]
        script_indexes = [
            index for index, arg in enumerate(decoded)
            if Path(arg).name in TRAINING_SCRIPTS
        ]
        if not script_indexes:
            continue
        try:
            cwd = (process / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        script = Path(decoded[script_indexes[0]])
        script = script if script.is_absolute() else cwd / script
        if script.resolve().parent != project_root.resolve() / "src":
            continue
        dataset = _argv_option(decoded, "--dataset")
        if dataset is not None and dataset.casefold() == "cirr":
            continue
        requested_output = _argv_option(decoded, "--output-root")
        effective_output = resolve_output_root(
            project_root, requested_output
        )
        if effective_output == output_root.resolve():
            matches.append(int(process.name))
    return tuple(sorted(matches))


def _verify_snapshot(root: Path, snapshot: FileSnapshot) -> None:
    path = root / snapshot.relative_path
    stat = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise SourceChangedError(f"source type changed: {path}")
    if stat.st_size != snapshot.size or stat.st_mtime_ns != snapshot.mtime_ns:
        raise SourceChangedError(f"source metadata changed: {path}")
    if sha256_file(path) != snapshot.sha256:
        raise SourceChangedError(f"source hash changed: {path}")
```

Define `_argv_option(argv: Sequence[str], name: str) -> Optional[str]` to support both `--name value` and `--name=value`; a missing `--dataset` remains conservatively blocking. Import `resolve_output_root` from `output_paths`. Call writer detection and `_verify_snapshot` for all actions immediately before creating staging directories.

- [ ] **Step 5: Implement a journaled two-phase rename and rollback**

```python
def apply_reclassification(plan: ReclassificationPlan) -> None:
    staging_root = plan.output_root / ".dataset-reclassify-staging"
    journal_path = staging_root / "transaction.json"
    moved_to_staging = []
    moved_to_final = []
    try:
        staging_root.mkdir(exist_ok=False)
        _write_json_atomic(
            journal_path,
            {"state": "prepared", "actions": _journal_actions(plan)},
        )
        for action in plan.actions:
            action.staging.parent.mkdir(parents=True, exist_ok=True)
            action.source.replace(action.staging)
            moved_to_staging.append(action)
        _verify_actions(plan.actions, location="staging")
        _write_json_atomic(
            journal_path,
            {"state": "staged", "actions": _journal_actions(plan)},
        )
        for action in plan.actions:
            action.destination.parent.mkdir(parents=True, exist_ok=True)
            action.staging.replace(action.destination)
            moved_to_final.append(action)
        _verify_actions(plan.actions, location="destination")
        _update_run_manifests_and_audits(plan, staging_root)
        _write_json_atomic(
            journal_path,
            {"state": "applied", "actions": _journal_actions(plan)},
        )
    except Exception as error:
        rollback_errors = _rollback_actions(
            plan, moved_to_staging, moved_to_final
        )
        if rollback_errors:
            raise TransactionError(
                f"apply failed and rollback was incomplete; preserve "
                f"{staging_root}: {rollback_errors}"
            ) from error
        raise TransactionError("apply failed and was rolled back") from error
```

The journal contains only paths, dataset identities, and transaction state; snapshots remain in the in-memory plan. Parent directories may be created, but rollback removes only empty directories with `rmdir`. If rollback is incomplete, retain staging and the journal and perform no cleanup.

- [ ] **Step 6: Implement recoverable manifest and audit rewrites**

```python
def _updated_run_manifest(action: ReclassificationAction) -> dict:
    path = action.destination / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({
        "dataset": action.classification.dataset_slug,
        "dataset_slug": action.classification.dataset_slug,
        "dataset_format": "fashioniq",
        "dataset_root_requested": action.classification.root_requested,
        "dataset_root_resolved": action.classification.root_resolved,
        "dataset_classification_evidence":
            list(action.classification.evidence),
    })
    return payload
```

Before rewriting, copy the original 215 manifest bytes and both audit files into `staging_root/metadata-backup/`. Write each replacement to a sibling temporary file, `fsync` the file, and use `Path.replace`. For each successful audit row whose `new_path` begins with one action's source, replace that prefix with its destination and set `dataset` to the classification slug. Preserve `old_path`, `size`, `sha256`, duplicate fields, and all failed-run rows exactly. Add to `migration_report.json`:

```json
{
  "actual_dataset_run_counts": {
    "idrid": 172,
    "uwf": 41,
    "combined-fundus-cir": 2
  },
  "reclassification_state": "applied"
}
```

If any metadata write fails, restore every backed-up byte before moving runs back. Remove `.tmp` files only by exact name.

- [ ] **Step 7: Implement applied-state verification**

```python
@dataclass(frozen=True)
class VerificationResult:
    run_counts: dict[str, int]
    checkpoint_count: int
    retained_audit_files: int
    errors: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def verify_reclassification(
    output_root: Path,
    *,
    expected_plan: Optional[ReclassificationPlan] = None,
    finalized: bool = False,
) -> VerificationResult:
    # Verify every planned destination snapshot except run_manifest.json
    # when a plan is supplied; that generated file is intentionally updated.
    # Otherwise reconstruct expected destinations from migration_manifest.csv.
    # Verify path/dataset agreement and required fields in all 215 manifests.
    # Verify every moved/deduplicated audit destination by size and sha256.
    # For deleted-approved-report, require destination absence.
    # In applied state, reports may still exist; in finalized state they may not.
```

The verifier treats the 215 generated `run_manifest.json` files separately from the original migration actions because they are not rows in `migration_manifest.csv`. It verifies 782 retained successful audit rows after finalize, 5 approved-deletion rows, and preserves the 2 failed-run history rows.

- [ ] **Step 8: Run all reclassification unit tests**

Run: `PYTHONPATH=src pytest -q tests/test_output_dataset_reclassification.py`

Expected: all tests pass.

- [ ] **Step 9: Commit transactional apply and verification**

```bash
git add src/output_dataset_reclassification.py \
  tests/test_output_dataset_reclassification.py
git commit -m "feat: apply dataset reclassification transactionally"
```

### Task 5: Add the Operational CLI, Finalize Cleanup, and Documentation

**Files:**
- Create: `scripts/reclassify_output_datasets.py`
- Modify: `src/output_dataset_reclassification.py`
- Modify: `tests/test_output_dataset_reclassification.py`
- Modify: `README.md:197-220,256-292`

**Interfaces:**
- Consumes: planner, apply, and verifier from Tasks 3-4.
- Produces: `finalize_reclassification(output_root, expected_plan)` and a dry-run-by-default CLI with mutually exclusive `--apply`, `--verify`, and `--finalize`.

- [ ] **Step 1: Write failing CLI-mode and finalize tests**

```python
def test_cli_defaults_to_read_only_dry_run(tmp_path):
    make_run(
        tmp_path,
        "clip-finetune",
        "dry",
        validation_headers=("IDRiD_recall_at1",),
    )
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "dry-run"
    assert payload["dataset_counts"] == {"idrid": 1}
    assert (tmp_path / "fashioniq").exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()


def test_finalize_deletes_only_the_five_audited_reports(tmp_path):
    plan = make_applied_fixture_with_five_reports(tmp_path)
    result = finalize_reclassification(tmp_path, expected_plan=plan)
    assert result.ok
    assert not (tmp_path / "reports").exists()
    assert not (tmp_path / "fashioniq").exists()
    assert not (tmp_path / ".dataset-reclassify-staging").exists()
    rows = list(csv.DictReader(
        (tmp_path / "migration_manifest.csv").open()
    ))
    deleted = [
        row for row in rows
        if row["status"] == "deleted-approved-report"
    ]
    assert len(deleted) == 5
    assert {row["reason"] for row in deleted} == {
        "user-approved-obsolete-legacy-summary"
    }
```

Add a negative test with a sixth unaudited file under `reports/legacy`; finalize must raise and delete nothing.

- [ ] **Step 2: Run the new mode/finalize tests and confirm they fail**

Run: `PYTHONPATH=src pytest -q tests/test_output_dataset_reclassification.py -k 'cli or finalize'`

Expected: failures report missing CLI and `finalize_reclassification`.

- [ ] **Step 3: Implement finalize as a verified, exact-path operation**

```python
APPROVED_REPORT_STATUS = "deleted-approved-report"
APPROVED_REPORT_REASON = "user-approved-obsolete-legacy-summary"


def finalize_reclassification(
    output_root: Path,
    *,
    expected_plan: Optional[ReclassificationPlan] = None,
) -> VerificationResult:
    verification = verify_reclassification(
        output_root, expected_plan=expected_plan, finalized=False
    )
    if not verification.ok:
        raise ReclassificationBlockedError(
            "applied-state verification failed: "
            + "; ".join(verification.errors)
        )
    rows = _read_migration_manifest(output_root)
    report_rows = [
        row for row in rows
        if row["new_path"].endswith(".xlsx")
        and "/outputs/reports/legacy/" in row["new_path"]
        and row["status"] == "moved"
    ]
    if len(report_rows) != 5:
        raise ReclassificationBlockedError(
            f"expected exactly 5 approved reports, found {len(report_rows)}"
        )
    _verify_no_unplanned_report_files(output_root, report_rows)
    for row in report_rows:
        path = Path(row["new_path"])
        _verify_audit_hash(path, row)
    quarantined = _move_reports_to_exact_staging_paths(
        output_root, report_rows
    )
    try:
        for row in report_rows:
            row["status"] = APPROVED_REPORT_STATUS
            row["reason"] = APPROVED_REPORT_REASON
        _write_updated_audits_atomically(rows, state="finalized")
        result = verify_reclassification(
            output_root, expected_plan=expected_plan, finalized=True
        )
        if not result.ok:
            raise TransactionError(
                "final verification failed: " + "; ".join(result.errors)
            )
    except Exception:
        _restore_applied_audits_and_reports(output_root, quarantined)
        raise
    _unlink_exact_quarantined_reports(quarantined)
    _unlink_exact_transaction_artifacts(output_root)
    _rmdir_exact_empty_tree(output_root)
    return result
```

The five reports first move by exact `Path.replace` calls into `.dataset-reclassify-staging/approved-report-deletions/`, making audit-write failure recoverable. Only after finalized-state verification succeeds may their five exact quarantine paths and the enumerated transaction journal/metadata backups be unlinked. `_rmdir_exact_empty_tree` may call `rmdir` only for exact leaf-to-root paths under `.dataset-reclassify-staging`, `reports/legacy`, `reports`, and `fashioniq`; it must refuse if any other entry remains. Update `migration_report.json` status counts so the 5 rows move from `moved` to `deleted-approved-report`, set `reclassification_state` to `finalized`, and retain all earlier counters.

- [ ] **Step 4: Implement the CLI with JSON output and nonzero blocked exits**

```python
#!/usr/bin/env python3
from argparse import ArgumentParser
import json
from pathlib import Path

from output_dataset_reclassification import (
    apply_reclassification,
    build_reclassification_plan,
    finalize_reclassification,
    verify_reclassification,
)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    # No mode: build and print a dry-run plan only.
    # --apply: build, apply, verify, and print applied counts.
    # --verify: verify current applied/finalized state without writes.
    # --finalize: verify, delete exact reports, update audits, and verify again.
    # Print one JSON object to stdout; blocked/error details go to stderr.
```

Use `sys.path.insert(0, str(PROJECT_ROOT / "src"))` following the existing scripts if the project has no installed package. Exit `0` only for a clean operation/verification, and `1` for blocked or failed states.

- [ ] **Step 5: Document the new layout and concrete commands**

Replace FashionIQ-only examples under `outputs/fashioniq/` with actual dataset paths. Add:

```markdown
outputs/<actual-dataset>/<clip-finetune|combiner>/<model>/<run-id>/
```

Document these examples:

```bash
python src/combiner_train.py \
  --dataset FashionIQ \
  --fashioniq-root /data0/qrchen/datasets/Combined_Fundus_CIR_Dataset \
  --dress-types Internal

python src/clip_fine_tune.py \
  --dataset FashionIQ \
  --fashioniq-root /path/to/custom-fashioniq-format-data \
  --dress-types custom \
  --output-dataset my-study
```

State that `--output-dataset` is an explicit actual-dataset override, not a data-loader format selector. Document:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py
PYTHONPATH=src python scripts/reclassify_output_datasets.py --apply
PYTHONPATH=src python scripts/reclassify_output_datasets.py --verify
PYTHONPATH=src python scripts/reclassify_output_datasets.py --finalize
```

- [ ] **Step 6: Run focused and full tests**

Run: `PYTHONPATH=src pytest -q tests/test_dataset_identity.py tests/test_output_paths.py tests/test_training_output_paths.py tests/test_training_cli_dataset_root.py tests/test_output_dataset_reclassification.py`

Expected: all focused tests pass.

Run: `PYTHONPATH=src pytest -q`

Expected: the clean-tree baseline passes. If the user's modified `命令.sh` still triggers the known GPU-distribution assertion, record that separately and confirm no new failure is introduced by these files.

- [ ] **Step 7: Commit CLI and documentation**

```bash
git add scripts/reclassify_output_datasets.py \
  src/output_dataset_reclassification.py \
  tests/test_output_dataset_reclassification.py README.md
git commit -m "feat: add safe output reclassification workflow"
```

### Task 6: Reclassify the Real Outputs Through Explicit Checkpoints

**Files:**
- Modify in place: `outputs/migration_manifest.csv`
- Modify in place: `outputs/migration_report.json`
- Move in place: `outputs/fashioniq/*` to actual dataset directories
- Delete after verification: the five exact audited `.xlsx` paths under `outputs/reports/legacy/`

**Interfaces:**
- Consumes: the committed CLI and all verification interfaces from Tasks 3-5.
- Produces: the approved actual-dataset directory structure with zero pending historical runs.

- [ ] **Step 1: Record the preflight repository and output state**

Run:

```bash
git status --short
find outputs/fashioniq -type f -name 'run_manifest.json' | wc -l
find outputs -type f | wc -l
find outputs -type f -name '*.pt' | wc -l
find outputs -type f -name 'run_manifest.json' | wc -l
find outputs/reports/legacy -type f -name '*.xlsx' | wc -l
```

Expected: 215 runs, 1004 total files, 201 checkpoints, 215 run manifests, and 5 Excel files. Preserve the displayed user-owned worktree changes.

- [ ] **Step 2: Run the real dry-run and inspect exact classification counts**

Run:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py \
  > /tmp/clip4cir-dataset-reclassify-dry-run.json
python -m json.tool /tmp/clip4cir-dataset-reclassify-dry-run.json
```

Expected:

```json
{
  "mode": "dry-run",
  "total_runs": 215,
  "dataset_counts": {
    "idrid": 172,
    "uwf": 41,
    "combined-fundus-cir": 2
  },
  "unresolved": 0,
  "collisions": 0
}
```

Do not continue if any count differs.

- [ ] **Step 3: Check for active training writers immediately before apply**

Run:

```bash
pgrep -af 'clip_fine_tune.py|combiner_train.py' || true
```

Expected: no training writer process. The apply command repeats this check internally and remains the authority.

- [ ] **Step 4: Apply the transaction without deleting reports**

Run:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py --apply \
  > /tmp/clip4cir-dataset-reclassify-apply.json
python -m json.tool /tmp/clip4cir-dataset-reclassify-apply.json
```

Expected: applied state, 215 moved runs, the exact 172/41/2 distribution, zero errors, and 201 checkpoints verified. If apply fails with incomplete rollback, stop and preserve `.dataset-reclassify-staging` and its journal.

- [ ] **Step 5: Run the independent applied-state verification checkpoint**

Run:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py --verify \
  > /tmp/clip4cir-dataset-reclassify-verify.json
python -m json.tool /tmp/clip4cir-dataset-reclassify-verify.json
```

Expected: `ok: true`, 215 run manifests consistent with paths, 201 checkpoints, all 787 pre-finalize successful audit destinations valid, and 5 legacy reports still present.

- [ ] **Step 6: Finalize the five approved report deletions and empty-directory cleanup**

Run:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py --finalize \
  > /tmp/clip4cir-dataset-reclassify-finalize.json
python -m json.tool /tmp/clip4cir-dataset-reclassify-finalize.json
```

Expected: `ok: true`; exactly five report rows have status `deleted-approved-report` and reason `user-approved-obsolete-legacy-summary`; `outputs/reports`, `outputs/fashioniq`, and staging are absent.

- [ ] **Step 7: Perform final filesystem and audit acceptance checks**

Run:

```bash
test ! -e outputs/fashioniq
test ! -e outputs/reports
test ! -e outputs/.dataset-reclassify-staging
find outputs/idrid -type f -name 'run_manifest.json' | wc -l
find outputs/uwf -type f -name 'run_manifest.json' | wc -l
find outputs/combined-fundus-cir -type f -name 'run_manifest.json' | wc -l
find outputs -type f -name '*.pt' | wc -l
find outputs -type f -name 'run_manifest.json' | wc -l
PYTHONPATH=src python scripts/reclassify_output_datasets.py
```

Expected: 172 IDRiD runs, 41 UWF runs, 2 Combined runs, 201 checkpoints, 215 run manifests, and a final dry-run with zero pending, unresolved, collision, or error items.

- [ ] **Step 8: Verify audit accounting and the full test suite**

Run:

```bash
PYTHONPATH=src python scripts/reclassify_output_datasets.py --verify
PYTHONPATH=src pytest -q
git status --short
```

Expected:

- 782 retained original migration files still match the original size and SHA-256;
- 5 approved Excel rows require absent destinations;
- 2 historical failed-run rows remain preserved;
- 215 generated run manifests match their actual dataset paths;
- the output verifier reports no errors;
- tests introduce no regression beyond the already-known assertion caused only by the user's modified `命令.sh`;
- Git status still preserves the user's unrelated changes and does not stage `outputs/`.
