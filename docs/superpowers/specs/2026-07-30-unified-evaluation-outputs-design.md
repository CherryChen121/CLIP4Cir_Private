# Unified Evaluation Outputs Design

## Goal

Route every formal validation result into the same actual-dataset-first
`outputs/` hierarchy used by training. Validation must no longer leave CSV or
log files in the project root, and a FashionIQ-compatible data format must not
cause IDRiD, UWF, or Combined Fundus evaluations to be filed under
`outputs/fashioniq`.

This change covers:

- `src/validate.py`
- `src/validate_retizero_lora.py`
- the three Combined Fundus validation command templates in `命令.sh`

Submission generation such as `src/cirr_test_submission.py`, training output
layout, and migration of historical files are outside this change.

## Chosen Approach

Use an actual-dataset-first evaluation tree and a shared evaluation-output
component:

```text
outputs/
└── <actual-dataset>/
    └── evaluation/
        └── <model-slug>/
            └── <timestamp-and-pid-run-id>/
                ├── evaluation_manifest.json
                ├── evaluation_metrics.json
                ├── evaluation_metrics.csv
                └── evaluation.log
```

This keeps each evaluation self-contained, prevents overwrite between repeated
runs, and makes training and evaluation discoverable beneath the same real
dataset directory. An alternative central `outputs/evaluations/` tree was
rejected because it would separate results from their data. Attaching an
evaluation directly to a training run was rejected because validation may
compare external checkpoints or multiple checkpoints that do not belong to one
training run.

## Components

### Shared evaluation-output component

Add a focused module, `src/evaluation_outputs.py`. It will reuse path primitives
from `src/output_paths.py` where appropriate, while keeping evaluation
lifecycle and metric serialization separate from training checkpoint layout.

The module will:

- resolve the default or explicit output root;
- slugify dataset and model path components;
- allocate an exclusive run directory with the existing timestamp-plus-PID run
  ID convention;
- create and atomically update `evaluation_manifest.json`;
- tee Python stdout and stderr to `evaluation.log` while preserving terminal
  output;
- atomically write JSON and CSV metrics;
- mark a run `succeeded` or `failed`.

The run directory is created with exclusive semantics. A collision is an error
rather than permission to reuse or overwrite another run.

### Dataset identity

Both validation scripts will use `src/dataset_identity.py`, the same identity
rules used by training.

For a FashionIQ-compatible evaluation:

1. Resolve the selected category roots for the selected split.
2. Require every category in one invocation to resolve to the same physical
   dataset root.
3. Classify the actual dataset from the resolved root and category evidence.
4. Let an explicit `--output-dataset` override classification when supplied.

This means a root named `IDRiD_CIR_Dataset_cold`,
`UWF_CIR_Dataset_cold`, or `Combined_Fundus_CIR_Dataset` maps to `idrid`,
`uwf`, or `combined-fundus-cir`, even though the loader and CLI continue to use
the FashionIQ-compatible format. CIRR evaluations map to `cirr`.

Conflicting root and category evidence, multiple resolved roots, or an
unidentifiable dataset fail before expensive model evaluation starts.

## Command-Line Interface

Both validation scripts will support:

- `--output-root`: output tree root, defaulting to the project's `outputs/`;
- `--output-dataset`: explicit actual-dataset slug override;
- `--evaluation-name`: a descriptive label stored in the manifest, such as
  `internal-test` or `odir5k-test`.

`src/validate_retizero_lora.py` will additionally expose the dataset selection
that is currently implicit:

- `--fashioniq-root`;
- `--dress-types`;
- `--fashioniq-split`.

Its existing `--output-csv` option remains accepted for command compatibility,
but its value becomes a filename inside the allocated run directory. Directory
components and absolute paths are rejected. The default filename is
`evaluation_metrics.csv`. The primary result cannot be redirected outside the
evaluation run.

The Combined Fundus validation templates in `命令.sh` will explicitly use:

```text
--output-dataset combined-fundus-cir
--evaluation-name <category-and-split-label>
```

Their project-root `eval_*.log` redirections will be removed. Because the
scripts create `evaluation.log` internally, background template commands may
redirect launcher stdout/stderr to `/dev/null` to prevent `nohup.out` from
appearing in the project root.

## Evaluation Lifecycle

The scripts perform these steps:

1. Parse and validate CLI arguments.
2. Resolve the physical dataset root and actual dataset identity.
3. Allocate the evaluation run directory.
4. Write a manifest with status `running`.
5. Start teeing stdout and stderr into `evaluation.log`.
6. Load models and execute validation.
7. Convert results into the shared in-memory result structure.
8. Atomically write `evaluation_metrics.json` and
   `evaluation_metrics.csv`.
9. Atomically update the manifest to `succeeded`.

An exception after run allocation updates the manifest to `failed`, records the
exception type and message, preserves the log, and is re-raised so the command
has a non-zero exit status. A failed run must not publish final metric files.

Failures before argument parsing or run allocation cannot be associated with a
run directory; this is limited to interpreter/import failures and does not
justify retaining project-root launcher logs.

## Manifest Schema

`evaluation_manifest.json` contains at least:

- schema version;
- status: `running`, `succeeded`, or `failed`;
- actual dataset slug and data format;
- requested and resolved dataset roots;
- dataset-classification evidence;
- evaluation script and evaluation name;
- model name and model slug;
- split and selected categories;
- combining function and preprocessing settings when applicable;
- input checkpoint and base-weight paths;
- complete parsed CLI arguments;
- run ID, process ID, start time, and completion time;
- relative names of the log and metric files;
- exception type and message only for failed runs.

Paths used as inputs remain paths; model or checkpoint files are not copied into
the evaluation run.

## Metric Schema

`evaluation_metrics.json` is the canonical representation. It contains common
run metadata plus a `results` list. Each result represents one evaluated model
or checkpoint and contains:

- input model/checkpoint path when applicable;
- checkpoint epoch or evaluation sequence number when available;
- per-category metrics;
- aggregate metrics.

`evaluation_metrics.csv` is a wide, spreadsheet-friendly projection with one
row per evaluated model or checkpoint. FashionIQ-compatible columns use the
existing names such as `<category>_recall_at1` and
`average_recall_at1`. CIRR columns use explicit global and group recall names.
The single-model `validate.py` path therefore writes one row, while a
multi-checkpoint RetiZero LoRA invocation writes one row per checkpoint.

Both scripts always publish JSON and CSV on success. They continue printing a
human-readable summary, which is captured in `evaluation.log`.

## Atomicity and Safety

- JSON and CSV files are written to temporary siblings and renamed only after
  a complete write.
- A final metric filename must be a basename, preventing `--output-csv` path
  traversal or writes outside the run.
- Existing run directories are never reused.
- Metric files are published only after all requested models and categories
  succeed, so a partial multi-checkpoint run is visibly failed rather than
  presented as complete.
- Existing training runs and validation outputs are not moved or deleted by
  this feature.

## Testing

Tests will cover:

- default and explicit evaluation output roots;
- exclusive run allocation and deterministic injected timestamp/PID values;
- actual-dataset classification for IDRiD, UWF, Combined Fundus, FashionIQ, and
  CIRR;
- rejection of conflicting or unresolved dataset identity;
- manifest transitions for success and failure;
- atomic JSON and CSV serialization;
- log tee behavior;
- rejection of absolute or directory-bearing `--output-csv` values;
- both validation CLIs exposing the new options;
- RetiZero LoRA passing explicit root, categories, and split into dataset
  construction;
- metric schema for single-model, multi-checkpoint, and CIRR results;
- all three Combined validation templates in `命令.sh` selecting
  `combined-fundus-cir`, naming their evaluations, and not creating root-level
  evaluation logs.

GPU-heavy evaluation will be mocked in focused tests. The complete existing
test suite and `bash -n 命令.sh` will run before completion.

## Success Criteria

The work is complete when:

1. Both validation entry points always create structured successful or failed
   runs beneath `outputs/<actual-dataset>/evaluation/`.
2. A FashionIQ-compatible loader no longer determines the output dataset name.
3. Successful runs contain a manifest, JSON metrics, CSV metrics, and a log.
4. Failed allocated runs contain a failed manifest and log but no final metric
   files.
5. The Combined validation command templates no longer generate root-level
   evaluation logs.
6. Focused and full project tests pass.
