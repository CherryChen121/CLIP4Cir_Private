# UWF Fundus CIR — CLIP4Cir & RetiZero on Ophthalmic Datasets

### Adapting Composed Image Retrieval for Ultra-Wide Field Fundus Disease Images

---

## Overview

This project adapts the **Composed Image Retrieval (CIR)** framework to the medical imaging domain. Specifically, we restructure an Ultra-Wide Field (UWF) fundus disease image dataset into the **FashionIQ format** and apply CLIP-family backbones (**RN50x4**, **ViT-B/32**, **ViT-L/14**) plus **RetiZero** within the CLIP4Cir training pipeline (CLIP fine-tuning + Combiner network training). The primary evaluation metric is **average recall rate (Recall@1 and Recall@10)** across 6 fundus disease categories.

### Source Projects

| Project | Description | GitHub |
|---------|-------------|--------|
| **CLIP4Cir** | Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features (ACM TOMM 2023) | [ABaldrati/CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir) |
| **RetiZero** | Vision-language foundation model for fundus images, pre-trained on 400+ diseases | [LooKing9218/RetiZero](https://github.com/LooKing9218/RetiZero) |

---

## Project Structure

```
CLIP4Cir/
├── src/
│   ├── clip_fine_tune.py          # Stage 1: CLIP task-oriented fine-tuning
│   ├── combiner_train.py          # Stage 2: Combiner network training
│   ├── validate.py                # Validation metrics computation
│   ├── data_utils.py              # Dataset loading (with UWF compatibility patches)
│   ├── combiner.py                # Combiner network definition
│   ├── retizero_adapter.py        # RetiZero → CLIP4Cir interface adapter
│   ├── validate_retizero_lora.py  # Validation script for RetiZero+LoRA variant
│   ├── utils.py                   # Utility functions
│   └── cirr_test_submission.py    # (Original) CIRR test submission
├── fashionIQ_dataset/             # UWF dataset restructured in FashionIQ format
│   ├── captions/                  # Text descriptions per disease category
│   ├── images/                    # UWF fundus images (.png / .jpg)
│   └── image_splits/              # Train / val / test splits
├── pretrained_models/
│   └── fashionIQ/
│       ├── tuned_clip_best.pt     # Fine-tuned CLIP checkpoint (RN50x4)
│       └── RetiZero.pth           # Pre-trained RetiZero weights
└── outputs/                       # Generated training runs (Git-ignored)
```

---

## Dataset: UWF Fundus CIR

The original UWF fundus disease images are restructured into the **FashionIQ format**, treating disease categories as analogues of fashion item types. The 6 categories used are:

| Abbreviation | Disease Category |
|---|---|
| **CH** | Choroidal Hemangioma |
| **CO** | Choroidal Others |
| **NM** | Normal (Myopia) |
| **RB** | Retinal Breaks |
| **RCH** | Retinal Choroidal |
| **UM** | Uveal Melanoma |

Each category has corresponding `captions/*.json`, `image_splits/*.json`, and images under `images/`, following the same directory structure expected by the original FashionIQ data loaders.

### Data Loader Compatibility Patches (`data_utils.py`)

Several modifications were made to handle domain-specific issues:

- **Mixed image format support**: Auto-detection of `.png` / `.jpg` suffixes to handle heterogeneous image extensions.
- **Missing image filtering**: Physical disk validation at dataset initialization to remove broken or absent image references from triplets and index sets.
- **Extended category list**: Added UWF disease abbreviations (CH, CO, NM, RB, RCH, UM) alongside the original FashionIQ categories (dress, shirt, toptee).
- **Text truncation**: Medical descriptions often exceed 77 tokens; enforced `truncate=True` in CLIP tokenizer to avoid runtime crashes.
- **Dynamic batch alignment**: Inserted dynamic slicing logic in the Combiner's `combine_features` to resolve tensor shape mismatches across heterogeneous batches.

---

## Two-Stage Training Pipeline

### Stage 1: CLIP Task-Oriented Fine-Tuning

Fine-tune both CLIP image and text encoders on the UWF CIR triplets using contrastive loss (element-wise sum of visual + textual features).

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/clip_fine_tune.py \
   --dataset FashionIQ \
   --dress-types CH CO NM RB RCH UM \
   --num-epochs 100 \
   --clip-model-name RN50x4 \
   --encoder both \
   --learning-rate 2e-6 \
   --batch-size 128 \
   --transform targetpad \
   --target-ratio 1.25 \
   --save-training \
   --save-best \
   --validation-frequency 1 > clip_finetune.log 2>&1 &
```

**OpenAI CLIP ViT variants (same entrypoint, different model/checkpoint):**

```bash
# ViT-B/32 fine-tuning
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/clip_fine_tune.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --num-epochs 100 \
    --clip-model-name ViT-B/32 \
    --clip-model-path pretrained_models/fashionIQ/ViT-B-32.pt \
    --encoder both \
    --learning-rate 2e-6 \
    --batch-size 128 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 1 > clip_finetune_vitb32.log 2>&1 &

# ViT-L/14 fine-tuning (typically lower batch size)
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/clip_fine_tune.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --num-epochs 100 \
    --clip-model-name ViT-L/14 \
    --clip-model-path pretrained_models/fashionIQ/ViT-L-14.pt \
    --encoder both \
    --learning-rate 2e-6 \
    --batch-size 64 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 1 > clip_finetune_vitl14.log 2>&1 &
```

### Stage 2: Combiner Network Training

Train the Combiner network on top of fixed (frozen) encoder features. Supports both CLIP (RN50x4) and RetiZero as backbone.

**CLIP backbone:**

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/combiner_train.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --projection-dim 2560 \
    --hidden-dim 5120 \
    --num-epochs 150 \
    --clip-model-name RN50x4 \
    --clip-model-path pretrained_models/fashionIQ/tuned_clip_best.pt \
    --combiner-lr 2e-5 \
    --batch-size 128 \
    --clip-bs 16 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 5 > combiner_clip.log 2>&1 &
```

**ViT-B/32 and ViT-L/14 Combiner training examples:**

```bash
# ViT-B/32 (feature dim = 512 for OpenAI ViT-B/32)
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/combiner_train.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --projection-dim 512 \
    --hidden-dim 1024 \
    --num-epochs 150 \
    --clip-model-name ViT-B/32 \
    --clip-model-path pretrained_models/fashionIQ/ViT-B-32.pt \
    --combiner-lr 2e-5 \
    --batch-size 128 \
    --clip-bs 16 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 5 > combiner_vitb32.log 2>&1 &

# ViT-L/14 (feature dim = 768)
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/combiner_train.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --projection-dim 768 \
    --hidden-dim 1536 \
    --num-epochs 150 \
    --clip-model-name ViT-L/14 \
    --clip-model-path pretrained_models/fashionIQ/ViT-L-14.pt \
    --combiner-lr 2e-5 \
    --batch-size 128 \
    --clip-bs 16 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 5 > combiner_vitl14.log 2>&1 &
```

Training outputs use a shared, slash-safe layout:

```text
outputs/<dataset>/<clip-finetune|combiner>/<model>/<run-id>/
├── checkpoints/
├── run_manifest.json
├── training_hyperparameters.json
├── train_metrics.csv
└── validation_metrics.csv
```

Model components are normalized (`ViT-B/32` becomes `vit-b-32`) while
`--clip-model-name` keeps the original model string. Both training entrypoints
accept `--output-root`; relative overrides are resolved from the project root.
Without an override, all new runs are written below `outputs/`.

**RetiZero backbone:**

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 nohup python src/combiner_train.py \
    --dataset FashionIQ \
    --dress-types CH CO NM RB RCH UM \
    --projection-dim 2560 \
    --hidden-dim 5120 \
    --num-epochs 150 \
    --clip-model-name RetiZero \
    --clip-model-path pretrained_models/fashionIQ/RetiZero.pth \
    --combiner-lr 2e-5 \
    --batch-size 128 \
    --clip-bs 16 \
    --transform targetpad \
    --target-ratio 1.25 \
    --save-training \
    --save-best \
    --validation-frequency 5 > combiner_retizero.log 2>&1 &
```

---

## RetiZero Integration

RetiZero is a fundus-specific vision-language model pre-trained on 341,896 fundus images covering 400+ diseases. To plug it into the CLIP4Cir Combiner training pipeline, a wrapper adapter (`src/retizero_adapter.py`) was implemented that:

- Exposes `encode_image()` and `encode_text()` interfaces compatible with CLIP4Cir's feature extraction calls.
- Mounts a `.visual` attribute with `input_resolution=224` and `output_dim=512` to satisfy internal interface checks.
- Loads a local Bio_ClinicalBERT tokenizer for text encoding (offline mode to avoid network dependency).
- Applies token truncation at `max_length=77` to align with CLIP's token limit.
- Supports LoRA checkpoint loading for lightweight fine-tuning evaluation via `validate_retizero_lora.py`.

---

## Validation

Compute Recall@1 and Recall@10 per category and overall average recall:

```bash
# CLIP backbone
python src/validate.py \
   --dataset FashionIQ \
   --dress-types CH CO NM RB RCH UM \
   --combining-function combiner \
   --combiner-path outputs/fashioniq/combiner/<model>/<run-id>/checkpoints/combiner.pt \
   --projection-dim 2560 \
   --hidden-dim 5120 \
   --clip-model-name RN50x4 \
   --clip-model-path pretrained_models/fashionIQ/tuned_clip_best.pt \
   --target-ratio 1.25 \
   --transform targetpad

# RetiZero backbone
python src/validate_retizero_lora.py \
    --model-paths outputs/fashioniq/clip-finetune/retizero/<run-id>/checkpoints/*.pt \
    --base-weight-path pretrained_models/fashionIQ/RetiZero.pth \
    --output-csv results_retizero.csv
```

---

## Legacy Output Migration

The migration tool audits the legacy `models/` tree, validates checkpoint
containers, computes SHA-256 hashes, and reports strict failures and exact
duplicates. Dry run is the default and does not create or modify files:

```bash
PYTHONPATH=src python scripts/organize_model_outputs.py
PYTHONPATH=src python scripts/organize_model_outputs.py --apply
PYTHONPATH=src python scripts/organize_model_outputs.py --verify
PYTHONPATH=src python scripts/organize_model_outputs.py --finalize
```

`--apply` is blocked while an old training process may still write to
`models/`. It migrates through `outputs/.staging`, preserves per-run metadata,
and replaces byte-identical checkpoints with hard links. `--finalize` first
verifies every migrated hash and duplicate inode, then removes only an empty
legacy source tree using guarded directory removal.

---

## Key Code Modifications Summary

| # | File | Modification |
|---|------|-------------|
| 001 | `data_utils.py` | Dual-extension auto-detection (.png/.jpg), missing image filtering, UWF category support |
| 027 | `validate.py` | Added `truncate=True` to CLIP tokenizer for long medical text |
| 030 | `combiner_train.py` | Dynamic tensor slicing to fix batch size mismatches |
| 037 | `combiner_train.py` | `nn.DataParallel` wrapping for multi-GPU training |
| — | `retizero_adapter.py` | Full CLIP-interface adapter for RetiZero model |
| — | `validate_retizero_lora.py` | Standalone validation pipeline for RetiZero+LoRA |

---

## Environment Setup

```bash
conda create -n clip4cir -y python=3.8
conda activate clip4cir
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
conda install -y -c anaconda pandas=1.4.2
pip install comet-ml==3.21.0
pip install git+https://github.com/openai/CLIP.git
pip install transformers
```

---

## Acknowledgements

- **CLIP4Cir** (ACM TOMM 2023): Alberto Baldrati et al. — [https://github.com/ABaldrati/CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir)
- **RetiZero**: LooKing9218 et al. — [https://github.com/LooKing9218/RetiZero](https://github.com/LooKing9218/RetiZero)
