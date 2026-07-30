# Med-REDUCE — Experiment Runbook

Full command sequence for the three frozen teachers (**DINOv3, ViT, BiomedCLIP**)
across all experiments: LP baseline, embedding distillation, and distilled-student
LP eval. Naming is uniform `med-reduce` (paths, W&B) and `med_reduce_distillation`
(Python package).

## Golden rules (baked into every command below)
- Every path in `EXTRAS` starts with `/scratch/users/$USER/…` (writable); **never a bare `/name`** (that resolves to the read-only container root).
- Always `++datamodule.force_recompute_embeddings=false` — `++` adds-or-overrides; `false` creates splits if missing, reuses them if present.
- **Never override `split_dir`** — it stays the shared `med-reduce-<domain>-results/splits`, which is what makes the teacher comparison fair.
- **Run DINOv3 first** — it creates the shared splits; ViT/BiomedCLIP then reuse them. Don't launch all three on a fresh split dir simultaneously (create race).

```bash
# Common handles (paste once per shell session)
DINOV3=facebook/dinov3-vits16-pretrain-lvd1689m
VIT=google/vit-base-patch16-224
BMC=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

---

## Phase 0 — one-time setup (on Sherlock)

```bash
# 0a. Sync BOTH checkouts to scratch (push from Mac + pull here, or rsync)
cd /scratch/users/$USER/med-reduce               && git pull
cd /scratch/users/$USER/med-reduce-distillation  && git pull

# 0b. Stage data OAK → scratch (adjust source subpaths to your OAK layout)
OAK=${OAK:-/oak/stanford/groups/roxanad/$USER}; DST=${MR_DATA_ROOT:-/scratch/groups/roxanad/datasets}
rsync -av $OAK/processed_chexpert/explore_chexpert/  $DST/chexpert/explore_chexpert/   # label CSV
rsync -av $OAK/processed_chexpert/combined_train_valid_chexpert_v1.0/ $DST/chexpert/combined_train_valid_chexpert_v1.0/
rsync -av $OAK/processed_isic/  $DST/isic/
rsync -av $OAK/processed_tcga/  $DST/tcga/
# verify the radiology blocker resolves:
ls $DST/chexpert/explore_chexpert/train_valid_combined.csv

# 0c. Main-repo venv + model prefetch (dinov3/vit/biomedclip + students)
cd /scratch/users/$USER/med-reduce
sbatch jobs/setup_container.sh

# 0d. Rebuild distillation container (package was renamed → old sif is stale)
cd /scratch/users/$USER/med-reduce-distillation
apptainer build /scratch/users/$USER/simg/pipeline.sif container/pipeline.def

# 0e. Authoritative tests on the node
cd /scratch/users/$USER/med-reduce               && python -m pytest -q
cd /scratch/users/$USER/med-reduce-distillation  && python -m pytest -q
```

---

## Phase A — LP baseline (per teacher)

`train_container.sh` loops seeds `42 123 456`, resolutions `512 256 128 64`, and
tunes HPs once at the top resolution.

```bash
cd /scratch/users/$USER/med-reduce

# 1) DINOv3 (native 512) — generates the shared splits
MODEL=dinov3 DOMAIN=dermatology \
  EXTRAS="runtime.run_dir=/scratch/users/$USER/med-reduce-dinov3/dermatology/runs/probe embedding_cache_dir=/scratch/users/$USER/med-reduce-dinov3/dermatology/cache/embeddings ++datamodule.force_recompute_embeddings=false" \
  sbatch jobs/train_container.sh

# …wait until splits exist, THEN launch the other two (avoid a create race):
ls /scratch/users/$USER/med-reduce-derm-results/splits/images/seed_42/   # train_indices.npy, test_indices.npy, metadata.json

# 2) ViT (native 224) — reuses the splits
MODEL=vit DOMAIN=dermatology \
  EXTRAS="runtime.run_dir=/scratch/users/$USER/med-reduce-vit/dermatology/runs/probe embedding_cache_dir=/scratch/users/$USER/med-reduce-vit/dermatology/cache/embeddings ++datamodule.force_recompute_embeddings=false" \
  sbatch jobs/train_container.sh

# 3) BiomedCLIP (native 224) — reuses the splits
MODEL=biomedclip DOMAIN=dermatology \
  EXTRAS="runtime.run_dir=/scratch/users/$USER/med-reduce-biomedclip/dermatology/runs/probe embedding_cache_dir=/scratch/users/$USER/med-reduce-biomedclip/dermatology/cache/embeddings ++datamodule.force_recompute_embeddings=false" \
  sbatch jobs/train_container.sh
```

**Repeat for `DOMAIN=radiology` and `DOMAIN=pathology`** — same three MODEL
commands, only swap `DOMAIN` and the `dermatology` path segment. Pathology
auto-uses the shared case-level TCGA split across all 5 tasks (override with
`TASKS="kras tp53"` to run a subset).

---

## Phase B — Distillation (container, per teacher × student × seed)

**Reuses the Phase-A teacher embeddings** (matched by image ID) via
`HOST_TEACHER_EMB_DIR` — no re-extraction, no teacher weights loaded. Also reuses
the Phase-A splits via `HOST_SPLITS_DIR`; outputs are teacher-tagged so nothing
overwrites. Students: `resnet` = `resnet50.a1_in1k`, `tinyvit` =
`tiny_vit_21m_224.dist_in22k`.

> **Run Phase A first.** The LP baseline writes the id-tagged teacher cache that
> Phase B consumes. The clean target is the LP top resolution (512px).

```bash
cd /scratch/users/$USER/med-reduce-distillation
# LP teacher cache layout: <results>/<domain>/cache/embeddings/<dataset>/<model>/seed_<S>
# For all three teachers model.name == TEACHER_TAG (dinov3 | vit | biomedclip).
for STU in resnet tinyvit; do for S in 42 123 456; do for TAG in dinov3 vit biomedclip; do
  case $TAG in
    dinov3)     TYPE=hf;         MODEL=$DINOV3 ;;
    vit)        TYPE=hf;         MODEL=$VIT ;;
    biomedclip) TYPE=biomedclip; MODEL=$BMC ;;
  esac
  EMB=/scratch/users/$USER/med-reduce-${TAG}/dermatology/cache/embeddings/images/${TAG}/seed_$S
  env STUDENT=$STU SEED=$S DOMAIN=dermatology DATASET_NAME=ISIC2017 \
    IMAGES_ROOT=/data/isic/challenges/2017/merged_isic_2017_data/images \
    HOST_SPLITS_DIR=/scratch/users/$USER/med-reduce-derm-results/splits SPLITS_DIR=/splits/images/seed_$S \
    TEACHER_TYPE=$TYPE TEACHER_MODEL=$MODEL TEACHER_TAG=$TAG \
    HOST_TEACHER_EMB_DIR=$EMB TEACHER_CLEAN_RES=512 \
    sbatch scripts/run_pipeline_container.sh
done; done; done
# → /scratch/users/$USER/pipeline_output/ISIC2017_${STU}_${TAG}/seed_$S/trained_model_${STU}_${TAG}.ckpt
```

Per-domain `HOST_TEACHER_EMB_DIR` differs only in the `<domain>` and `<dataset>`
segments (e.g. `.../med-reduce-dinov3/radiology/cache/embeddings/combined_train_valid_chexpert_v1.0/dinov3/seed_$S`).

To **extract fresh instead of reusing** (old behaviour), just omit
`HOST_TEACHER_EMB_DIR`. Pathology also needs thumbnail-order splits per seed first:
```bash
python scripts/build_tcga_distill_split.py   # writes /splits/tcga/seed_X (see script header for args)
```

---

## Phase C — LP eval of distilled students (no HP tuning)

`TUNE_HP=0` reuses teacher HPs (per project requirement — no tuning in Phase C).
Point `CHECKPOINT` at each Phase-B `.ckpt`.

```bash
cd /scratch/users/$USER/med-reduce
for S in 42 123 456; do for TAG in dinov3 vit biomedclip; do
  # resnet50 student
  CHECKPOINT=/scratch/users/$USER/pipeline_output/ISIC2017_resnet_${TAG}/seed_$S/trained_model_resnet_${TAG}.ckpt \
  STUDENT=resnet50 STUDENT_NAME=resnet50_${TAG}_distilled DOMAIN=dermatology SEEDS="$S" TUNE_HP=0 \
    EXTRAS="runtime.run_dir=/scratch/users/$USER/med-reduce-${TAG}/dermatology/runs/eval_resnet50 embedding_cache_dir=/scratch/users/$USER/med-reduce-${TAG}/dermatology/cache/eval_resnet50 ++datamodule.force_recompute_embeddings=false" \
    sbatch jobs/eval_distilled_container.sh

  # tiny_vit student
  CHECKPOINT=/scratch/users/$USER/pipeline_output/ISIC2017_tinyvit_${TAG}/seed_$S/trained_model_tinyvit_${TAG}.ckpt \
  STUDENT=tiny_vit_21m_224 STUDENT_NAME=tinyvit_${TAG}_distilled DOMAIN=dermatology SEEDS="$S" TUNE_HP=0 \
    EXTRAS="runtime.run_dir=/scratch/users/$USER/med-reduce-${TAG}/dermatology/runs/eval_tinyvit embedding_cache_dir=/scratch/users/$USER/med-reduce-${TAG}/dermatology/cache/eval_tinyvit ++datamodule.force_recompute_embeddings=false" \
    sbatch jobs/eval_distilled_container.sh
done; done
```

---

## Per-domain substitutions

| Domain | `DOMAIN` | `DATASET_NAME` | `IMAGES_ROOT` (in-container `/data/…`) | split subfolder |
|---|---|---|---|---|
| Dermatology | `dermatology` | `ISIC2017` | `/data/isic/challenges/2017/merged_isic_2017_data/images` | `images` |
| Radiology | `radiology` | `CheXpert` | `/data/chexpert/combined_train_valid_chexpert_v1.0` | `combined_train_valid_chexpert_v1.0` |
| Pathology | `pathology` | `TCGA` | `/data/tcga/thumbnails` | `tcga_<task>` |

## Native resolutions
- DINOv3 = 512 · ViT = 224 · BiomedCLIP = 224 (auto-pinned via `MODEL_CONFIGS`; the degradation ladder `512 256 128 64` is downsampled→upsampled to native).
