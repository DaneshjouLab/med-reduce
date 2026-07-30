#!/bin/bash
#SBATCH --job-name=diagnose_ckpt
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Checkpoint Diagnostics
#
# Inspects distilled student checkpoints to verify they load correctly
# into the timm model used by the linear probe pipeline.
#
# USAGE:
#   sbatch jobs/diagnose_checkpoint.sh
#
# Optional overrides:
#   CKPT_DIR=/path/to/checkpoints sbatch jobs/diagnose_checkpoint.sh
# =============================================================================

CKPT_DIR="${CKPT_DIR:-/scratch/users/$USER/pipeline_output}"

# Container-based execution
TOOL=$(command -v apptainer || command -v singularity)
SIF_STORE="/scratch/users/$USER/simg"
SIF_IMAGE="${SIF_IMAGE:-python_3.10-slim-v2.sif}"

if [ ! -f "$SIF_STORE/$SIF_IMAGE" ]; then
    echo "ERROR: Container not found at $SIF_STORE/$SIF_IMAGE"
    exit 1
fi

"$TOOL" exec --nv \
     -B "/scratch/users/$USER/med-reduce:/workspace" \
     -B "/scratch/users/$USER:/scratch_user" \
     -B "/scratch/users/$USER/pip_cache:/root/.cache/pip" \
     -B "$CKPT_DIR:/checkpoints" \
     -B "/tmp:/tmp" \
     --pwd /workspace \
     "$SIF_STORE/$SIF_IMAGE" \
      bash -c "

    set -e
    cd /workspace
    source .venv/bin/activate

    export TMPDIR=/scratch/users/$USER/tmp
    export TORCH_HOME=/scratch/users/$USER/torch
    mkdir -p \$TMPDIR \$TORCH_HOME

    python3 -u - <<'PYEOF'
import torch
import timm
import os
import glob

CKPT_DIR = '/checkpoints'
SEP = '=' * 80

# ─── Discover all checkpoints ───────────────────────────────────────────────
ckpt_files = sorted(
    f for f in
    glob.glob(os.path.join(CKPT_DIR, '**', '*.ckpt'), recursive=True)
    + glob.glob(os.path.join(CKPT_DIR, '**', '*.pt'), recursive=True)
    if 'embedding' not in os.path.basename(f).lower()
)

if not ckpt_files:
    print(f'No .ckpt or .pt files found under {CKPT_DIR}')
    exit(1)

print(f'Found {len(ckpt_files)} checkpoint(s)\n')

# ─── Reference: timm model keys ─────────────────────────────────────────────
ref = {}
for name in ['resnet18', 'resnet50', 'tiny_vit_21m_224']:
    m = timm.create_model(name, pretrained=False, num_classes=3)
    all_keys = set(m.state_dict().keys())
    backbone = {k for k in all_keys if not k.startswith(('fc.', 'head.'))}
    ref[name] = {'model': m, 'all': all_keys, 'backbone': backbone}

# ─── extract_state_dict (mirrors src/utils/checkpoint_utils.py) ─────────────
def extract_state_dict(ckpt):
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        for prefix in ['model.backbone.model.', 'backbone.model.', 'model.', 'backbone.']:
            if any(k.startswith(prefix) for k in sd):
                sd = {k.removeprefix(prefix): v for k, v in sd.items()}
                break
        return sd
    if isinstance(ckpt, dict) and 'student_state_dict' in ckpt:
        return ckpt['student_state_dict']
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    return ckpt

# ─── Diagnose each checkpoint ───────────────────────────────────────────────
for ckpt_path in ckpt_files:
    rel = os.path.relpath(ckpt_path, CKPT_DIR)
    print(SEP)
    print(f'CHECKPOINT: {rel}')
    print(SEP)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Top-level keys
    if isinstance(ckpt, dict):
        print(f'  Top-level keys: {list(ckpt.keys())}')
        # Show which extraction path will be used
        if 'state_dict' in ckpt:
            print(f'  -> Will extract via "state_dict" key (Lightning format)')
        elif 'student_state_dict' in ckpt:
            print(f'  -> Will extract via "student_state_dict" key (distillation format)')
        elif 'model_state_dict' in ckpt:
            print(f'  -> Will extract via "model_state_dict" key')
        else:
            print(f'  -> Will use raw dict as state_dict')

    # Extract
    sd = extract_state_dict(ckpt)
    # Filter to tensors only
    sd = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
    sd_keys = set(sd.keys())

    print(f'  Extracted: {len(sd)} tensor parameters')
    print(f'  Sample keys: {sorted(sd.keys())[:5]}')

    # ─── Summary table ───────────────────────────────────────────────────
    header = '\n  %-22s %-18s %-18s %s' % ('Model', 'Backbone', 'Load Result', 'Verdict')
    print(header)
    print('  ' + '-'*22 + ' ' + '-'*18 + ' ' + '-'*18 + ' ' + '-'*20)

    for name, r in ref.items():
        backbone_match = len(sd_keys & r['backbone'])
        backbone_total = len(r['backbone'])

        # Actual load test
        try:
            result = r['model'].load_state_dict(sd, strict=False)
            loaded = len(r['all']) - len(result.missing_keys)
            total = len(r['all'])
            pct = 100 * loaded / total if total else 0
        except RuntimeError:
            loaded = 0
            total = len(r['all'])
            pct = 0

        if pct >= 90:
            verdict = 'OK'
        elif pct >= 50:
            verdict = 'PARTIAL -- check shapes'
        else:
            # Distinguish shape mismatch from key mismatch
            if backbone_match > backbone_total * 0.5 and pct == 0:
                verdict = 'SHAPE MISMATCH -- wrong variant!'
            else:
                verdict = 'FAIL -- random weights!'

        bb_str = '%d/%d' % (backbone_match, backbone_total)
        load_str = '%d/%d (%d%%)' % (loaded, total, pct)
        print('  %-22s %-18s %-18s %s' % (name, bb_str, load_str, verdict))

        # If failed, show what prefix stripping would do
        if pct < 50:
            for prefix in ['backbone', 'encoder', 'model', 'student', 'module']:
                stripped = {}
                for k, v in sd.items():
                    if k.startswith(prefix + '.'):
                        stripped[k[len(prefix)+1:]] = v
                if stripped:
                    match = len(set(stripped.keys()) & r['backbone'])
                    if match > backbone_total * 0.5:
                        print('    ^ FIX: strip "%s." -> %d/%d backbone keys match' % (prefix, match, backbone_total))

    print()

print(SEP)
print('DONE')
print(SEP)
PYEOF
  "
