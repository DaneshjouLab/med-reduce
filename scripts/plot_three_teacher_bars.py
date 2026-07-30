#!/usr/bin/env python3
"""Two figures from results-med-reduce-clean/ for the three-teacher comparison (512 px, mean+-SD/3 seeds):

  three_teacher_accuracy.png  -- 7 task-rows x 3 metric-cols (AUROC, Top-1, Macro-F1),
                                 each panel = 3 teachers x {Teacher(LP), ResNet-50, TinyViT}.
  three_teacher_memory.png    -- 1x3 domains, peak GPU memory (MB), same grouping.

Pathology tasks are shown separately (not macro-averaged). Efficiency = peak GPU memory only.
"""
import os, re, json, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN = os.environ.get('MR_RESULTS_CLEAN', 'results-med-reduce-clean')
PAPER = os.environ.get('MR_PAPER_DIR', '.')
RES = 512

# (dataset, teacher, role) -> seed -> {auroc, top1, f1}
acc = defaultdict(lambda: defaultdict(dict))
# (domain, teacher, role) -> [mem, ...]
mem = defaultdict(list)
for root, _, fs in os.walk(CLEAN):
    for f in fs:
        if not re.match(r'results_.*px\.json$', f):
            continue
        parts = root.split('/')
        try:
            i = parts.index('results-med-reduce-clean')
        except ValueError:
            continue
        if len(parts) < i + 4:
            continue
        domain, teacher, cat = parts[i + 1], parts[i + 2], parts[i + 3]
        if cat not in ('baseline', 'resnet', 'tinyvit'):
            continue
        d = json.load(open(os.path.join(root, f)))
        ei = d['experiment_info']
        if ei['resolution'] != RES:
            continue
        a = d['accuracy_metrics']
        acc[(ei['dataset'], teacher, cat)][ei['seed']] = {
            'auroc': a['best_metric'], 'top1': a.get('final_val_acc'), 'f1': a.get('final_val_f1')}
        m = d.get('efficiency_metrics', {}).get('peak_gpu_memory_mb')
        if m is not None:
            mem[(domain, teacher, cat)].append(m)

TEACHERS = [('dinov3', 'DINOv3'), ('vit', 'ViT'), ('biomedclip', 'BiomedCLIP')]
ROLES = [('baseline', 'Teacher (LP)'), ('resnet', 'ResNet-50 (distilled)'), ('tinyvit', 'TinyViT (distilled)')]
RCOL = {'baseline': '#2176AE', 'resnet': '#E8572A', 'tinyvit': '#57A773'}
TASKS = [
    ('images', 'Dermatology (ISIC)'),
    ('tcga_luad_vs_lusc', 'Pathology: LUAD vs LUSC'),
    ('tcga_lgg_vs_gbm', 'Pathology: LGG vs GBM'),
    ('tcga_kras', 'Pathology: KRAS'),
    ('tcga_tp53', 'Pathology: TP53'),
    ('tcga_egfr', 'Pathology: EGFR'),
    ('combined_train_valid_chexpert_v1.0', 'Radiology (CheXpert)'),
]
METRICS = [('auroc', 'AUROC'), ('top1', 'Top-1 accuracy'), ('f1', 'Macro F1')]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})


def ms_acc(ds, teacher, role, metric):
    perseed = acc.get((ds, teacher, role), {})
    vals = [v[metric] for v in perseed.values() if v.get(metric) is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


# ---------- Figure A: accuracy 7 x 3 ----------
nT, nR = len(TEACHERS), len(ROLES)
bw = 0.8 / nR
x = np.arange(nT)
figA, axesA = plt.subplots(len(TASKS), len(METRICS), figsize=(12, 20))
# per-column shared y-limits
for ci, (mkey, mlab) in enumerate(METRICS):
    allv = []
    for ds, _ in TASKS:
        for tk, _ in TEACHERS:
            for role, _ in ROLES:
                m, _ = ms_acc(ds, tk, role, mkey)
                if m is not None:
                    allv.append(m)
    lo, hi = (min(allv), max(allv)) if allv else (0, 1)
    ylo, yhi = max(0.0, lo - 0.06), min(1.0, hi + 0.06)
    for ri, (ds, tlab) in enumerate(TASKS):
        ax = axesA[ri][ci]
        for j, (role, rlab) in enumerate(ROLES):
            ms = [ms_acc(ds, tk, role, mkey) for tk, _ in TEACHERS]
            vals = [v[0] if v[0] is not None else 0 for v in ms]
            errs = [v[1] if v[1] is not None else 0 for v in ms]
            off = (j - (nR - 1) / 2) * bw
            ax.bar(x + off, vals, bw, yerr=errs, capsize=2.5, color=RCOL[role], alpha=0.9,
                   edgecolor='white', linewidth=0.5,
                   label=(rlab if (ri == 0 and ci == 0) else None))
        ax.set_xticks(x); ax.set_xticklabels([t[1] for t in TEACHERS], fontsize=9)
        ax.set_ylim(ylo, yhi)
        if ci == 0:
            ax.set_ylabel(tlab, fontsize=10)
        if ri == 0:
            ax.set_title(mlab, fontsize=13)
handles, labels = axesA[0][0].get_legend_handles_labels()
figA.legend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.005))
figA.suptitle("Accuracy across three teachers and their distilled students "
              "(mean $\\pm$ SD, 3 seeds, 512 px)", y=1.02, fontsize=15)
figA.tight_layout(rect=[0, 0, 1, 0.995])
figA.savefig(os.path.join(PAPER, 'three_teacher_accuracy.png'))
print("saved three_teacher_accuracy.png")

# ---------- Figure B: memory 1 x 3 ----------
DOMAINS = [('dermatology', 'Dermatology (ISIC)'),
           ('pathology', 'Pathology (TCGA)'),
           ('radiology', 'Radiology (CheXpert)')]
figB, axesB = plt.subplots(1, 3, figsize=(13, 4.5))
for ci, (dom, dlab) in enumerate(DOMAINS):
    ax = axesB[ci]
    for j, (role, rlab) in enumerate(ROLES):
        vals, errs = [], []
        for tk, _ in TEACHERS:
            v = mem.get((dom, tk, role), [])
            vals.append(st.mean(v) if v else 0)
            errs.append(st.pstdev(v) if len(v) > 1 else 0)
        off = (j - (nR - 1) / 2) * bw
        ax.bar(x + off, vals, bw, yerr=errs, capsize=3, color=RCOL[role], alpha=0.9,
               edgecolor='white', linewidth=0.5, label=(rlab if ci == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels([t[1] for t in TEACHERS])
    ax.set_ylim(0, 2000)
    ax.set_title(dlab)
    if ci == 0:
        ax.set_ylabel("Peak GPU memory (MB)")
handles, labels = axesB[0].get_legend_handles_labels()
figB.legend(handles, labels, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.06))
figB.suptitle("Deployment memory footprint (peak GPU memory, mean $\\pm$ SD, 3 seeds, 512 px)",
              y=1.10, fontsize=14)
figB.tight_layout()
figB.savefig(os.path.join(PAPER, 'three_teacher_memory.png'))
print("saved three_teacher_memory.png")

# cleanup old combined figure
old = os.path.join(PAPER, 'three_teacher_bars.png')
if os.path.exists(old):
    os.remove(old); print("removed old three_teacher_bars.png")
