#!/usr/bin/env python3
"""Per-task, per-regime accuracy figures + per-domain, per-regime memory figures.

Layout (per figure): grouped bars vs resolution (512,256,128,64), teachers as the 3 bars.
Accuracy figure = one per (task, regime): 3 metric sub-panels (AUROC, Top-1, Macro-F1).
Memory figure   = one per (domain, regime): single panel (peak GPU memory).
Regimes (roles) are split into separate figures: baseline / ResNet-50-distilled / TinyViT-distilled.
mean +- SD over 3 seeds. Reads only results-med-reduce-clean/.
"""
import os, re, json, statistics as st, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN = os.environ.get('MR_RESULTS_CLEAN', 'results-med-reduce-clean')
ONLY = sys.argv[1] if len(sys.argv) > 1 else None      # e.g. "dermatology" to preview
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('MR_PAPER_DIR', '.')
RESO = [512, 256, 128, 64]

# acc[(dataset, teacher, role, res)] -> seed->dict ; mem[(dataset,teacher,role,res)] -> [mb]
acc = defaultdict(lambda: defaultdict(dict))
mem = defaultdict(list)
for root, _, fs in os.walk(CLEAN):
    for f in fs:
        if not re.match(r'results_.*px\.json$', f):
            continue
        p = root.split('/')
        try:
            i = p.index('results-med-reduce-clean')
        except ValueError:
            continue
        if len(p) < i + 4:
            continue
        domain, teacher, cat = p[i + 1], p[i + 2], p[i + 3]
        if cat not in ('baseline', 'resnet', 'tinyvit'):
            continue
        d = json.load(open(os.path.join(root, f)))
        ei = d['experiment_info']; a = d['accuracy_metrics']
        acc[(ei['dataset'], teacher, cat, ei['resolution'])][ei['seed']] = {
            'auroc': a['best_metric'], 'top1': a.get('final_val_acc'), 'f1': a.get('final_val_f1')}
        m = d.get('efficiency_metrics', {}).get('peak_gpu_memory_mb')
        if m is not None:
            mem[(ei['dataset'], teacher, cat, ei['resolution'])].append(m)

TEACHERS = [('dinov3', 'DINOv3'), ('vit', 'ViT'), ('biomedclip', 'BiomedCLIP')]
TCOL = {'dinov3': '#2176AE', 'vit': '#E8572A', 'biomedclip': '#57A773'}
ROLES = [('baseline', 'LP baseline (teacher)'), ('resnet', 'Distilled ResNet-50'), ('tinyvit', 'Distilled TinyViT')]  # capitalized 'Distilled'
METRICS = [('auroc', 'AUROC'), ('top1', 'Top-1 accuracy'), ('f1', 'Macro F1')]
TASKS = [
    ('images', 'dermatology', 'Dermatology (ISIC)'),
    ('tcga_luad_vs_lusc', 'pathology', 'Pathology: LUAD vs LUSC'),
    ('tcga_lgg_vs_gbm', 'pathology', 'Pathology: LGG vs GBM'),
    ('tcga_kras', 'pathology', 'Pathology: KRAS'),
    ('tcga_tp53', 'pathology', 'Pathology: TP53'),
    ('tcga_egfr', 'pathology', 'Pathology: EGFR'),
    ('combined_train_valid_chexpert_v1.0', 'radiology', 'Radiology (CheXpert)'),
]
DOMAINS = [('dermatology', 'Dermatology (ISIC)'), ('pathology', 'Pathology (TCGA)'), ('radiology', 'Radiology (CheXpert)')]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})
nT = len(TEACHERS); bw = 0.8 / nT; x = np.arange(len(RESO))


def ms(perseed_vals):
    if not perseed_vals:
        return None, None
    return st.mean(perseed_vals), (st.pstdev(perseed_vals) if len(perseed_vals) > 1 else 0.0)


def acc_fig(ds, domain, tlabel):
    """One stacked figure per task: rows = regimes, cols = metrics (3x3)."""
    fig, axes = plt.subplots(len(ROLES), len(METRICS), figsize=(13, 11))
    for ri, (role, rlabel) in enumerate(ROLES):
        for ci, (mkey, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            for j, (tk, tname) in enumerate(TEACHERS):
                means, sds = [], []
                for r in RESO:
                    vals = [v[mkey] for v in acc[(ds, tk, role, r)].values() if v.get(mkey) is not None]
                    m, s = ms(vals)
                    means.append(m if m is not None else 0); sds.append(s if s is not None else 0)
                ax.bar(x + (j - (nT - 1) / 2) * bw, means, bw, yerr=sds, capsize=2.5,
                       color=TCOL[tk], alpha=0.9, edgecolor='white', linewidth=0.5,
                       label=(tname if (ri == 0 and ci == 0) else None))
            ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RESO])
            ax.set_ylim(0.3, 1.0)
            if ri == len(ROLES) - 1:
                ax.set_xlabel("Resolution (px)")
            if ri == 0:
                ax.set_title(mlab, fontsize=13)
            if ci == 0:
                ax.set_ylabel(f"{rlabel}\nScore", fontsize=11)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle(f"{tlabel} — accuracy by regime (mean $\\pm$ SD, 3 seeds)", y=1.03, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(OUTDIR, f"acc_{domain}_{ds}.png")
    fig.savefig(out); plt.close(fig); print("saved", os.path.basename(out))


def mem_fig(ds, domain, tlabel):
    """One figure per task: 1x3 regime panels; a single resolution-independent memory value per teacher.

    Peak GPU memory is essentially resolution-independent (degraded inputs are upsampled to the
    encoder's native 512px), so we pool over resolutions rather than plotting a redundant per-resolution axis.
    """
    fig, axes = plt.subplots(1, len(ROLES), figsize=(11, 3.8), sharey=True)
    xt = np.arange(nT); top = 0
    for ci, (role, rlabel) in enumerate(ROLES):
        ax = axes[ci]
        means, sds, cols = [], [], []
        for tk, tname in TEACHERS:
            vals = [m for r in RESO for m in mem[(ds, tk, role, r)]]  # pool over resolutions + seeds
            means.append(st.mean(vals) if vals else 0)
            sds.append(st.pstdev(vals) if len(vals) > 1 else 0)
            cols.append(TCOL[tk])
        top = max(top, max(mn + sd for mn, sd in zip(means, sds)))
        ax.bar(xt, means, 0.6, yerr=sds, capsize=3, color=cols, alpha=0.9, edgecolor='white', linewidth=0.5)
        ax.set_xticks(xt); ax.set_xticklabels([tn for _, tn in TEACHERS], fontsize=10)
        ax.set_title(rlabel, fontsize=12)
        if ci == 0:
            ax.set_ylabel("Peak GPU memory (MB)")
    for ax in axes:
        ax.set_ylim(0, top * 1.2 if top else 1)
    fig.suptitle(f"{tlabel} — deployment memory by regime (mean $\\pm$ SD over resolutions and seeds)", y=1.03, fontsize=13)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f"mem_{domain}_{ds}.png")
    fig.savefig(out); plt.close(fig); print("saved", os.path.basename(out))


def run(preview_domain=None):
    os.makedirs(OUTDIR, exist_ok=True)
    for ds, domain, tlabel in TASKS:
        if preview_domain and domain != preview_domain:
            continue
        acc_fig(ds, domain, tlabel)
        mem_fig(ds, domain, tlabel)


if __name__ == '__main__':
    run(ONLY)
