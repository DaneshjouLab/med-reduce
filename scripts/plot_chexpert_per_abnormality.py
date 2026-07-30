#!/usr/bin/env python3
"""CheXpert per-abnormality resolution sensitivity: 2x4 grid of the 8 observation labels,
each panel showing AUROC vs resolution for the three frozen teachers (DINOv3, ViT-B/16, BiomedCLIP).
Reads per_class_auroc from results-med-reduce-clean/ (radiology teacher baselines). mean +/- SD, 3 seeds.
"""
import os, re, json, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN = os.environ.get('MR_RESULTS_CLEAN', 'results-med-reduce-clean')
OUT = os.path.join(os.environ.get('MR_PAPER_DIR', '.'), 'chexpert_per_abnormality.png')
RESO = [512, 256, 128, 64]
LABELS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Edema',
          'Pneumothorax', 'Pleural Effusion', 'Fracture', 'Support Devices']

# (label_idx, teacher, res) -> [auroc over seeds]
D = defaultdict(list)
root_rad = os.path.join(CLEAN, 'radiology')
for root, _, fs in os.walk(root_rad):
    if '/baseline/' not in root + '/':
        continue
    teacher = root.split('/')[root.split('/').index('radiology') + 1]
    for f in fs:
        if not re.match(r'results_.*px\.json$', f):
            continue
        d = json.load(open(os.path.join(root, f)))
        pca = d['accuracy_metrics'].get('per_class_auroc') or {}
        res = d['experiment_info']['resolution']
        for i in range(8):
            v = pca.get(str(i))
            if v is not None:
                D[(i, teacher, res)].append(v)

TEACHERS = [('dinov3', 'DINOv3'), ('vit', 'ViT'), ('biomedclip', 'BiomedCLIP')]
TCOL = {'dinov3': '#2176AE', 'vit': '#E8572A', 'biomedclip': '#57A773'}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})
nT = len(TEACHERS); bw = 0.8 / nT; x = np.arange(len(RESO))
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True)
for li, lab in enumerate(LABELS):
    ax = axes[li // 4][li % 4]
    for j, (tk, tname) in enumerate(TEACHERS):
        means, sds = [], []
        for r in RESO:
            vals = D[(li, tk, r)]
            means.append(st.mean(vals) if vals else 0)
            sds.append(st.pstdev(vals) if len(vals) > 1 else 0)
        ax.bar(x + (j - (nT - 1) / 2) * bw, means, bw, yerr=sds, capsize=2,
               color=TCOL[tk], alpha=0.9, edgecolor='white', linewidth=0.5,
               label=(tname if li == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels([str(r) for r in RESO], fontsize=9)
    ax.set_title(lab, fontsize=11)
    ax.set_ylim(0.5, 0.9)
    if li % 4 == 0:
        ax.set_ylabel("AUROC")
    if li // 4 == 1:
        ax.set_xlabel("Resolution (px)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("CheXpert per-abnormality AUROC vs. resolution, by teacher (mean $\\pm$ SD, 3 seeds)",
             y=1.06, fontsize=14)
fig.tight_layout()
fig.savefig(OUT); print("saved", OUT)

# console dump for the text
print("\n=== per-label AUROC (dinov3 teacher) 512 -> 64, drop ===")
for li, lab in enumerate(LABELS):
    a = st.mean(D[(li, 'dinov3', 512)]); b = st.mean(D[(li, 'dinov3', 64)])
    print(f"  {lab:26s} 512={a:.3f} 64={b:.3f} drop={a-b:+.3f}")
