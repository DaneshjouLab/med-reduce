#!/usr/bin/env python3
"""Per-domain Pareto frontier of AUROC vs encoder parameter count (model-intrinsic cost).

Params (M): DINOv3 ViT-S/16 = 21, ViT-B/16 = 86, BiomedCLIP ViT-B/16 = 86,
ResNet-50 = 25, TinyViT-21M = 21. Points = (teacher, regime, resolution); pathology AUROC
macro-averaged over the 5 TCGA tasks. Frontier maximizes AUROC while minimizing parameters.
"""
import os, re, json, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

CLEAN = os.environ.get('MR_RESULTS_CLEAN', 'results-med-reduce-clean')
OUT = os.path.join(os.environ.get('MR_PAPER_DIR', '.'), 'pareto_params.png')
RESO = [512, 256, 128, 64]

# encoder parameter count (millions) — role determines student size; teacher size for baseline
def params_M(teacher, role):
    if role == 'resnet':
        return 25
    if role == 'tinyvit':
        return 21
    return {'dinov3': 21, 'vit': 86, 'biomedclip': 86}[teacher]   # baseline = teacher encoder


au_raw = defaultdict(lambda: defaultdict(list))   # (domain,teacher,role,res) -> seed -> [auroc per task]
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
        ei = d['experiment_info']
        au_raw[(domain, teacher, cat, ei['resolution'])][ei['seed']].append(d['accuracy_metrics']['best_metric'])


def auroc(domain, teacher, role, res):
    per = au_raw[(domain, teacher, role, res)]
    if not per:
        return None
    return st.mean([st.mean(v) for v in per.values()])   # macro-avg over tasks, then seeds


TEACHERS = [('dinov3', 'DINOv3'), ('vit', 'ViT'), ('biomedclip', 'BiomedCLIP')]
TCOL = {'dinov3': '#2176AE', 'vit': '#E8572A', 'biomedclip': '#57A773'}
ROLES = [('baseline', 'Teacher (LP)', 'o'), ('resnet', 'ResNet-50', 's'), ('tinyvit', 'TinyViT', '^')]
RSIZE = {512: 150, 256: 95, 128: 55, 64: 30}
DOMAINS = [('dermatology', 'Dermatology (ISIC)'), ('pathology', 'Pathology (TCGA, macro-avg)'),
           ('radiology', 'Radiology (CheXpert)')]
JIT = {'dinov3': -1.4, 'vit': 0.0, 'biomedclip': 1.4}   # small x-jitter so teachers separate at equal params

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ci, (dom, dlab) in enumerate(DOMAINS):
    ax = axes[ci]
    pts = []
    for tk, _ in TEACHERS:
        for role, _, mk in ROLES:
            for r in RESO:
                au = auroc(dom, tk, role, r)
                if au is not None:
                    pts.append((params_M(tk, role), au, tk, role, r))
    for px, au, tk, role, r in pts:
        mk = dict((ro, m) for ro, _, m in ROLES)[role]
        ax.scatter(px + JIT[tk], au, s=RSIZE[r], c=TCOL[tk], marker=mk, alpha=0.8,
                   edgecolors='white', linewidths=0.6, zorder=10)
    # Pareto frontier: min params, max AUROC (use true params, not jittered)
    best = -1; front = []
    for px, au, *_ in sorted(pts, key=lambda z: z[0]):
        if au > best:
            front.append((px, au)); best = au
    if front:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color='#CC3333', ls='--', lw=2, alpha=0.8, zorder=5)
    ax.set_xlabel("Encoder parameters (M)")
    ax.set_xlim(15, 95)
    if ci == 0:
        ax.set_ylabel("AUROC")
    ax.set_title(dlab)

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=TCOL[t], markersize=9, label=n) for t, n in TEACHERS]
handles += [Line2D([0], [0], marker=m, color='w', markerfacecolor='#888', markersize=9, label=n) for _, n, m in ROLES]
handles += [Line2D([0], [0], color='#CC3333', ls='--', lw=2, label='Pareto frontier')]
handles += [Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbb', markersize=np.sqrt(RSIZE[r]) / 1.6, label=f'{r}px') for r in RESO]
fig.legend(handles=handles, loc='upper center', ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.09), fontsize=10)
fig.suptitle("Accuracy vs. encoder parameters with per-domain Pareto frontier (mean over 3 seeds)", y=1.15, fontsize=14)
fig.tight_layout()
fig.savefig(OUT); print("saved", OUT)

print("\n=== Pareto frontier (params M, AUROC, teacher/role/res) ===")
for dom, dlab in DOMAINS:
    pts = []
    for tk, _ in TEACHERS:
        for role, _, _ in ROLES:
            for r in RESO:
                au = auroc(dom, tk, role, r)
                if au is not None:
                    pts.append((params_M(tk, role), au, tk, role, r))
    best = -1; front = []
    for px, au, tk, role, r in sorted(pts, key=lambda z: z[0]):
        if au > best:
            front.append((px, au, tk, role, r)); best = au
    print(f"[{dlab}]")
    for px, au, tk, role, r in front:
        print(f"   {px:3d}M  AUROC={au:.3f}  {tk}/{role}/{r}px")
