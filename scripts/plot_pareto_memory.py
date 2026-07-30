#!/usr/bin/env python3
"""Per-domain Pareto frontier of AUROC vs peak GPU memory (mean over seeds), 512..64 px.

Points = (teacher, regime, resolution); pathology AUROC & memory macro-averaged over the 5 TCGA tasks.
Frontier maximizes AUROC while minimizing peak GPU memory. Reads results-med-reduce-clean/.
"""
import os, re, json, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

CLEAN = os.environ.get('MR_RESULTS_CLEAN', 'results-med-reduce-clean')
OUT = os.path.join(os.environ.get('MR_PAPER_DIR', '.'), 'pareto_memory.png')
RESO = [512, 256, 128, 64]

# (domain, teacher, role, res) -> per-seed lists (auroc averaged over tasks within domain per seed)
raw = defaultdict(lambda: defaultdict(lambda: {'au': [], 'mem': []}))  # key -> seed -> {au:[per-task],mem:[]}
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
        k = (domain, teacher, cat, ei['resolution'])
        raw[k][ei['seed']]['au'].append(d['accuracy_metrics']['best_metric'])
        m = d.get('efficiency_metrics', {}).get('peak_gpu_memory_mb')
        if m is not None:
            raw[k][ei['seed']]['mem'].append(m)


def agg(domain, teacher, role, res):
    per = raw[(domain, teacher, role, res)]
    if not per:
        return None
    au = st.mean([st.mean(v['au']) for v in per.values()])            # macro-avg over tasks, then over seeds
    memvals = [m for v in per.values() for m in v['mem']]
    mem = st.mean(memvals) if memvals else None
    return au, mem


TEACHERS = [('dinov3', 'DINOv3'), ('vit', 'ViT'), ('biomedclip', 'BiomedCLIP')]
TCOL = {'dinov3': '#2176AE', 'vit': '#E8572A', 'biomedclip': '#57A773'}
ROLES = [('baseline', 'Teacher (LP)', 'o'), ('resnet', 'ResNet-50', 's'), ('tinyvit', 'TinyViT', '^')]
RSIZE = {512: 150, 256: 95, 128: 55, 64: 30}
DOMAINS = [('dermatology', 'Dermatology (ISIC)'), ('pathology', 'Pathology (TCGA, macro-avg)'),
           ('radiology', 'Radiology (CheXpert)')]

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif"})
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ci, (dom, dlab) in enumerate(DOMAINS):
    ax = axes[ci]
    pts = []  # (mem, au, teacher, role, res)
    for tk, _ in TEACHERS:
        for role, _, mk in ROLES:
            for r in RESO:
                a = agg(dom, tk, role, r)
                if a and a[1] is not None:
                    pts.append((a[1], a[0], tk, role, r))
    # scatter
    for mem, au, tk, role, r in pts:
        mk = dict((ro, m) for ro, _, m in ROLES)[role]
        ax.scatter(mem, au, s=RSIZE[r], c=TCOL[tk], marker=mk, alpha=0.8,
                   edgecolors='white', linewidths=0.6, zorder=10)
    # Pareto frontier: min memory, max AUROC
    front = []
    best = -1
    for mem, au, *_ in sorted(pts, key=lambda z: z[0]):
        if au > best:
            front.append((mem, au)); best = au
    if front:
        fx, fy = zip(*front)
        ax.plot(fx, fy, color='#CC3333', ls='--', lw=2, alpha=0.8, zorder=5, label='Pareto frontier')
    ax.set_xlabel("Peak GPU memory (MB)")
    if ci == 0:
        ax.set_ylabel("AUROC")
    ax.set_title(dlab)

# combined legend
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=TCOL[t], markersize=9, label=n) for t, n in TEACHERS]
handles += [Line2D([0], [0], marker=m, color='w', markerfacecolor='#888', markersize=9, label=n) for _, n, m in ROLES]
handles += [Line2D([0], [0], color='#CC3333', ls='--', lw=2, label='Pareto frontier')]
handles += [Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbb', markersize=np.sqrt(RSIZE[r]) / 1.6, label=f'{r}px') for r in RESO]
fig.legend(handles=handles, loc='upper center', ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.09), fontsize=10)
fig.suptitle("Accuracy vs. peak GPU memory with per-domain Pareto frontier (mean over 3 seeds)", y=1.15, fontsize=14)
fig.tight_layout()
fig.savefig(OUT); print("saved", OUT)

# print frontier membership per domain for the text
print("\n=== Pareto frontier members (mem MB, AUROC, teacher/role/res) ===")
for dom, dlab in DOMAINS:
    pts = []
    for tk, _ in TEACHERS:
        for role, rl, _ in ROLES:
            for r in RESO:
                a = agg(dom, tk, role, r)
                if a and a[1] is not None:
                    pts.append((a[1], a[0], tk, role, r))
    best = -1; front = []
    for mem, au, tk, role, r in sorted(pts, key=lambda z: z[0]):
        if au > best:
            front.append((mem, au, tk, role, r)); best = au
    print(f"[{dlab}]")
    for mem, au, tk, role, r in front:
        print(f"   {mem:6.0f}MB  AUROC={au:.3f}  {tk}/{role}/{r}px")
