#!/usr/bin/env python3
"""
Generate a degradation panel showing downsample→upsample artifacts per domain.

Expects 2 sample images per domain in data/sample/.

Produces: figures/degradation_panel.pdf and .png

Each row = one image, columns = 512px (original) → 256 → 128 → 64px
All images are upsampled back to 512px to match the actual pipeline input.

Dependencies: numpy, matplotlib, Pillow (no torch/torchvision needed)
Optional: scikit-image (for SSIM; falls back to PSNR-only if unavailable)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_DIR = Path("data/sample")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NATIVE = 512
RESOLUTIONS = [512, 256, 128, 64]

DOMAINS = {
    "Dermatology": [
        ("ISIC_0001118.jpg", "Melanoma"),
        ("ISIC_0012375.jpg", "Seborrheic Keratosis"),
    ],
    "Radiology": [
        ("patient00001_study1_view1_frontal.jpg", "Support Device"),
        ("patient00003_study1_view1_frontal.jpg", "Edema"),
    ],
    "Pathology": [
        ("fe1e8496-3f3f-41a3-a716-7f14928f9002.jpg", "TCGA-LUSC"),
        ("ffef9df6-4ddc-4aef-b84b-8db49cbc5fff.jpg", "TP53"),
    ],
}

# ---------------------------------------------------------------------------
# Degradation transform (matches the pipeline — bilinear resize via PIL)
# ---------------------------------------------------------------------------
def degrade(img: Image.Image, target_res: int, native_res: int = NATIVE) -> Image.Image:
    """Downsample to target_res then upsample back to native_res."""
    img = img.resize((target_res, target_res), Image.BILINEAR)
    if target_res != native_res:
        img = img.resize((native_res, native_res), Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def compute_psnr(original: Image.Image, degraded: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher = less distortion."""
    a = np.asarray(original, dtype=np.float64)
    b = np.asarray(degraded, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(original: Image.Image, degraded: Image.Image) -> float | None:
    """Structural Similarity Index. 1.0 = identical. None if skimage unavailable."""
    if not HAS_SKIMAGE:
        return None
    a = np.asarray(original)
    b = np.asarray(degraded)
    return _ssim_fn(a, b, channel_axis=2, data_range=255)


# ---------------------------------------------------------------------------
# Collect images
# ---------------------------------------------------------------------------
rows = []  # (domain_label, sublabel, fname, pil_images, metrics)
for domain, entries in DOMAINS.items():
    for fname, sublabel in entries:
        path = SAMPLE_DIR / fname
        if not path.exists():
            print(f"WARNING: {path} not found — skipping")
            continue
        img = Image.open(path).convert("RGB")
        # Resize original to NATIVE for consistent comparison
        original = img.resize((NATIVE, NATIVE), Image.BILINEAR)
        degraded = [degrade(img, r) for r in RESOLUTIONS]
        metrics = []
        for d in degraded:
            metrics.append({
                "psnr": compute_psnr(original, d),
                "ssim": compute_ssim(original, d),
            })
        rows.append((domain, sublabel, fname, degraded, metrics))

if not rows:
    print("No images found. Place sample images in data/sample/ and re-run.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
n_rows = len(rows)
n_cols = len(RESOLUTIONS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4.0 * n_cols, 4.0 * n_rows),
    squeeze=False,
)

BADGE_COLORS = {512: "#2ecc71", 256: "#f39c12", 128: "#e74c3c", 64: "#8e44ad"}

for row_idx, (domain, sublabel, fname, images, metrics) in enumerate(rows):
    for col_idx, (res, img, m) in enumerate(zip(RESOLUTIONS, images, metrics)):
        ax = axes[row_idx][col_idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column headers
        if row_idx == 0:
            if res == NATIVE:
                ax.set_title(f"{res}px (original)", fontsize=20, fontweight="bold")
            else:
                ax.set_title(f"{res}\u2192{NATIVE}px", fontsize=20, fontweight="bold")

        # Resolution badge (top-left)
        color = BADGE_COLORS[res]
        badge = mpatches.FancyBboxPatch(
            (4, 4), 75, 28,
            boxstyle="round,pad=3",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.85,
            transform=ax.transData,
        )
        ax.add_patch(badge)
        badge_fontsize = 11
        ax.text(
            40, 18, f"{res}px",
            ha="center", va="center", fontsize=badge_fontsize, fontweight="bold",
            color="white", transform=ax.transData,
        )

        # Quality metrics (bottom-left) — skip for original
        if res != NATIVE:
            psnr_val = m["psnr"]
            ssim_val = m["ssim"]
            if ssim_val is not None:
                label = f"PSNR {psnr_val:.1f} dB\nSSIM {ssim_val:.3f}"
            else:
                label = f"PSNR {psnr_val:.1f} dB"
            ax.text(
                0.03, 0.03, label,
                transform=ax.transAxes,
                fontsize=13, fontfamily="monospace",
                color="white", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=3),
                va="bottom", ha="left",
            )

        # Compression ratio (bottom-right)
        if res != NATIVE:
            ratio = (NATIVE / res) ** 2
            ax.text(
                0.97, 0.03, f"{ratio:.0f}\u00d7",
                transform=ax.transAxes,
                fontsize=16, fontweight="bold",
                color="white",
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="white", pad=3),
                va="bottom", ha="right",
            )

    # Row label: domain (bold) + sublabel (smaller italic)
    ax0 = axes[row_idx][0]
    ax0.text(
        -0.02, 0.55, domain,
        transform=ax0.transAxes,
        fontsize=18, fontweight="bold",
        ha="right", va="bottom", rotation=0,
    )
    ax0.text(
        -0.02, 0.45, sublabel,
        transform=ax0.transAxes,
        fontsize=14, fontstyle="italic",
        ha="right", va="top", rotation=0,
        color="#555555",
    )

plt.tight_layout()

out_path = OUTPUT_DIR / "degradation_panel.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=200)
print(f"Saved: {out_path}")

out_png = OUTPUT_DIR / "degradation_panel.png"
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"Saved: {out_png}")
