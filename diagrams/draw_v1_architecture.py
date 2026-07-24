#!/usr/bin/env python3
"""
Architecture diagram for HeatmapGuidedDetector v1
(without-temporal(1).ipynb — heatmap-guided-sparse-detector-v1)

Generates a publication-quality PNG/PDF of the complete pipeline.

Usage:
    MPLBACKEND=Agg python draw_v1_architecture.py
    MPLBACKEND=Agg python draw_v1_architecture.py --output my_diagram.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Headless-safe default
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C = {
    "bg": "#FAFBFC",
    "input": "#D6EAF8",
    "backbone": "#FDEBD0",
    "temporal": "#E8DAEF",
    "bifpn": "#D5F5E3",
    "heatmap": "#FADBD8",
    "sparse": "#FCF3CF",
    "refine": "#D1F2EB",
    "loss": "#F5B7B1",
    "output": "#D6DBDF",
    "border": "#2C3E50",
    "arrow": "#34495E",
    "text": "#1B2631",
    "muted": "#566573",
    "accent": "#1A5276",
    "train": "#C0392B",
}


def rbox(ax, x, y, w, h, title, color, subtitle=None, fs=8, bold=False,
         lw=1.4, ls="-", alpha=1.0):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=color, edgecolor=C["border"],
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=2,
    )
    ax.add_patch(p)
    fw = "bold" if bold else "normal"
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C["text"], zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=fs - 1.5, color=C["muted"], zorder=3, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C["text"], zorder=3)
    return p


def arr(ax, x1, y1, x2, y2, label=None, rad=0.0, color=None, lw=1.5, ls="-",
        fs=7):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", color=color or C["arrow"], linewidth=lw,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        mutation_scale=13, zorder=1,
    )
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label,
                ha="center", va="bottom", fontsize=fs, color=C["muted"], zorder=4)


def section(ax, x, y, n, title):
    ax.text(x, y, f"{'①②③④⑤⑥'[n-1]}  {title}",
            fontsize=9.5, fontweight="bold", color=C["accent"], ha="left")


def draw_architecture(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # Title
    ax.text(10, 11.55,
            "Heatmap-Guided Sparse Detector v1",
            ha="center", fontsize=17, fontweight="bold", color=C["text"])
    ax.text(10, 11.05,
            "without-temporal(1).ipynb  ·  run: heatmap-guided-sparse-detector-v1  ·  "
            "~6.5M params  ·  640² input  ·  T=3  ·  no motion maps",
            ha="center", fontsize=9, color=C["muted"])

    # ══════════════════════════════════════════════════════════════════════
    # ROW 1 — INPUT  (y ≈ 9.5 – 10.8)
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 0.4, 10.85, 1, "INPUT")
    for i, (lbl, x) in enumerate([("Frame t−2", 0.4), ("Frame t−1", 2.9), ("Frame t (anchor)", 5.4)]):
        rbox(ax, x, 9.5, 2.2, 1.1, lbl, C["input"], "RGB  640×640",
             bold=(i == 2))

    rbox(ax, 8.0, 9.5, 2.4, 1.1, "Stack + Normalize", C["input"],
         "B × T × 3 × 640 × 640\nT = 3")
    for x in [1.5, 4.0, 6.5]:
        arr(ax, x + 1.1, 10.05, 8.0, 10.05)

    rbox(ax, 10.8, 9.5, 2.6, 1.1, "Reshape", C["input"],
         "(B·T) × 3 × 640 × 640")
    arr(ax, 10.4, 10.05, 10.8, 10.05)

    rbox(ax, 13.8, 9.5, 2.8, 1.1, "Temporal Augmentation", C["input"],
         "shared flip/scale\n+ per-frame noise")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 2 — BACKBONE + TEMPORAL FUSION  (y ≈ 7.0 – 8.8)
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 0.4, 8.55, 2, "BACKBONE  +  TEMPORAL FUSION")

    rbox(ax, 0.4, 7.0, 3.6, 1.5, "EfficientNet-B0", C["backbone"],
         "ImageNet pretrained\nfeatures_only  ·  out_indices (2,3,4)",
         bold=True, fs=9)
    arr(ax, 11.0, 9.5, 2.2, 8.5, rad=0.15)

    levels = [
        ("C3", "40 ch", "stride 8", "80×80", 7.6),
        ("C4", "112 ch", "stride 16", "40×40", 6.1),
        ("C5", "320 ch", "stride 32", "20×20", 4.6),
    ]
    for name, ch, st, sz, y in levels:
        rbox(ax, 4.4, y, 1.5, 1.0, name, C["backbone"],
             f"{ch}\n{st}  {sz}", fs=7)
        arr(ax, 4.0, 7.75, 4.4, y + 0.5, rad=0.08)

    for name, ch, st, sz, y in levels:
        rbox(ax, 6.3, y, 2.4, 1.0, "TemporalFusion", C["temporal"],
             "Conv1×1(cat T frames)\nBN + SiLU", fs=7)
        arr(ax, 5.9, y + 0.5, 6.3, y + 0.5)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 3 — BiFPN  (y ≈ 2.8 – 4.5)
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 0.4, 5.05, 3, "BiFPN  —  3 layers  ·  64 channels")

    rbox(ax, 0.4, 2.8, 4.2, 1.8, "BiFPN", C["bifpn"],
         "① Lateral 1×1 proj (C3,C4,C5)\n"
         "② ×3 BiFPNLayer: TD + BU fusion\n"
         "③ DW-Separable conv nodes\n"
         "④ P1 = Upsample(P2, 2×) + DW-Conv",
         bold=True, fs=8)

    for name, ch, st, sz, y in levels:
        arr(ax, 8.7, y + 0.5, 4.6, 3.7, rad=0.12)

    pyramid = [
        ("P1  ★", "stride 4", "160×160", 4.85, C["heatmap"]),
        ("P2", "stride 8", "80×80", 3.95, C["bifpn"]),
        ("P3", "stride 16", "40×40", 3.05, C["bifpn"]),
        ("P4", "stride 32", "20×20", 2.15, C["bifpn"]),
    ]
    for name, st, sz, y, col in pyramid:
        rbox(ax, 8.5, y, 1.6, 0.7, name, col, f"{st}  {sz}", fs=7,
             bold=("★" in name))
        arr(ax, 4.6, 3.5, 8.5, y + 0.35)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 4 — HEATMAP BRANCH  (right side, y ≈ 5.5 – 8.5)
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 9.2, 8.55, 4, "HEATMAP PROPOSAL  (sparse)")

    rbox(ax, 9.2, 7.4, 3.0, 1.1, "HeatmapHead", C["heatmap"],
         "2× DW-SepConv + Conv1×1\ninput: P1 (64 ch)", bold=True)
    arr(ax, 6.6, 4.7, 9.2, 7.9, rad=-0.15, label="P1")

    rbox(ax, 9.2, 5.9, 3.0, 1.1, "Heatmap Logits", C["heatmap"],
         "B × 1 × 160 × 160")
    arr(ax, 10.7, 7.4, 10.7, 7.0)

    rbox(ax, 9.2, 4.4, 3.0, 1.1, "Peak Extraction", C["sparse"],
         "sigmoid → 3×3 max-pool NMS\nTop-K = 50  (conf > 0.01)",
         bold=True, ls="--")
    arr(ax, 10.7, 5.9, 10.7, 5.5, ls="--")

    ax.text(12.5, 4.9, "inference\npath", fontsize=7, color=C["muted"],
            style="italic", ha="center")

    # ══════════════════════════════════════════════════════════════════════
    # ROW 5 — ROI REFINEMENT  (far right)
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 13.0, 8.55, 5, "ROI REFINEMENT")

    rbox(ax, 13.0, 7.4, 3.4, 1.1, "Multi-level ROIAlign", C["refine"],
         "levels P1 + P2 + P3\n7×7 per level → concat", bold=True)
    arr(ax, 12.2, 4.95, 13.0, 7.9, rad=0.1, label="peak (y,x)")
    arr(ax, 6.6, 3.8, 13.0, 7.6, rad=-0.2, label="FPN feats")

    rbox(ax, 13.0, 5.9, 3.4, 1.1, "Fixed Anchor Prior", C["sparse"],
         "8 × 8 px  at each peak\n(cx, cy, pw=8, ph=8)")
    arr(ax, 14.7, 7.4, 14.7, 7.0)

    rbox(ax, 13.0, 4.0, 3.4, 1.5, "RefinementHead  (MLP)", C["refine"],
         "flat 9408 → Linear 256\n→ LayerNorm → SiLU → Dropout\n"
         "→ Linear 128 → SiLU\n├─ objectness head (1)\n"
         "└─ box delta head (4)",
         bold=True, fs=7.5)
    arr(ax, 14.7, 5.9, 14.7, 5.5)

    # ══════════════════════════════════════════════════════════════════════
    # ROW 6 — OUTPUT
    # ══════════════════════════════════════════════════════════════════════
    section(ax, 13.0, 2.55, 6, "DECODE  &  OUTPUT")

    rbox(ax, 13.0, 0.8, 3.4, 1.3, "Final Detections", C["output"],
         "cx = anc_x + Δx·pw\ncy = anc_y + Δy·ph\n"
         "w  = pw·exp(Δw)\nconf ≥ 0.3  ·  NMS IoU 0.3",
         bold=True, fs=7.5)
    arr(ax, 14.7, 4.0, 14.7, 2.1)

    rbox(ax, 17.0, 0.8, 2.6, 1.3, "Output", C["output"],
         "list[{boxes, scores}]\nper image in batch")
    arr(ax, 16.4, 1.45, 17.0, 1.45)

    # Legend
    items = [
        (C["input"], "Input"), (C["backbone"], "Backbone"),
        (C["temporal"], "Temporal"), (C["bifpn"], "BiFPN"),
        (C["heatmap"], "Heatmap"), (C["sparse"], "Sparse prop."),
        (C["refine"], "Refinement"), (C["output"], "Output"),
    ]
    for i, (col, lbl) in enumerate(items):
        x = 0.4 + (i % 4) * 1.9
        y = 0.35 - (i // 4) * 0.38
        ax.add_patch(FancyBboxPatch(
            (x, y), 0.22, 0.18, boxstyle="round,pad=0.01",
            facecolor=col, edgecolor=C["border"], linewidth=0.7))
        ax.text(x + 0.3, y + 0.09, lbl, fontsize=7, va="center", color=C["muted"])

    plt.tight_layout(pad=0.4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=C["bg"])
    pdf = output_path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor=C["bg"])
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=Path(__file__).parent / "v1_architecture.png")
    draw_architecture(p.parse_args().output)


if __name__ == "__main__":
    main()
