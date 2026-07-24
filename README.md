# Drone-to-Drone Detection on NPS-Drones

Vision-based detection of small airborne drones filmed from another drone.  
Built for ITU's Deep Learning course (Spring 2026).

---

## What this is

The NPS-Drones dataset contains 50 video clips of a delta-wing drone filmed from a second airborne platform. Targets are tiny — typically **4–8 pixels wide** at 640×640 input — and move against cluttered sky backgrounds. This makes standard anchor-dense detectors overkill: most of the image is background, and dense classification over thousands of anchor boxes wastes compute on pixels that are obviously not drones.

We took a different approach. Instead of predicting a bounding box at every anchor location, we first narrow the search to a sparse set of candidate locations using a CenterNet-style heatmap, then refine only those ~50 candidates with ROIAlign and an MLP head. The result is a detector that does less work per image and runs at real-time speed on a single GPU.

Alongside the custom model, we benchmarked five off-the-shelf detectors (YOLO11n/s/m, YOLO12s, RT-DETRv1) as honest baselines.

---

## Project structure

```
Drone-to-Drone-Detection/
│
├── model.py                          ← standalone model definition (importable)
├── requirements.txt
│
├── experiments/
│   ├── heatmap_based_detection_without_motion_diff.ipynb   ← v1 (appearance only)
│   ├── hestmap-with temporal.ipynb                         ← v2 (3 anchors)
│   ├── heatmap_based_detection_motiondiff_epoch_subsampling_phase1.ipynb
│   ├── heatmap_based_detection_motiondiff_epoch_subsampling_phase2.ipynb
│   └── heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb  ← best
│
├── baselines/
│   ├── yolo11n.ipynb
│   ├── yolo11s.ipynb
│   ├── Baseline yolo v12s.ipynb
│   └── baseline deter.ipynb
│
├── diagrams/
│   ├── draw_v1_architecture.py       ← generates the v1 architecture figure
│   └── v1_architecture.png
│
└── docs/
    ├── README_BASELINES_AND_PIPELINE.md
    └── README_CUSTOM_EXPERIMENTS.md
```

All training ran on Kaggle with a Tesla T4 GPU.

---

## Architecture

The model is called `HeatmapGuidedDetector`. The data flow is:

```
3 consecutive RGB frames (640×640)
  → EfficientNet-B0 (features_only, out_indices 2,3,4)
  → TemporalFusion per scale (1×1 conv over T concatenated frames)
  → 3-layer BiFPN (64 ch) → P1 (stride 4, 160×160), P2, P3, P4
  → YOLOMG-style motion maps: |frame_t - frame_{t-1}|, |frame_t - frame_{t-2}|
  → MotionEncoder (downsampled to P1 res, 2 conv blocks)
  → HeatmapHead (P1 + motion features fused, 2× DW-SepConv, output 160×160)
  → extract_peaks: sigmoid → 3×3 max-pool NMS → top-50 candidates
  → Multi-level ROIAlign on P1/P2/P3 (11×11 grid per level)
  → RefinementHead MLP → objectness + (dx, dy, dw, dh) deltas
```

**Why stride 4?** At 640×640 input a 6-pixel drone projects onto roughly 1–2 heatmap cells at stride 4 (160×160 map). At stride 8 or 16 you'd need sub-pixel accuracy in the heatmap just to localize the target — those strides work for people or cars, not for objects this small.

**Why sparse instead of dense anchors?** With top-K=50 candidates, the refinement head only processes 50 ROIs per image regardless of resolution. Dense anchor heads (YOLO-style) generate tens of thousands of proposals that must each be classified, which is where the inference cost goes.

**Motion maps.** Inspired by YOLOMG: subtracting consecutive grayscale frames highlights moving objects and suppresses static background. For a hovering or slow-moving drone this signal is subtle, but for a drone crossing the frame it produces a visible blob. The encoder processes these at P1 resolution (not full 640×640) to keep the compute cost low.

### Phase 3 changes

The model went through four training phases. Phase 3 introduces:

| Change | Rationale |
|--------|-----------|
| Adaptive Gaussian sigma: σ = max(min(w,h)/3, 0.5) | Fixed σ=2.0 over-spreads supervision for sub-6-pixel targets. Adaptive sigma tightens the target for small objects. |
| Data-driven anchor (~5.93 px, computed from training labels) | The median ground-truth box dimension from 4,000 label measurements. Removes the guess-work from fixed anchors. |
| DIoU assignment for ROI-GT matching | For a 6-pixel box, a 2-pixel center offset collapses standard IoU to near zero. DIoU subtracts center-distance / enclosing-box-diagonal, which remains positive for nearby-but-not-overlapping proposals. |
| 11×11 ROIAlign grid (up from 7×7) | More spatial resolution in the ROI feature for box refinement. |
| 35% epoch subsampling | Each epoch trains on a randomly sampled 35% of training triplets. More epoch diversity within the Kaggle GPU budget. |

---

## Results

### Spatial baselines (every 4th test frame, Tesla T4)

| Model | AP@50 | Precision | Recall | F1 | FPS | FPPI |
|-------|-------|-----------|--------|----|-----|------|
| YOLO11n | 0.673 | 0.799 | 0.567 | 0.663 | 261.6 | 5.23 |
| YOLO11s | 0.729 | 0.820 | 0.601 | 0.694 | 154.2 | 4.15 |
| YOLO11m | 0.718 | 0.832 | 0.588 | 0.688 | 55.2 | 3.71 |
| **YOLO12s** | **0.854** | **0.917** | 0.812 | **0.862** | **117.7** | 3.27 |
| RT-DETRv1 | 0.913 | 0.917 | 0.870 | 0.893 | 22.3 | 78.57 |

RT-DETRv1 hits 0.91 AP but its FPPI (false positives per image) is 78× higher than YOLO12s — fine for a controlled experiment, not for a real system where one wrong detection per frame triggers an avoidance maneuver.

### Custom heatmap detector (T=3 triplets, Tesla T4)

| Model | AP@50 | AP@25 | F1 | FPS |
|-------|-------|-------|----|-----|
| v1 (appearance only) | 0.609 | 0.760 | 0.753 | 49.2 |
| v2 (3-anchor) | 0.649 | 0.812 | 0.748 | 41.3 |
| Phase 1 | 0.633 | 0.800 | 0.736 | 47.4 |
| **Phase 3** | **0.772** | **0.891** | **0.812** | **49.0** |

Phase 3 reaches 0.77 AP@50 and 0.81 F1 at 49 FPS. The gap to YOLO12s (0.854) is partly a protocol difference: YOLO trains and evaluates on single frames (every 4th), while our detector processes overlapping T=3 triplets — roughly 4× more test samples — which includes harder frames near clip boundaries.

### Published methods on NPS-Drones (for reference)

| Method | AP@50 | FPS | Notes |
|--------|-------|-----|-------|
| TransVisDrone (1280²) | 0.95 | 24.6 | RTX A6000 |
| YOLOMG-640 | 0.92 | 133 | Authors' hardware |
| Dogfight | 0.89 | ~1.0 | Patch-based |

---

## Setup

```bash
pip install -r requirements.txt
```

The notebooks run on Kaggle with the NPS-Drones dataset mounted from Kaggle datasets. The dataset is public — search "NPS Drone" on Kaggle.

To run a quick shape-check of the model without any data:

```bash
python model.py
```

Expected output:
```
hm_logits  : torch.Size([2, 1, 160, 160])
obj_logits : torch.Size([N, 1])
box_deltas : torch.Size([N, 4])
Parameters : ~10.06M
Shape check PASSED.
```

To regenerate the architecture diagram:

```bash
python diagrams/draw_v1_architecture.py
```

---

## Dataset

**NPS-Drones** (Li et al., 2016): 50 clips of a delta-wing drone filmed from another airborne drone. Labels are YOLO-format axis-aligned boxes, single class `drone`.

| Split | Videos | Frames |
|-------|--------|--------|
| Train | #01–#36 | 32,220 |
| Val   | #37–#40 | 3,753  |
| Test  | #41–#50 | 12,355 |

Public Kaggle release — no sign-up required beyond a Kaggle account.

---

## References

1. Li et al., *Vision-based Autonomous Obstacle Avoidance for Small UAVs*, ICUAS 2016. — NPS-Drones dataset.
2. Law & Deng, *CornerNet: Detecting Objects as Paired Keypoints*, ECCV 2018. — CenterNet heatmap formulation.
3. Sangam et al., *TransVisDrone: Spatiotemporal Transformer for Vision-based Drone-to-Drone Detection in Aerial Videos*, CVPR 2023.
4. Guo et al., *YOLOMG: Tiny Drone Detection with Motion Difference*, 2025.
5. Ashraf et al., *Dogfight: Detecting Drones from Drones Videos*, arXiv 2021.

---

## License

Academic / course project use. NPS-Drones dataset terms apply to the data.
