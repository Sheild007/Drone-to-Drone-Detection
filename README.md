# Drone-to-Drone Detection

Semester project for ITU's Deep Learning course (Spring 2026). Detect tiny airborne drones from another drone using temporal RGB frames and motion difference maps — without dense anchor grids.

## The problem

NPS-Drones has 50 video clips of a delta-wing drone filmed from a second drone mid-air. Targets are 4–8 pixels wide at 640×640. Dense anchor-based detectors work but spend most of their compute classifying sky pixels. We wanted something sparser.

## What we built

A CenterNet-style detector — predict a heatmap first, extract the top-50 candidate locations, then run ROIAlign + MLP only on those 50 spots. Compared against YOLO11n/s/m, YOLO12s, and RT-DETRv1 as baselines.

**Architecture** — `model.py` has the full definition:
- EfficientNet-B0 backbone, features extracted at 3 scales
- TemporalFusion: 1×1 conv over T=3 concatenated frame features per scale
- 3-layer BiFPN → P1 at stride 4 (160×160) — necessary for sub-10px targets
- Motion maps: `|frame_t − frame_{t-1}|` and `|frame_t − frame_{t-2}|`, encoded and fused into the heatmap head (YOLOMG-style)
- Peak extraction: sigmoid → 3×3 max-pool NMS → top-50 peaks
- Multi-level ROIAlign on P1/P2/P3 (11×11) → MLP → objectness + box deltas

Phase 3 refinements: adaptive Gaussian sigma per object size, data-driven anchor size (5.93 px median from training labels), DIoU-based proposal matching.

## Results

Baselines — every 4th test frame, Tesla T4:

| Model | AP@50 | F1 | FPS | FPPI |
|-------|-------|----|-----|------|
| YOLO11n | 0.673 | 0.663 | 261.6 | 5.23 |
| YOLO11s | 0.729 | 0.694 | 154.2 | 4.15 |
| YOLO11m | 0.718 | 0.688 | 55.2 | 3.71 |
| YOLO12s | 0.854 | 0.862 | 117.7 | 3.27 |
| RT-DETRv1 | 0.913 | 0.893 | 22.3 | 78.57 |

Our model — T=3 triplets, Tesla T4:

| Variant | AP@50 | F1 | FPS |
|---------|-------|----|-----|
| v1 (no motion) | 0.609 | 0.753 | 49.2 |
| v2 (3-anchor) | 0.649 | 0.748 | 41.3 |
| Phase 1 | 0.633 | 0.736 | 47.4 |
| **Phase 3** | **0.772** | **0.812** | **49.0** |

RT-DETRv1 has higher AP but 78 false positives per image — a problem if each triggers an avoidance maneuver.

## Files

```
Drone-to-Drone-Detection/
 ├── model.py                    — standalone model class (importable)
 ├── requirements.txt
 ├── result.jpeg                 — sample detection output
 ├── diagrams/
 │    ├── draw_v1_architecture.py
 │    └── v1_architecture.png
 ├── experiments/
 │    ├── heatmap_based_detection_without_motion_diff.ipynb     — v1
 │    ├── hestmap-with temporal.ipynb                           — v2
 │    ├── heatmap_based_detection_motiondiff_epoch_subsampling_phase1.ipynb
 │    ├── heatmap_based_detection_motiondiff_epoch_subsampling_phase2.ipynb
 │    └── heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb
 └── baselines/
      ├── yolo11n.ipynb
      ├── yolo11s.ipynb
      ├── Baseline yolo v12s.ipynb
      └── baseline deter.ipynb
```

## Quick start

```bash
pip install -r requirements.txt

# shape check — no data needed
python model.py

# regenerate architecture diagram
python diagrams/draw_v1_architecture.py
```

All training runs on Kaggle (Tesla T4). Dataset: NPS-Drones — search it on Kaggle, it's public.

## References

- Li et al., *Vision-based Autonomous Obstacle Avoidance for Small UAVs*, ICUAS 2016 — NPS-Drones dataset
- Law & Deng, *CornerNet*, ECCV 2018 — heatmap formulation
- Sangam et al., *TransVisDrone*, CVPR 2023
- Guo et al., *YOLOMG*, 2025 — motion difference maps
