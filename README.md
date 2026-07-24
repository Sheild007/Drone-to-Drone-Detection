# Drone-to-Drone Detection

Custom sparse heatmap detector on the NPS-Drones dataset, benchmarked against YOLO and RT-DETR baselines.

## Architecture

`model.py` has the full definition:

- EfficientNet-B0 backbone, features at 3 scales
- TemporalFusion — 1×1 conv over T=3 concatenated frame features per scale
- 3-layer BiFPN → P1 at stride 4 (160×160)
- Motion maps: `|frame_t − frame_{t-1}|` and `|frame_t − frame_{t-2}|`, fused into the heatmap head
- Peak extraction: sigmoid → 3×3 max-pool NMS → top-50 candidates
- Multi-level ROIAlign on P1/P2/P3 (11×11) → MLP → objectness + box deltas

Phase 3: adaptive Gaussian sigma per object size, data-driven anchor (5.93 px), DIoU proposal matching.

## Results

Baselines — every 4th test frame, Tesla T4:

| Model | AP@50 | F1 | FPS |
|-------|-------|----|-----|
| YOLO11n | 0.673 | 0.663 | 261.6 |
| YOLO11s | 0.729 | 0.694 | 154.2 |
| YOLO11m | 0.718 | 0.688 | 55.2 |
| YOLO12s | 0.854 | 0.862 | 117.7 |
| RT-DETRv1 | 0.913 | 0.893 | 22.3 |

Custom model — T=3 triplets, Tesla T4:

| Variant | AP@50 | F1 | FPS |
|---------|-------|----|-----|
| v1 (no motion) | 0.609 | 0.753 | 49.2 |
| v2 (3-anchor) | 0.649 | 0.748 | 41.3 |
| Phase 1 | 0.633 | 0.736 | 47.4 |
| **Phase 3** | **0.772** | **0.812** | **49.0** |

## Setup

```bash
pip install -r requirements.txt

# shape check — no data needed
python model.py
```
