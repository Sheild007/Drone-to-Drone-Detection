# Study Guide 2: Custom Experiments, Architecture & All Variations

Use this document to **fully understand and present** our heatmap-guided sparse detector: every component, every variant (v1 → v2 → Phase 1/2/3), and how they connect.

> **New to the project?** Read **`PHASE3_HEATMAP_DETECTOR_ARCHITECTURE.md` first** — it explains the base architecture, EfficientNet/C3–C5, TemporalFusion, and every *why/how* with intuition. This file is the variant reference + results tables.

---

## Core idea (one sentence)

> Predict a **heatmap of drone centers** on a coarse grid, keep only **top-K peaks**, then **refine tiny boxes** with ROIAlign — instead of scanning the full image with dense YOLO anchors.

Inspired by **CenterNet** (sparse keypoints) + **YOLOMG** (motion difference maps).

---

## End-to-end pipeline (all variants)

```text
Input: T=3 consecutive RGB frames (640×640)
        │
        ├─[optional] MotionEncoder ← |I_anchor − I_t| for t−1, t−2
        │
        ▼
EfficientNet-B0 (per frame) → C3, C4, C5 features
        │
        ▼
TemporalFusion (1×1 conv per scale)  ← merges 3 frames at each level
        │
        ▼
BiFPN (3 layers) → P1(s4, 160×160), P2, P3, P4
        │
        ▼
HeatmapHead on P1 [+ motion] → hm_logits (1×160×160)
        │
        ▼
extract_peaks: sigmoid → local NMS → top-K=50 centers
        │
        ▼
ROIAlign on P1,P2,P3 at each peak → feature vector per candidate
        │
        ▼
RefinementHead (MLP) → objectness + box deltas (dx,dy,dw,dh)
        │
        ▼
decode_boxes + conf filter (0.3) + NMS (IoU 0.3)
        │
        ▼
Final bounding boxes
```

**When presenting:** Walk left-to-right on `diagrams/v1_architecture.png` — motion branch appears from v2 onward.

---

## Notebook map (which file is which)

| Name | Notebook | What it is |
|------|----------|------------|
| **v1 (base)** | `experiments/heatmap_based_detection_without_motion_diff.ipynb` | Appearance-only, fixed 8×8 anchor |
| **v2 (multi-anchor)** | `experiments/hestmap-with temporal.ipynb` | Motion + 3 anchor sizes |
| **Phase 1** | `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase1.ipynb` | Single data-driven anchor, 11×11 ROI |
| **Phase 2** | `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase2.ipynb` | **Same code as Phase 1** (separate Kaggle run) |
| **Phase 3** | `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb` | Adaptive σ, DIoU assignment, 35% subsampling |

Root copies (`without-temporal(1).ipynb`, `epoch-subsapling-phase*.ipynb`) are duplicates of the `experiments/` versions.

---

## Shared configuration (`CFG` dict)

Same across all custom notebooks unless noted:

```python
CFG = {
    "project": "drone-detection-nps",
    "seed": 42,
    "data_root": "/kaggle/input/datasets/kamikazeeeeeee88",
    "img_size": 640,
    "temporal_len": 3,           # T=3 frames per clip
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "backbone_out_channels": [40, 112, 320],  # C3, C4, C5
    "bifpn_channels": 64,
    "bifpn_layers": 3,
    "heatmap_stride": 4,         # 640/4 = 160×160 heatmap
    "topk": 50,                  # max proposals per image
    "num_classes": 1,
    "refine_hidden": 256,
    "roi_levels": [0, 1, 2],     # P1, P2, P3
    "epochs": 60,
    "batch_size": 4,
    "accum_steps": 4,            # effective batch = 16
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "cosine_min_lr": 1e-6,
    "clip_grad": 2.0,
    "heatmap_loss_w": 1.0,
    "obj_loss_w": 1.0,
    "box_loss_w": 2.0,
    "focal_alpha": 2.0,
    "focal_beta": 4.0,
    "nms_iou_thr": 0.3,
    "conf_thr": 0.3,
    "amp": True,
}
```

**Dataset after triplet building:** 32,148 train / 3,745 val / 12,335 test triplets.

---

## Component reference (know each class)

### 1. EfficientNet-B0 backbone
```python
timm.create_model("efficientnet_b0", pretrained=True,
                  features_only=True, out_indices=(2, 3, 4))
```
- Runs **independently on each of the 3 frames**
- Outputs 3 scales: C3 (s8), C4 (s16), C5 (s32)
- Channels: 40, 112, 320

**Why EfficientNet-B0?** Light enough for ~50 FPS; ImageNet pretrained.

---

### 2. TemporalFusion
```python
# Per backbone level:
Conv2d(ch * T, ch, kernel_size=1) → BatchNorm → SiLU
```
- Concatenates features from all T frames along channel dim
- Lets model see **appearance change across time** without a transformer
- Applied at all 3 backbone scales before BiFPN

**Present as:** “Cheap temporal fusion — 1×1 conv over stacked frames.”

---

### 3. BiFPN (3-layer neck)
- **BiFPNNode:** weighted sum of 2–3 feature inputs + depthwise separable conv
- **BiFPNLayer:** top-down then bottom-up pass
- Lateral 1×1 projections → 64 channels
- Extra upsample creates **P1 at stride 4** (160×160) — critical for tiny drones

**Outputs:** `[P1, P2, P3, P4]` at strides 4, 8, 16, 32

---

### 4. MotionEncoder (v2, Phase 1/2/3 only)

**Motion input** (2 channels):
```python
anchor = gray_frame[-1]   # last frame of triplet
diff_1 = |anchor - gray_frame[-2]|
diff_2 = |anchor - gray_frame[-3]|
```

| Variant | MotionEncoder design |
|---------|---------------------|
| **v2** | Convs at full 640×640, pooled to P1 size in forward |
| **Phase 1/2/3** | Downsample motion first, then 2 conv blocks → 64 ch |

```python
Conv2d(2, 32, 3) → BN → SiLU → Conv2d(32, 64, 3) → BN → SiLU
```

**Why motion?** Tiny drones are hard to see in one frame; pixel differences highlight movement (YOLOMG insight).

---

### 5. HeatmapHead
- Input: P1 features [+ concatenated motion features in v2+]
- Stack: 2× DepthwiseSeparableConv → 1×1 conv → 1 channel logits
- **Bias init:** `-4.595` so sigmoid ≈ 0.01 (avoids all-background collapse)

**Training targets:**
- Gaussian blobs centered on GT box centers in heatmap space (160×160)
- **Fixed σ=2.0** (v1, v2, Phase 1/2)
- **Adaptive σ** (Phase 3): `σ = max(min(w,h)/3, 0.5)` per object

**Loss:** CenterNet-style focal loss (`FocalLossHeatmap`, α=2, β=4)

---

### 6. Peak extraction (`extract_peaks`)
```python
hm = sigmoid(hm_logits)
# 3×3 max-pool NMS on heatmap
# keep top-K=50 peaks above conf_thr=0.01
```
→ Sparse list of `(y, x)` centers in heatmap coordinates

**Key design choice:** Only ~50 locations get expensive ROI refinement, not 160×160 = 25,600.

---

### 7. ROIAlign
- For each peak: convert to image coords `peak × stride(4)`
- Sample fixed grid from P1, P2, P3 at that location
- **Grid size:** 7×7 (v1, v2) or **11×11** (Phase 1/2/3)
- Concatenate level features → flat vector per candidate

---

### 8. RefinementHead (single-anchor variants)

```python
flat_dim = 3 * 64 * roi_size * roi_size   # 9,408 (7×7) or 23,232 (11×11)
MLP: Linear → LayerNorm → SiLU → Dropout → Linear → SiLU
obj_head: Linear(128, 1)    # objectness logit
box_head: Linear(128, 4)    # (dx, dy, dw, dh) relative to anchor
```

### 8b. MultiAnchorRefinementHead (v2 only)
- Same MLP trunk
- **3 parallel heads** for anchor sizes 8, 16, 24 px
- `anchor_select`: Linear(128, 3) — picks best anchor size at inference

---

### 9. Box decoding
```python
cx = anc_cx + dx * anc_w
cy = anc_cy + dy * anc_h
w  = anc_w * exp(clamp(dw, -4, 4))
h  = anc_h * exp(clamp(dh, -4, 4))
# convert to xyxy, filter conf > 0.3, NMS iou=0.3
```

---

## Loss functions (total training objective)

### v1 / Phase 1/2/3
```
L = 1.0·L_heatmap + 1.0·L_objectness + 2.0·L_box
```

### v2 (multi-anchor)
```
L = 1.0·L_heatmap + 1.0·L_objectness + 2.0·L_box + 0.5·L_selection
```

| Loss | What it does |
|------|--------------|
| **L_heatmap** | Focal loss on Gaussian center map |
| **L_objectness** | BCE on ROI candidates (pos/neg by IoU thresholds) |
| **L_box** | Smooth L1 on `(dx,dy,dw,dh)` for positive ROIs |
| **L_selection** (v2) | Cross-entropy: which anchor size best matches GT |

### ROI assignment thresholds

| Variant | Positive IoU | Negative IoU | Assignment metric |
|---------|-------------|--------------|-------------------|
| v1, v2 | ≥ 0.15 | < 0.05 | IoU |
| Phase 1/2/3 | ≥ 0.25 | < 0.05 | IoU |
| **Phase 3** | ≥ 0.25 | < 0.05 | **DIoU** (distance-aware) |

---

## Dataset: T=3 triplet construction

### Building triplets
```python
def build_temporal_triplets(pairs, T=3):
    # Group frames by clip ID (e.g. "Clip_001" from filename)
    # Sliding window: frames[i : i+3] → one training sample
```

### One `DroneDataset` sample returns
```python
{
    "frames":    (3, 3, 640, 640),   # T × RGB, ImageNet normalized
    "motion":    (2, 640, 640),      # v2+ only
    "hm_target": (1, 160, 160),     # Gaussian heatmap
    "gt_boxes":  (64, 4),           # xyxy pixels, anchor frame
    "gt_mask":   (64,),              # valid box mask
}
```

### Augmentation rules
- **Same spatial transform** applied to all T frames (Albumentations)
- Labels taken from **last (anchor) frame**
- Train aug: blur, noise, HSV, flip, shift-scale-rotate, crop

### Evaluation protocol (different from baselines!)
- Overlapping T=3 triplets on test → **12,335 samples**
- COCO JSON built from last frame of each triplet
- Metrics via `pycocotools.COCOeval`: AP50, AP50-95, AP25, F1

**Present carefully:** Custom numbers are **not directly comparable** to every-4th-frame baselines without harmonization.

---

## Architecture evolution (master table)

| Module | v1 | v2 | Phase 1/2 | Phase 3 |
|--------|----|----|-----------|---------|
| **Notebook** | `heatmap_..._without_motion_diff` | `hestmap-with temporal` | `..._phase1` / `..._phase2` | `..._phase3` |
| **run_name** | `sparse-detector-v1` | `v2-multianchor-motion` | `single-anchor-motion-v3` | `v3-adaptive-sigma-diou` |
| **Input** | RGB ×3 | RGB + motion | RGB + motion | RGB + motion |
| **MotionEncoder** | — | full-res conv | downsample-first | downsample-first |
| **Heatmap σ** | fixed 2.0 | fixed 2.0 | fixed 2.0 | **adaptive per box** |
| **Anchor at ROI** | fixed 8×8 px | 8 / 16 / 24 px | median GT **5.93 px** | median GT 5.93 px |
| **ROI grid** | 7×7 | 7×7 | **11×11** | 11×11 |
| **Refinement head** | single | multi-anchor | single | single |
| **Box loss** | Smooth L1 | Smooth L1 | Smooth L1 | Smooth L1 |
| **Assignment** | IoU | IoU | IoU 0.25 | **DIoU** |
| **Epoch subsample** | 100% | 100% | 100% | **35% random/epoch** |
| **Params** | 6.49M | 6.52M | 10.06M | 10.06M |

---

## Variant-by-variant story (what changed and why)

### v1 — Base appearance model
**File:** `experiments/heatmap_based_detection_without_motion_diff.ipynb`

**Design:**
- No motion maps — temporal info only via TemporalFusion of RGB
- Fixed **8×8 px** anchor (matches typical NPS GT size)
- 7×7 ROI grid
- Simplest proof that sparse heatmap pipeline works at ~49 FPS

**Test results:** AP50 **0.609**, F1 0.753, FPS 49.2

**Say:** “v1 validates the sparse proposal idea without motion engineering.”

---

### v2 — Motion + multi-anchor
**File:** `experiments/hestmap-with temporal.ipynb`

**Added:**
1. **MotionEncoder** + 2 diff maps fused into heatmap head
2. **MultiAnchorRefinementHead** with 3 priors (8, 16, 24 px)
3. **Selection loss** — network learns which anchor size fits each peak

**Trade-off:** +0.04 AP50 over v1 but slower (41.3 FPS) due to motion encoding

**Test results:** AP50 **0.649**, F1 0.748, FPS 41.3

**Say:** “Motion helps detection; multi-anchor handles scale variation but adds compute.”

---

### Phase 1 — Data-driven anchor + larger ROI
**File:** `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase1.ipynb`

**Changes from v2:**
- Drop multi-anchor → **single anchor = median GT size (5.93 px)**
- Enlarge ROIAlign **7×7 → 11×11** (finer spatial detail around peak)
- MotionEncoder: downsample-first (more efficient)
- **Partial checkpoint load** from 7×7 model: backbone + heatmap transfer; MLP re-init (shape change)

**Test results:** AP50 **0.633**, F1 0.736, FPS 47.4

**Say:** “We simplified to one data-driven anchor and gave the refiner more spatial context.”

---

### Phase 2 — Full 11×11 training
**File:** `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase2.ipynb`

**Important:** Source code is **100% identical to Phase 1**. Separate notebook = separate Kaggle training run continuing from Phase 1 checkpoint without partial-load constraint.

**Test evaluation:** Not completed — omitted from report tables.

---

### Phase 3 — Best custom model
**File:** `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb`

**Three key additions:**

1. **Adaptive Gaussian heatmap targets**
   ```python
   sigma = max(min(box_w, box_h) / 3, 0.5)   # in heatmap pixel space
   ```
   Tighter supervision for sub-6 px drones.

2. **DIoU assignment** (not loss — used to match ROIs to GT)
   ```python
   DIoU = IoU - d²(center_dist) / c²(enclosing_diag²)
   ```
   Better for tiny objects where IoU alone is noisy.

3. **Epoch subsampling (35%)**
   ```python
   epoch_subsample_fraction = 0.35
   # Each epoch: random 35% of 32,148 triplets → more epochs in same GPU time
   ```
   Full val set every epoch; only train subset is sampled.

**Also:** FP32-forced focal loss, logit clamping, resume from epoch 32 checkpoint (full load, ROI sizes match).

**Test results:** AP50 **0.772**, AP50-95 **0.307**, F1 **0.812**, FPS **49.0**  
**Val peak:** AP50 0.855

**Say:** “Phase 3 combines motion, adaptive supervision, better ROI-GT matching, and subsampling — our best result, approaching YOLO12s while staying sparse.”

---

## Training loop (what each epoch does)

```text
for epoch in 1..60:
    if epoch_subsample_fraction < 1.0:
        train_dl = random 35% subset of triplets   # Phase 3 only
    else:
        train_dl = full train set

    for batch in train_dl:
        with autocast:
            hm_logits, obj_logits, box_deltas, ... = model(frames, motion)
            loss = focal(hm) + obj_loss + 2*box_loss [+ 0.5*sel_loss for v2]
        scaler.scale(loss).backward()
        every 4 steps: optimizer.step(), grad_clip=2.0

    validate on full val set → COCO AP50
    if best AP50: save checkpoint_best.pt
    log to W&B + upload checkpoint artifact
```

### Optimizer
```python
# Two param groups:
backbone + temporal_fusion [+ motion_encoder]: lr * 0.1
heatmap + refine heads:                       lr * 1.0
# AdamW + cosine schedule after 5-epoch warmup
```

### Checkpoint files
```text
/kaggle/working/checkpoints/
├── checkpoint_last.pt
├── checkpoint_best.pt
└── checkpoint_epoch_NNN.pt
```

### Partial load (Phase 1 only)
```python
load_checkpoint_partial(...)
# Loads 534/535 tensors
# SKIPS: refine_head.mlp.0.weight  (9408 → 23232 input dim change)
```

---

## Inference / evaluation pipeline

```text
model.eval()
frames, motion → forward → hm_logits
extract_peaks(topk=50, conf_thr=0.01)
ROIAlign + RefinementHead
decode_predictions(conf_thr=0.3, nms_iou=0.3)
COCOEvaluator → AP50, AP50-95, AP25, F1
measure_fps() → end-to-end FPS on T4
```

---

## Results summary (memorize for presentation)

### Custom models (T=3 triplets, Tesla T4)

| Model | AP50 | AP50-95 | AP25 | F1 | FPS |
|-------|------|---------|------|-----|-----|
| v1 (base) | 0.609 | 0.198 | 0.760 | 0.753 | 49.2 |
| v2 (multi-anc.) | 0.649 | 0.216 | 0.812 | 0.748 | 41.3 |
| Phase 1 | 0.633 | 0.200 | 0.800 | 0.736 | 47.4 |
| **Phase 3** | **0.772** | **0.307** | **0.891** | **0.812** | **49.0** |

### vs best baseline
| | YOLO12s (baseline) | Phase 3 (ours) |
|--|-------------------|----------------|
| AP50 | 0.854 | 0.772 |
| FPS | 117.7 | 49.0 |
| Protocol | every 4th frame | T=3 triplets |

---

## Evolution lineage (diagram)

```text
v1  appearance-only, 8px anchor, 7×7 ROI
 │
 ├─ v2  + motion diff, multi-anchor [8,16,24], sel loss
 │    │
 │    └─ Phase 1/2  single 5.93px anchor, 11×11 ROI, partial ckpt from 7×7
 │         │
 │         └─ Phase 3  adaptive σ + DIoU assign + 35% subsample  ← BEST
```

---

## Class & function checklist (for code walkthrough)

### Model classes
`DepthwiseSeparableConv`, `BiFPNNode`, `BiFPNLayer`, `BiFPN`, `TemporalFusion`, `MotionEncoder`, `HeatmapHead`, `RefinementHead`, `MultiAnchorRefinementHead`, `HeatmapGuidedDetector`, `FocalLossHeatmap`, `DroneDataset`, `COCOEvaluator`

### Key functions
| Function | Purpose |
|----------|---------|
| `build_temporal_triplets` | Sliding T-frame windows per video clip |
| `compute_motion_diffs` | 2-channel motion maps from grayscale |
| `compute_data_driven_anchor_size` | Median GT box dimension → anchor |
| `compute_adaptive_sigma` | Phase 3 per-object heatmap width |
| `extract_peaks` | Top-K heatmap maxima |
| `compute_refinement_loss` | Objectness + box loss on ROIs |
| `decode_predictions` | Deltas → xyxy boxes + NMS |
| `get_epoch_subset_dataloader` | Phase 3 random 35% subset |
| `load_checkpoint_partial` | Phase 1 ROI resize transfer |

---

## How to present custom experiments (3-minute script)

> “Our custom detector avoids dense YOLO anchors. We take three consecutive frames, fuse them with EfficientNet and a light BiFPN, and predict a center heatmap at stride 4. Only the top 50 peaks get refined with ROIAlign — that’s where we save compute.
>
> We evolved in four stages. v1 proved the sparse design at 49 FPS but only 0.61 AP. v2 added YOLOMG-style motion difference maps and multi-scale anchors. Phases 1 and 2 moved to a data-driven 5.9 pixel anchor and an 11×11 ROI grid for finer refinement. Phase 3 added adaptive Gaussian targets for tiny boxes, DIoU-based ROI assignment, and 35% epoch subsampling to fit more training diversity in limited GPU time.
>
> Phase 3 reaches 0.77 AP at 49 FPS — our best result — closing much of the gap to YOLO12s while using an interpretable heatmap we can visualize.”

Then show `heatmap_predictions.jpeg`: bright peaks on the tiny drone.

---

## Common questions & answers

**Q: Why heatmap instead of YOLO anchors?**  
A: At stride 4 we have 160×160 = 25,600 locations. Dense anchors are expensive. Heatmap + top-50 focuses compute where the model already believes a drone exists.

**Q: Why T=3 frames?**  
A: Short clip captures motion without Video-Swin cost. Matches practical latency for onboard pipelines.

**Q: Phase 1 vs Phase 2?**  
A: Same code, different training continuation. Phase 2 test eval was never run.

**Q: Is DIoU the box regression loss in Phase 3?**  
A: No — box regression is still Smooth L1 on deltas. DIoU is used only to **assign** which ROIs are positive/negative.

**Q: Why 35% subsampling?**  
A: Each epoch randomly samples 35% of triplets so we can run more effective epochs within Kaggle time limits while seeing diverse clips.

---

## Related document

For **baselines, preprocessing, YOLO/RT-DETR training and FPPI**, read:

**`docs/README_BASELINES_AND_PIPELINE.md`**
