# Complete Project Guide — Heatmap-Guided Drone Detector (Phase 3)

**Read this document top to bottom.** It is written so someone with basic deep-learning knowledge (knows CNNs, loss, training) can understand the **whole project** — not just Phase 3 tricks, but the **base idea**, **why each block exists**, and **how data actually flows**.

Notebook: `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb`

---

## Part 0 — Start here: what are we even building?

### The real-world problem
You have a camera on one drone filming another tiny drone in the sky. The target is **absurdly small** — often only **6×8 pixels** in a 640×640 image. It can look like noise, a bird speck, or nothing at all.

### What a normal detector does (YOLO-style)
YOLO places thousands of **anchor boxes** at many grid locations and asks at each one: “Is there an object here? Adjust the box.”

That works, but at stride 4 you have **160×160 = 25,600 locations**. Running heavy box regression everywhere is wasteful when 99.99% of the image is empty sky.

### Our core idea (base architecture — all versions)
Inspired by **CenterNet**:

1. **Find centers first** — paint a heatmap: “drone probably here”
2. **Keep only top-K peaks** (~50) — ignore the rest of the sky
3. **Refine boxes only at those peaks** — small MLP + ROIAlign

Inspired by **YOLOMG** (from v2 onward):

4. **Use motion** — subtract consecutive frames; moving drones pop out even when RGB is weak

```text
THREE FRAMES IN  →  WHERE IS THE CENTER?  →  TOP 50 PEAKS  →  TIGHT BOXES OUT
```

That pipeline is the **base architecture**. Phase 3 only improves *how we train* (adaptive targets, DIoU assignment, subsampling) — the skeleton is the same.

---

## Part 1 — Vocabulary you need (no jargon without explanation)

| Term | Plain meaning |
|------|----------------|
| **Backbone** | A pretrained CNN that turns an image into **feature maps** (richer, smaller tensors). Like compressing the image into “what the network sees.” |
| **Feature map** | A 2D grid of vectors. Each cell summarizes a region of the image. |
| **Stride** | How much one cell “covers” in the original image. Stride 8 → each cell ≈ 8×8 pixels of input. |
| **Channels** | Depth of the feature map. Each channel is one type of pattern (edge, texture, etc.). |
| **Neck (FPN/BiFPN)** | Combines feature maps at different scales so the model sees both **fine detail** and **context**. |
| **Head** | Small layers on top that produce the actual task output (heatmap, boxes). |
| **Heatmap** | A 2D map where bright = “object center likely here.” |
| **ROIAlign** | Crops a fixed grid of features around a point — like zooming into a region in feature space. |
| **Anchor** | A starting guess box the network refines (not YOLO’s dense anchors — we use **one** per peak). |

---

## Part 2 — Input: what goes into the model?

### 2.1 Temporal triplets — why 3 frames?

We don’t use one image. We use **3 consecutive frames** from the same video clip:

```text
Frame i−2   Frame i−1   Frame i  ← we detect drones on THIS frame only
   │            │          │
   └────────────┴──────────┘
         one training sample
```

**Why?** A single frame may not show the drone clearly. Two older frames give **history** — motion, trajectory, faint appearance.

**Labels:** bounding boxes come **only from frame `i`** (the last / “anchor” frame). Earlier frames have no separate labels in the loss — they’re context.

**How triplets are built:** group images by clip (`Clip_001_...`), sort by frame number, sliding window of size 3 → ~32k train triplets.

### 2.2 Motion maps — what and why?

After aligning all 3 frames (same crop/flip), convert to grayscale:

```text
anchor     = gray frame i
motion_ch1 = |gray i  − gray i−1|
motion_ch2 = |gray i  − gray i−2|
```

**Intuition:** Static clouds/sky → difference ≈ 0. A moving drone → pixels change → bright blob in the diff map.

**Why not only RGB?** At 6 px, RGB texture is almost useless. Motion is an extra **“something moved here”** signal (YOLOMG paper insight).

Shape: `(2, 640, 640)` — two single-channel images.

### 2.3 What the dataset returns per sample

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `frames` | (3, 3, 640, 640) | 3 RGB frames, ImageNet normalized |
| `motion` | (2, 640, 640) | 2 diff maps |
| `heatmap` | (1, 160, 160) | training target — Gaussian bumps at GT centers |
| `gt_boxes` | (64, 4) | xyxy boxes in pixels on frame `i` |
| `gt_mask` | (64,) | 1 = real box, 0 = padding |

---

## Part 3 — EfficientNet-B0 and C3, C4, C5 (the backbone)

This is the part most people get stuck on. Read slowly.

### 3.1 What is EfficientNet-B0?

**EfficientNet-B0** is a standard image classifier CNN — winner of the “accuracy vs speed” trade-off in 2019. We don’t use its classification head. We use it as a **feature extractor**.

```python
timm.create_model("efficientnet_b0", pretrained=True,
                  features_only=True, out_indices=(2, 3, 4))
```

- **pretrained=True:** ImageNet weights — already knows edges, shapes, textures
- **features_only=True:** stop before “which class is this?” — output intermediate layers
- **out_indices=(2,3,4):** take outputs from **3 internal stages** — we name them C3, C4, C5

**Why EfficientNet-B0 and not ResNet-50?** Smaller, faster (~49 FPS target), still strong enough when paired with BiFPN. ResNet would work but heavier.

### 3.2 What are C3, C4, C5?

Think of the network as a **pyramid of summaries**:

```text
Original image     640 × 640   ← full resolution, 3 RGB channels
       │
   [layer blocks]
       │
C3 feature map      80 × 80    stride 8   ← “medium detail”
       │
C4 feature map      40 × 40    stride 16  ← “coarser”
       │
C5 feature map      20 × 20    stride 32  ← “big picture / context”
```

| Name | Grid size | Stride | Channels | Each cell “sees” roughly |
|------|-----------|--------|----------|--------------------------|
| **C3** | 80×80 | 8 | 40 | 8×8 px patch of input |
| **C4** | 40×40 | 16 | 112 | 16×16 px |
| **C5** | 20×20 | 32 | 320 | 32×32 px |

**Why three levels?**
- **C3** — still fine enough for **tiny** objects (drone might span ~1 cell)
- **C4** — neighborhood context
- **C5** — “sky vs ground”, large-scale scene

**Analogy:** C3 = reading letters, C4 = reading words, C5 = reading the paragraph.

### 3.3 How the backbone runs on video

We have 3 frames. The backbone does **not** magically fuse time yet.

```text
Batch shape: (B, 3, 3, 640, 640)   →  B samples, 3 frames, RGB

Reshape to: (B×3, 3, 640, 640)     →  treat as 3×B independent images
Run EfficientNet on each
Get C3, C4, C5 per frame
```

So after backbone, for each scale you have features for frame i−2, i−1, and i **separately**. Temporal fusion comes **next**.

---

## Part 4 — TemporalFusion: merging 3 frames with a 1×1 conv

### 4.1 The problem it solves

Backbone gave you 3 separate feature maps per scale — one per time step. The detector needs **one** fused map per scale that encodes “what changed / what persisted across time.”

### 4.2 What it does — step by step (concrete example at C3)

For one scale (say C3, 40 channels, 80×80):

```text
Frame i−2 features:  (B, 40, 80, 80)
Frame i−1 features:  (B, 40, 80, 80)
Frame i   features:  (B, 40, 80, 80)
```

**Step 1 — Stack along channels** (after reshaping batch×time):

```text
(B, 40×3, 80, 80) = (B, 120, 80, 80)
```

Each spatial position now holds a **120-dimensional vector**: 40 numbers from t−2, 40 from t−1, 40 from t.

**Step 2 — 1×1 convolution: 120 → 40**

A 1×1 conv is just a **learned linear mix** at every pixel:

```text
new_feature[pixel] = W × concat(f_t-2, f_t-1, f_t) + b
```

No spatial blending — only **channel mixing** at the same (x,y).

**Step 3 — BatchNorm + SiLU activation**

Output: `(B, 40, 80, 80)` — **one** fused C3 map.

### 4.3 Intuition — why this works

The network **learns** rules like:
- “If frame t brightened here but t−1 didn’t → maybe motion”
- “If all three frames agree on an edge → strong static structure”

You don’t hand-design temporal logic. The 1×1 conv is the **cheapest** way to let gradients learn it.

**Why not LSTM / Transformer?** Too heavy for 49 FPS on a T4. 1×1 conv over stacked frames is ~free compared to video transformers.

**We have 3 TemporalFusion modules** — one each for C3, C4, C5 (different channel widths).

### 4.4 What TemporalFusion is NOT

- It does **not** align moving objects (alignment already done in data augmentation — same crop on all 3 frames)
- It does **not** replace motion maps — motion diffs are a **separate explicit branch** fused later into the heatmap head

---

## Part 5 — BiFPN neck: mixing scales

### 5.1 Why we need a neck

C3/C4/C5 live at different resolutions. The heatmap head wants **one fine grid (160×160)** with **rich semantics**. BiFPN:

1. Projects all levels to **64 channels**
2. Runs **top-down + bottom-up** fusion (BiFPN from EfficientDet)
3. **Extra upsample** creates **P1 at stride 4** → **160×160**

| Output | Stride | Size | Role |
|--------|--------|------|------|
| **P1** | 4 | 160×160 | Heatmap + finest ROI features |
| P2 | 8 | 80×80 | ROI level 1 |
| P3 | 16 | 40×40 | ROI level 2 |
| P4 | 32 | 20×20 | Built but not used in ROI |

**Why stride 4 matters:** 6 px drone ÷ 4 = **1.5 heatmap cells** — you can still localize a sub-pixel target. Stride 8 would crush it to <1 cell.

### 5.2 Intuition

BiFPN answers: “At this pixel, combine **local detail** (fine level) with **what the coarse level thinks** (context).” Learnable weights decide how much to trust each source.

---

## Part 6 — MotionEncoder + HeatmapHead

### MotionEncoder

```text
motion (2, 640, 640)
  → pool down to P1 size (160×160)   ← cheap: don’t convolve at full res
  → 2 small conv blocks → 64 channels
```

Fused with P1 in HeatmapHead:

```text
concat(P1, motion_feat) → 1×1 conv → 64 ch
→ 2× depthwise conv → 1×1 → 1 logit map (160×160)
```

**Output:** `hm_logits` — raw scores before sigmoid. High = “center here.”

### Training target (heatmap)

Place a **Gaussian bump** at each GT box center on the 160×160 grid.

**Phase 3 — adaptive width:**

```python
σ = max(min(box_w, box_h) / 3, 0.5)   # in heatmap pixels
```

Small drone → narrow bump (precise). Larger drone → wider bump.

**Bias init trick:** last conv bias ≈ −4.6 so untrained sigmoid ≈ **0.01 everywhere**. Without this, the model predicts “drone everywhere” on day 1 (25,600 negatives vs ~3 positives).

---

## Part 7 — Peaks → ROI → boxes (the sparse trick)

### 7.1 extract_peaks

```text
sigmoid(hm_logits) → 3×3 max-pool NMS → keep top 50 local maxima
```

**Why:** 160×160 = 25,600 cells. We only refine **50**. That’s the speed secret.

### 7.2 Data-driven anchor (~5.93 px)

At each peak, place a **square box** centered on the peak. Side length = **median GT box size** measured from training labels (~5.93 px).

**Why median from data?** Hand-picking 8 px or 16 px is arbitrary. Median says “typical drone is ~6 px — start there, MLP adjusts.”

**Why one anchor?** Multi-anchor (v2) needed extra heads + selection loss. One anchor + delta regression is simpler and stable for NPS where sizes are similar.

### 7.3 ROIAlign + RefinementHead

For each peak, sample **11×11 feature grids** from P1, P2, P3 → flatten → MLP:

- **objectness:** is this peak a real drone?
- **box deltas:** `(dx, dy, dw, dh)` tweak the 5.93 px anchor

Decode → filter conf > 0.3 → NMS IoU 0.3 → final boxes.

---

## Part 8 — Losses (what the network is trained to do)

```text
Total = 1.0 × heatmap_focal + 1.0 × objectness + 2.0 × box_regression
```

| Loss | Trains what | Intuition |
|------|-------------|-----------|
| **Heatmap focal** | Dense 160×160 map | “Paint bright at true centers, stay dark elsewhere” — handles extreme class imbalance |
| **Objectness** | Each of 50 ROIs | “This peak is / isn’t a drone” |
| **Box L1** | Positive ROIs only | “Shift and resize the anchor to match GT” |

**Phase 3 — DIoU for assignment only:** When deciding if an ROI is positive, use DIoU (IoU minus center-distance penalty) instead of raw IoU. For 6 px boxes, a 2 px shift kills IoU but DIoU still says “close enough to learn from.”

**DIoU is NOT the box loss** — box loss is still L1 on deltas.

---

## Part 9 — Base architecture (v1) vs later versions

Understanding the **base** helps you see what Phase 3 added.

### v1 — base sparse detector (no motion)

```text
3 frames → EfficientNet → TemporalFusion → BiFPN → HeatmapHead (RGB only)
→ peaks → ROI 7×7 → refine with fixed 8×8 anchor
```

Proved sparse heatmap design works: **AP50 0.609, 49 FPS**.

### v2 — add motion + multi-anchor

- MotionEncoder + diff maps
- 3 anchor sizes (8, 16, 24) + selection loss
- **AP50 0.649** but slower (41 FPS)

### Phase 1/2 — simplify anchors, bigger ROI

- Single data-driven anchor (~5.93 px)
- ROI 7×7 → **11×11** (finer spatial sampling)
- Motion encoder downsamples first (faster)
- **AP50 0.633, 47 FPS**

### Phase 3 — training improvements (this notebook)

Same **architecture skeleton** as Phase 1/2. Changes:

| Change | What | Why |
|--------|------|-----|
| Adaptive σ | Heatmap target width per box | Tiny vs larger drones need different supervision |
| DIoU assignment | ROI ↔ GT matching | IoU unstable at 6 px |
| 35% epoch subsample | Random 35% of train triplets per epoch | More epoch diversity in limited GPU time |
| Focal loss FP32 clamp | Numerical stability | Prevented training freeze ~epoch 10–12 |

**Result: AP50 0.772, F1 0.812, ~49 FPS**

---

## Part 10 — Full forward pass (one sentence per stage)

Read this as a story:

1. **Load** frames `[i−2, i−1, i]` and motion diffs vs frame `i`.
2. **EfficientNet** on each frame → C3, C4, C5 (multi-scale appearance).
3. **TemporalFusion** at each scale → one fused map per scale (learned temporal mix).
4. **BiFPN** → P1..P4 pyramid; P1 is 160×160 (fine enough for 6 px drones).
5. **MotionEncoder** pools diffs to P1, convolves → motion features.
6. **HeatmapHead** fuses P1 + motion → 160×160 center map.
7. **extract_peaks** → 50 brightest local maxima (ignore rest of sky).
8. **ROIAlign** at each peak on P1,P2,P3 → big feature vector.
9. **RefinementHead** → drone or not + box tweak relative to ~6 px anchor.
10. **Decode + NMS** → final boxes on frame `i`.

---

## Part 11 — Training loop (how it learns)

- **Optimizer:** AdamW; backbone/motion at **0.1× LR**, heads at **1× LR** (don’t destroy pretrained features early)
- **Schedule:** 5-epoch warmup + cosine decay
- **Batch:** 4 × grad accum 4 = effective 16
- **Each epoch:** new random 35% of train triplets; **full val** every epoch
- **Resume:** from epoch 32 checkpoint, train to epoch 60

---

## Part 12 — Evaluation (how numbers are produced)

- **COCO AP50, AP50-95, F1** via `pycocotools`
- Test on **all 12,335 triplets** (not every-4th-frame like YOLO baselines — **don’t compare blindly**)
- **FPS:** batch=1, 3 frames, end-to-end GPU timing

---

## Part 13 — FAQ (the “why / how” gaps)

**Q: Why heatmap instead of YOLO anchors?**  
A: 25,600 locations vs 50 peaks. Compute goes where the model already suspects a drone.

**Q: Why 3 frames specifically?**  
A: Enough for motion/temporal fusion; not so many that latency explodes.

**Q: Does TemporalFusion see motion explicitly?**  
A: It learns from stacked RGB features. **Motion maps are separate** — explicit pixel diffs fed into heatmap head. Both paths complement each other.

**Q: What is stride 4 in plain English?**  
A: Heatmap cell (x,y) corresponds to a 4×4 pixel block in the original 640 image.

**Q: Why pretrained ImageNet on sky/drone data?**  
A: Low-level filters (edges, blobs) transfer. Fine-tuning adapts higher layers.

**Q: Why is the anchor square if drones aren’t?**  
A: Starting guess is square; `dw, dh` learn aspect ratio. One scalar `anchor_size` keeps the head simple.

**Q: Phase 1 vs Phase 2?**  
A: Identical code; Phase 2 was a continued training run.

---

## Part 14 — Results at a glance

| Model | Motion | Anchor | ROI | AP50 | FPS |
|-------|--------|--------|-----|------|-----|
| v1 base | No | 8 px fixed | 7×7 | 0.609 | 49.2 |
| v2 | Yes | multi | 7×7 | 0.649 | 41.3 |
| Phase 1/2 | Yes | 5.93 px data | 11×11 | 0.633 | 47.4 |
| **Phase 3** | Yes | 5.93 px | 11×11 | **0.772** | **49.0** |

Best baseline (YOLO12s, different eval protocol): AP50 0.854, 118 FPS.

---

## Part 15 — Related files

| File | Use |
|------|-----|
| This document | Full intuitive architecture guide |
| `docs/README_CUSTOM_EXPERIMENTS.md` | Variant tables, config reference, presentation script |
| `docs/README_BASELINES_AND_PIPELINE.md` | YOLO / RT-DETR baselines |
| `experiments/heatmap_based_detection_motiondiff_epoch_subsampling_phase3.ipynb` | Phase 3 training code |

---

## Part 16 — 60-second explanation (memorize this)

> “We detect 6-pixel drones in video using three frames. EfficientNet extracts appearance features per frame; a 1×1 conv fuses the three timelines at each scale. BiFPN builds a fine 160×160 map. Motion difference maps highlight moving pixels and merge into a center heatmap. We keep only 50 peaks, ROIAlign features around each, and a small network refines a ~6-pixel anchor into a tight box. Phase 3 improves training with size-adaptive heatmap targets, better tiny-box assignment, and subsampled epochs. We hit 0.77 AP at 49 FPS.”
