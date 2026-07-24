# Study Guide 1: Baselines, Preprocessing & Full Pipeline

Use this document to **understand, explain, and present** the spatial baseline side of the project: data setup, training, evaluation, and how each notebook fits together.

---

## What you are building (big picture)

```text
Kaggle NPS-Drones raw folders
        │
        ▼
  Symlink into Ultralytics layout  (/kaggle/working/nps/)
        │
        ▼
  Write nps_drone.yaml (train / val / test paths)
        │
        ▼
  Train YOLO or RT-DETR  (100 epochs, 640², AdamW)
        │
        ▼
  Build every-4th-frame test subset  (3,089 images)
        │
        ▼
  model.val()  →  AP50, precision, recall, F1, FPS
        │
        ▼
  compute_fppi_fixed()  →  false positives per image
        │
        ▼
  Log to W&B + baseline_results.csv
```

**Goal:** fair comparison with TransVisDrone and Dogfight on NPS-Drones using the same test protocol.

---

## 1. The dataset (NPS-Drones)

### What it is
- 50 videos of a small delta-wing drone filmed from another airborne platform
- Single class: `drone`
- Labels: YOLO format — `class cx cy w h` (normalized 0–1)
- Typical box size at 640²: about **6–8 pixels** wide/tall

### Official splits (same in all notebooks)

| Split | Videos | Frames | Purpose |
|-------|--------|--------|---------|
| Train | #01–#36 | 32,220 | Learn detector |
| Val   | #37–#40 | 3,753  | Pick best epoch |
| Test  | #41–#50 | 12,355 | Final numbers |

### Where files live on Kaggle

```text
/kaggle/input/datasets/kamikazeeeeeee88/
├── nps-drone-train-part1/images_train, labels_train   (18,118 imgs)
├── nps-drone-train-part2/images_train, labels_train   (14,102 imgs)
├── nps-drone-val/images_val, labels_val               (3,753 imgs)
└── nps-drone-test/images_test, labels_test            (12,355 imgs)
```

**When presenting:** “We did not collect new data. We used the public Kaggle release and followed the TransVisDrone train/val/test video split.”

---

## 2. Preprocessing pipeline (step by step)

All four main baseline notebooks (`yolo11n`, `yolo11s`, `Baseline yolo v12s`, `baseline deter`) do the same data prep.

### Step A — Build Ultralytics folder layout

Raw Kaggle data is split across part1/part2. We **symlink** (not copy) into one tree:

```text
/kaggle/working/nps/
├── images/
│   ├── train/   ← part1 + part2 images
│   ├── val/
│   └── test/
└── labels/
    ├── train/   ← matching .txt labels
    ├── val/
    └── test/
```

**Code pattern:**
```python
os.symlink(img_path.resolve(), dst_img)
# label symlinked only if .txt exists
```

**Why symlinks?** Saves disk; Ultralytics expects `images/` + `labels/` mirror structure.

### Step B — Write dataset YAML

Path: `/kaggle/working/nps_drone.yaml`

```yaml
path: /kaggle/working/nps
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: drone
```

Variable in notebooks: `YAML_PATH = "/kaggle/working/nps_drone.yaml"`

Ultralytics reads this for `model.train(data=YAML_PATH)` and `model.val(data=...)`.

### Step C — Every-4th-frame test subset (evaluation only)

**Training uses all frames.** Testing follows TransVisDrone / Dogfight:

```python
def get_every_4th_frame_paths(test_img_dir):
    all_frames = sorted(glob("*.png") + glob("*.jpg"))
    subset = all_frames[::4]   # indices 0, 4, 8, ...
    # 12,355 → 3,089 frames
```

Temporary eval tree:
```text
/kaggle/working/tmp_test_every4th/
├── images/   ← symlinks to every 4th test frame
└── labels/   ← symlinks to matching labels
```

Subset YAML: `/kaggle/working/subset_test.yaml`

**When presenting:** “We evaluate on every 4th test frame — 3,089 images — so our numbers are comparable to published NPS-Drones papers.”

---

## 3. Shared training config (`CONFIG` dict)

Frozen across YOLO11n, YOLO11s, YOLO12s, RT-DETRv1 (except batch size overrides):

| Category | Key settings |
|----------|--------------|
| **Input** | `imgsz=640`, `single_cls=True`, COCO pretrained |
| **Optimizer** | AdamW, `lr0=3e-5`, `lrf=0.1`, `momentum=0.843`, `weight_decay=0.0005` |
| **Schedule** | 100 epochs, cosine LR (`cos_lr=True`), warmup 3 epochs |
| **Augmentation** | mosaic=1.0, HSV, translate, scale, fliplr=0.5, mixup=0 |
| **Early stop** | patience=30 |
| **Eval (not train)** | `conf=0.001`, `iou=0.6` (NMS) |

### Batch sizes (important difference)

| Model | Batch | Why |
|-------|-------|-----|
| YOLO11n/s | 32 | Fits on T4 |
| YOLO12s | 0.6 (auto) | Ultralytics picks ~60% VRAM |
| RT-DETRv1 | **4** (hardcoded) | OOM at higher batch on T4 |

### Default Ultralytics detection losses (not changed)

- `box_loss` — CIoU regression (weight 7.5)
- `cls_loss` — classification (0.5)
- `dfl_loss` — distribution focal loss (1.5)

RT-DETR instead logs: `giou_loss`, `cls_loss`, `l1_loss`.

---

## 4. Notebook-by-notebook walkthrough

### `baselines/yolo11n.ipynb`
```python
MODEL_ID = "yolo11n_640"
model = YOLO("yolo11n.pt")
model.train(data=YAML_PATH, project="NPS-Drone-Baseline-Benchmark", name=f"{MODEL_ID}_baseline", **CONFIG)
```
- Smallest / fastest baseline
- **Results:** AP50 0.673, F1 0.663, **261.6 FPS**, FPPI 5.23

### `baselines/yolo11s.ipynb`
```python
MODEL_ID = "yolo11s_640"
model = YOLO("yolo11s.pt")
```
- Medium YOLO11
- **Results:** AP50 0.729, F1 0.694, 154.2 FPS, FPPI 4.15

### `baselines/Baseline yolo v12s.ipynb`
```python
MODEL_ID = "yolo12s_640"
model = YOLO("yolo12s.pt")
```
- **Best spatial trade-off in our project**
- **Results:** AP50 **0.854**, F1 **0.862**, **117.7 FPS**, FPPI 3.27

### `baselines/baseline deter.ipynb`
```python
MODEL_ID = "rtdetr_l_640"
model = RTDETR("rtdetr-l.pt")
# train with batch=4 override
```
- Transformer detector (DETR family)
- Highest AP50 but slow + many false positives
- **Results:** AP50 **0.913**, F1 0.893, **22.3 FPS**, FPPI **78.57**

### `baselines/yolo11s-p2-nwm.ipynb` (experimental — different protocol)

**Not aligned with main baseline table.** Know it exists but explain separately:

- Custom YAML: extra **P2/4 head** (stride 4) for tiny objects
- **NWD loss** instead of CIoU for box regression
- **SGD** optimizer (`lr0=0.01`), not AdamW
- Different W&B project; **no every-4th-frame eval**

---

## 5. Training flow (what happens when you run a cell)

```text
1. wandb.login() + wandb.init(project, config=CONFIG)
2. model.train(...)
      ├── loads COCO weights
      ├── augments batches (mosaic, HSV, ...)
      ├── forward → box/cls/dfl losses
      ├── val every epoch on full val set
      └── saves weights/ + early stopping
3. W&B callbacks log train losses + val metrics each epoch
4. Checkpoint artifacts uploaded every 3 epochs
```

**W&B project:** `NPS-Drone-Baseline-Benchmark`  
**Tags:** `spatial-only-640`, model name

---

## 6. Evaluation flow (test set)

### Metrics extraction
```python
ap50    = val_results.box.ap50.mean()
ap50_95 = val_results.box.map
prec    = val_results.box.mp
rec     = val_results.box.mr
f1      = 2 * prec * rec / (prec + rec)
fps     = 1000.0 / sum(val_results.speed.values())
```

### Run validation on subset
```python
eval_model.val(
    data=SUBSET_YAML,
    split="test",
    imgsz=640,
    conf=0.001,    # very low — standard for AP computation
    iou=0.6,       # NMS threshold
    save_json=True,
    save_txt=True, # needed for FPPI
)
```

### FPPI (False Positives Per Image)

**Definition:** On frames with **no ground-truth drone**, count predicted boxes; divide by total evaluated frames.

```python
# Correct version (yolo12s, RT-DETR):
compute_fppi_fixed(pred_label_dir, evaluated_stems, test_lbl_dir="/kaggle/working/nps/labels/test")
# → 809 empty frames out of 3,089 on every-4th subset
# FPPI = total_fp / 3089
```

**Why RT-DETRv1 FPPI is 78.6:** It fires many boxes on empty sky frames despite high precision on detections.

**Presentation tip:** “AP50 alone hid RT-DETR’s background false alarm problem — FPPI exposes it.”

---

## 7. Results table (what to memorize)

| Model | AP50 | F1 | FPS | FPPI | Role |
|-------|------|-----|-----|------|------|
| YOLO11n | 0.673 | 0.663 | 261.6 | 5.23 | Speed reference |
| YOLO11s | 0.729 | 0.694 | 154.2 | 4.15 | Mid YOLO |
| **YOLO12s** | **0.854** | **0.862** | **117.7** | 3.27 | **Best baseline** |
| RT-DETRv1 | 0.913 | 0.893 | 22.3 | 78.57 | High AP, bad FPPI |

### vs published methods (different hardware/protocol)

| Method | AP50 | FPS |
|--------|------|-----|
| TransVisDrone (1280²) | 0.95 | 24.6 |
| YOLOMG-640 | 0.92 | 133 |
| Dogfight | 0.89 | ~1 |

---

## 8. How to present this section (2-minute script)

> “We first established reproducible spatial baselines on NPS-Drones. We symlinked the Kaggle dataset into Ultralytics format, trained YOLO11n/s and YOLO12s plus RT-DETRv1 with identical AdamW hyperparameters at 640 resolution, and evaluated on every 4th test frame to match TransVisDrone.
>
> YOLO12s gave the best practical trade-off: 0.854 AP at 118 FPS. RT-DETRv1 reached 0.913 AP but only 22 FPS and 78 false positives per image on background frames. That motivated our custom model: we wanted temporal motion cues like YOLOMG, but with a lighter sparse design.”

---

## 9. File map (quick reference)

```text
baselines/
├── yolo11n.ipynb              → YOLO11n, batch 32
├── yolo11s.ipynb              → YOLO11s, batch 32
├── Baseline yolo v12s.ipynb   → YOLO12s, auto batch
├── baseline deter.ipynb       → RT-DETR-l, batch 4
└── yolo11s-p2-nwm.ipynb       → experimental P2+NWD (separate protocol)

Runtime paths:
  /kaggle/working/nps/              dataset
  /kaggle/working/nps_drone.yaml   train config
  /kaggle/working/tmp_test_every4th/   eval subset
  /kaggle/working/baseline_results.csv   aggregated results
```

---

## 10. Common questions & answers

**Q: Why every 4th frame?**  
A: TransVisDrone and Dogfight use skip-rate 4 on NPS test. We match them for fair comparison.

**Q: Why conf=0.001 at test?**  
A: Standard for COCO-style AP — sweep all confidence levels; AP integrates over them.

**Q: Why not YOLO11m notebook?**  
A: README lists YOLO11m numbers (AP50 0.718) but no notebook is in the repo — trained with same pipeline as YOLO11s.

**Q: What is the preprocessing contribution?**  
A: Symlinked Ultralytics layout, unified YAML, every-4th-frame test export, FPPI script, shared CONFIG — all baselines run identically except architecture.

---

## Next document

For the **custom heatmap detector** (EfficientNet, BiFPN, motion maps, phases 1–3), read:

**`docs/README_CUSTOM_EXPERIMENTS.md`**
