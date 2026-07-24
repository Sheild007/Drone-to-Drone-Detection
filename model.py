import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tv_ops
import timm


# ---------------------------------------------------------------------------
# Default config — mirror of the Phase 3 CFG dict
# ---------------------------------------------------------------------------
CFG_DEFAULT = {
    "seed": 42,
    "img_size": 640,
    "temporal_len": 3,
    "heatmap_stride": 4,
    "heatmap_sigma": 2.0,          # fallback; actual sigma is adaptive per-object
    "use_adaptive_sigma": True,
    "topk": 50,
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "backbone_out_channels": [40, 112, 320],
    "bifpn_channels": 64,
    "bifpn_layers": 3,
    "use_motion": True,
    "motion_channels": 2,
    "roi_output_size": 11,
    "roi_levels": [0, 1, 2],
    "anchor_size": None,           # set from training data via Cell 5
    "refine_hidden": 256,
    "num_classes": 1,
    "amp": True,
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, kernel, padding=padding, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class TemporalFusion(nn.Module):
    """Fuse T backbone feature maps with a 1x1 conv over their concatenation."""
    def __init__(self, ch, T):
        super().__init__()
        self.T    = T
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * T, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        BT, C, H, W = x.shape
        B = BT // self.T
        return self.fuse(x.view(B, self.T * C, H, W))


class BiFPNNode(nn.Module):
    def __init__(self, ch, n_inputs=2):
        super().__init__()
        self.w   = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = 1e-4
        self.conv = DepthwiseSeparableConv(ch, ch)

    def forward(self, feats):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        fused = sum(w[i] * feats[i] for i in range(len(feats)))
        return self.conv(fused)


class BiFPNLayer(nn.Module):
    def __init__(self, ch, num_levels=3):
        super().__init__()
        self.td_nodes = nn.ModuleList(
            [BiFPNNode(ch, n_inputs=2) for _ in range(num_levels - 1)])
        self.bu_nodes = nn.ModuleList(
            [BiFPNNode(ch, n_inputs=3 if i < num_levels - 2 else 2)
             for i in range(num_levels - 1)])

    def forward(self, features):
        n = len(features)
        td = [None] * n
        td[-1] = features[-1]
        for i in range(n - 2, -1, -1):
            up    = F.interpolate(td[i + 1], size=features[i].shape[-2:], mode="nearest")
            td[i] = self.td_nodes[i]([features[i], up])
        out = [None] * n
        out[0] = td[0]
        for i in range(1, n):
            dn = F.adaptive_avg_pool2d(out[i - 1], features[i].shape[-2:])
            if i < n - 1:
                out[i] = self.bu_nodes[i - 1]([features[i], td[i], dn])
            else:
                out[i] = self.bu_nodes[i - 1]([features[i], dn])
        return out


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_ch, num_layers=3):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True))
            for c in in_channels_list
        ])
        self.layers = nn.ModuleList([
            BiFPNLayer(out_ch, num_levels=3) for _ in range(num_layers)
        ])
        # Extra upsample creates P1 at stride 4 — critical for 6 px drones
        self.p1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, c_feats):
        feats = [lat(c) for lat, c in zip(self.lateral, c_feats)]
        for layer in self.layers:
            feats = layer(feats)
        p1 = self.p1_upsample(feats[0])
        return [p1] + feats


class MotionEncoder(nn.Module):
    """
    Downsample-first motion encoder.

    Raw pixel-difference maps (YOLOMG-style) are downsampled to P1 resolution
    before convolutions — far cheaper than full-resolution convs with negligible
    quality cost for the tiny-target motion signal.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, motion, target_size):
        motion_ds = F.adaptive_avg_pool2d(motion, target_size)
        return self.enc(motion_ds)


class HeatmapHead(nn.Module):
    def __init__(self, in_ch, num_classes=1, use_motion=True):
        super().__init__()
        self.use_motion = use_motion
        if use_motion:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True),
            )
        self.head = nn.Sequential(
            DepthwiseSeparableConv(in_ch, in_ch),
            DepthwiseSeparableConv(in_ch, in_ch),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, p1, motion_feat=None):
        if self.use_motion and motion_feat is not None:
            x = torch.cat([p1, motion_feat], dim=1)
            x = self.fuse_conv(x)
        else:
            x = p1
        return self.head(x)


def extract_peaks(heatmap_logits, topk=50, kernel=3, conf_thr=0.01):
    """
    CenterNet-style peak extraction:
    sigmoid → 3×3 max-pool NMS → top-K=50 above confidence threshold.
    Runs under torch.no_grad() in the main forward pass; gradients flow
    through the dense heatmap loss instead.
    """
    B    = heatmap_logits.shape[0]
    heat = torch.sigmoid(heatmap_logits)
    pad  = kernel // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (heat == hmax).float() * heat

    peaks_list = []
    for b in range(B):
        flat = keep[b, 0].reshape(-1)
        k    = min(topk, int((flat > conf_thr).sum().item()))
        k    = max(k, 1)
        vals, inds = flat.topk(k)
        H, W  = keep.shape[-2:]
        ys    = inds // W
        xs    = inds %  W
        coords = torch.stack([ys, xs], dim=1).float()
        peaks_list.append((coords, vals))
    return peaks_list


class RefinementHead(nn.Module):
    """Single-anchor MLP refinement head."""
    def __init__(self, roi_size, num_fpn_levels, fpn_ch, hidden):
        super().__init__()
        flat_dim = num_fpn_levels * fpn_ch * roi_size * roi_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
        )
        self.obj_head = nn.Linear(hidden // 2, 1)
        self.box_head = nn.Linear(hidden // 2, 4)

    def forward(self, roi_feats):
        h   = self.mlp(roi_feats)
        obj = self.obj_head(h)
        box = self.box_head(h)
        return obj, box


# ---------------------------------------------------------------------------
# Full detector
# ---------------------------------------------------------------------------

class HeatmapGuidedDetector(nn.Module):
    """
    Heatmap-guided sparse detector for tiny airborne drones.

    Parameters
    ----------
    cfg : dict
        Configuration dict (see CFG_DEFAULT).
        cfg['anchor_size'] must be set (compute from training labels).
    """

    def __init__(self, cfg):
        super().__init__()
        T        = cfg["temporal_len"]
        fpn_ch   = cfg["bifpn_channels"]
        roi_size = cfg["roi_output_size"]

        self.backbone = timm.create_model(
            cfg["backbone"], pretrained=cfg["pretrained"],
            features_only=True, out_indices=(2, 3, 4))
        bb_ch = cfg["backbone_out_channels"]

        self.temporal_fuse = nn.ModuleList([TemporalFusion(c, T) for c in bb_ch])
        self.bifpn = BiFPN(bb_ch, fpn_ch, cfg["bifpn_layers"])

        self.use_motion = cfg["use_motion"]
        if self.use_motion:
            self.motion_encoder = MotionEncoder(cfg["motion_channels"], fpn_ch)

        self.hm_head = HeatmapHead(fpn_ch, cfg["num_classes"], use_motion=self.use_motion)

        self.roi_levels  = cfg["roi_levels"]
        self.refine_head = RefinementHead(roi_size, len(self.roi_levels), fpn_ch, cfg["refine_hidden"])

        self.roi_size  = roi_size
        self.fpn_ch    = fpn_ch
        self.hm_stride = 4
        self.topk      = cfg["topk"]
        self.img_size  = cfg["img_size"]

        assert cfg["anchor_size"] is not None, (
            "cfg['anchor_size'] is None — compute from training labels first.")
        self.anchor_size = cfg["anchor_size"]

    def _extract_backbone(self, frames):
        B, T, C, H, W = frames.shape
        x     = frames.view(B * T, C, H, W)
        feats = self.backbone(x)
        fused = [self.temporal_fuse[i](feats[i]) for i in range(3)]
        return fused

    def _roi_align_single(self, fpn_feats, peaks_batch, B):
        all_roi_feats = []
        all_anchors   = []
        batch_splits  = []

        for b in range(B):
            coords, scores = peaks_batch[b]
            N = coords.shape[0]
            batch_splits.append(N)

            cxs = coords[:, 1] * self.hm_stride
            cys = coords[:, 0] * self.hm_stride

            level_roi_list = []
            for lvl_idx in self.roi_levels:
                fmap     = fpn_feats[lvl_idx]
                h_f, w_f = fmap.shape[-2:]
                scale_x  = w_f / self.img_size
                scale_y  = h_f / self.img_size

                fc_x = cxs * scale_x
                fc_y = cys * scale_y
                half = self.roi_size / 2.0

                x1_f = fc_x - half; y1_f = fc_y - half
                x2_f = fc_x + half; y2_f = fc_y + half

                batch_col = torch.full((N, 1), float(b), device=fmap.device)
                rois_in   = torch.stack([batch_col[:, 0], x1_f, y1_f, x2_f, y2_f], dim=1)

                roi_out = tv_ops.roi_align(
                    fmap, rois_in, output_size=self.roi_size,
                    spatial_scale=1.0, sampling_ratio=2, aligned=True)
                level_roi_list.append(roi_out)

            cat_roi = torch.cat(level_roi_list, dim=1)
            all_roi_feats.append(cat_roi.view(N, -1))

            anchors_b = torch.stack([
                cxs, cys,
                torch.full_like(cxs, self.anchor_size),
                torch.full_like(cys, self.anchor_size),
            ], dim=1)
            all_anchors.append(anchors_b)

        all_roi_feats_cat = torch.cat(all_roi_feats, dim=0)
        return all_roi_feats_cat, all_anchors, batch_splits

    def forward(self, frames, motion=None):
        """
        Parameters
        ----------
        frames : Tensor  (B, T, 3, H, W)  — normalized RGB triplets
        motion : Tensor  (B, 2, H, W)     — |anchor - prev| difference maps

        Returns
        -------
        hm_logits   : (B, 1, 160, 160)
        obj_logits  : (sum_N, 1)
        box_deltas  : (sum_N, 4)
        anchors     : list of (N_b, 4) tensors per image
        batch_splits: list of N_b ints
        """
        B         = frames.shape[0]
        c_feats   = self._extract_backbone(frames)
        fpn_feats = self.bifpn(c_feats)

        motion_feat = None
        if self.use_motion and motion is not None:
            p1_size     = fpn_feats[0].shape[-2:]
            motion_feat = self.motion_encoder(motion, p1_size)

        hm_logits = self.hm_head(fpn_feats[0], motion_feat)

        with torch.no_grad():
            peaks_batch = extract_peaks(hm_logits.detach(), topk=self.topk, conf_thr=0.01)

        roi_feats, anchors, splits = self._roi_align_single(fpn_feats, peaks_batch, B)
        obj_logits, box_deltas     = self.refine_head(roi_feats)

        return hm_logits, obj_logits, box_deltas, anchors, splits


# ---------------------------------------------------------------------------
# CenterNet-style bias init helper (call after constructing the model)
# ---------------------------------------------------------------------------

def init_heatmap_bias(model, prior_prob=0.01):
    """
    Initialize the final heatmap conv bias so sigmoid ≈ prior_prob.
    Prevents the all-positive heatmap at the start of training that
    causes the focal loss to backpropagate huge gradients on every pixel.
    """
    bias_val = -math.log((1.0 - prior_prob) / prior_prob)
    for m in model.hm_head.head.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    nn.init.constant_(last_conv.bias, bias_val)
    nn.init.normal_(last_conv.weight, std=0.001)
    return bias_val


if __name__ == "__main__":
    # Quick shape-check (no weights loaded)
    import copy
    cfg = copy.deepcopy(CFG_DEFAULT)
    cfg["pretrained"]  = False
    cfg["anchor_size"] = 5.93       # typical value from NPS training data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = HeatmapGuidedDetector(cfg).to(device)
    init_heatmap_bias(model)

    B, T = 2, cfg["temporal_len"]
    H, W = cfg["img_size"], cfg["img_size"]
    frames = torch.randn(B, T, 3, H, W, device=device)
    motion = torch.randn(B, cfg["motion_channels"], H, W, device=device)

    model.eval()
    with torch.no_grad():
        hm, obj, box, anc, splits = model(frames, motion)

    print(f"hm_logits  : {hm.shape}")      # [2, 1, 160, 160]
    print(f"obj_logits : {obj.shape}")
    print(f"box_deltas : {box.shape}")
    assert hm.shape[-2:] == (160, 160), "Stride mismatch!"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {n_params / 1e6:.2f}M")
    print("Shape check PASSED.")
