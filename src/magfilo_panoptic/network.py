"""Multi-head 3-output Panoptic U-Net and soft-clDice topology loss (Stage N)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import disk_crop_box

try:
    import timm

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import torchvision.models as tv_models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class SCSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation (Roy et al., MICCAI 2018)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.c_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.c_se(x) + x * self.s_se(x)


class PPMContext(nn.Module):
    """Lightweight Multi-Scale Atrous + Global Context Bottleneck."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        b_ch = max(out_ch // 4, 16)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, b_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, b_ch, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(b_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        p = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([self.b0(x), self.b1(x), self.b2(x), p], dim=1)
        return self.proj(out)


class DecoderBlock(nn.Module):
    """U-Net skip decoder stage with 1x1 compression, depthwise-separable 3x3 refinement, and scSE."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dw_sep: bool = False) -> None:
        super().__init__()
        if dw_sep:
            refine: List[nn.Module] = [
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 1, bias=False),
            ]
        else:
            refine = [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)]
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            *refine,
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.scse = SCSEBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.scse(self.conv(x))


class TorchvisionResNet18Features(nn.Module):
    """Pure torchvision ResNet18 multi-scale feature extractor fallback."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = None
        if pretrained and HAS_TORCHVISION:
            try:
                weights = tv_models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        net = tv_models.resnet18(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.channels = [64, 64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c1, c2, c3, c4, c5]


class MagfiloPanopticUNet(nn.Module):
    """
    Lightweight 3-head (fg, spine, sep) Panoptic U-Net built from PyTorch + timm
    (with torchvision fallback). Uses a compact stride-1 detail stem to retain
    1-3 px filament barbs and 1-8 px inter-filament separation gaps.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        bottleneck_ch: int = 160,
        dec_widths: Sequence[int] = (96, 64, 40, 24),
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.encoder, enc_channels = self._build_encoder(backbone_name, pretrained)

        self.full_stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
        )

        self.ppm = PPMContext(enc_channels[-1], bottleneck_ch)

        n_skips = len(enc_channels) - 1
        widths = list(dec_widths[:n_skips])
        while len(widths) < n_skips:
            widths.append(max(widths[-1] // 2, 24))

        self.dec_blocks = nn.ModuleList()
        prev_ch = bottleneck_ch
        for idx in range(n_skips):
            skip_ch = enc_channels[-2 - idx]
            out_ch = widths[idx]
            self.dec_blocks.append(DecoderBlock(prev_ch, skip_ch, out_ch, dw_sep=(idx >= 2)))
            prev_ch = out_ch

        head_in = prev_ch + 8
        self.head = nn.Sequential(
            nn.Conv2d(head_in, head_in, 3, padding=1, groups=head_in, bias=False),
            nn.BatchNorm2d(head_in),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_in, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, 1),
        )

    @staticmethod
    def _build_encoder(backbone_name: str, pretrained: bool) -> Tuple[nn.Module, List[int]]:
        if HAS_TIMM:
            for try_pretrained in ([True, False] if pretrained else [False]):
                try:
                    enc = timm.create_model(
                        backbone_name,
                        pretrained=try_pretrained,
                        features_only=True,
                        in_chans=3,
                    )
                    return enc, list(enc.feature_info.channels())
                except Exception:
                    pass
        if HAS_TORCHVISION:
            for try_pretrained in ([True, False] if pretrained else [False]):
                try:
                    enc = TorchvisionResNet18Features(pretrained=try_pretrained)
                    return enc, enc.channels
                except Exception:
                    pass
        raise RuntimeError("Neither timm nor torchvision encoder could be initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        s_full = self.full_stem(x)
        feats = self.encoder(x)
        d = self.ppm(feats[-1])
        for idx, blk in enumerate(self.dec_blocks):
            d = blk(d, feats[-2 - idx])
        d = F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
        out = torch.cat([d, s_full], dim=1)
        return self.head(out)


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 5) -> torch.Tensor:
    """Differentiable soft skeletonization (Shit et al., CVPR 2021)."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(
    fg_prob: torch.Tensor, y_fg: torch.Tensor, y_sp: torch.Tensor, valid: torch.Tensor, cl_iters: int = 5
) -> torch.Tensor:
    """Compute soft-clDice topology preservation loss with explicit spine target."""
    fg_m = fg_prob * valid
    y_fg_m = y_fg * valid
    sp_m = y_sp * valid
    skel_pred = soft_skel(fg_m, iter_=cl_iters)
    skel_gt = torch.maximum(sp_m, soft_skel(y_fg_m, iter_=cl_iters))
    tprec = (skel_pred * y_fg_m).sum() / skel_pred.sum().clamp_min(1e-4)
    tsens = (skel_gt * fg_m).sum() / skel_gt.sum().clamp_min(1e-4)
    return 1.0 - (2.0 * tprec * tsens) / (tprec + tsens + 1e-4)


def masked_soft_dice_loss(
    prob: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    p = prob.float() * valid_mask
    t = target.float() * valid_mask
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + 1.0) / (denom + 1.0 + eps)
    return (1.0 - dice).mean()


class SolarMultiTaskLoss(nn.Module):
    """
    Boundary-upweighted BCE + SoftDice (3x weight near y_sep) + spine/sep losses
    + 0.25 * soft_clDice topology preservation (Shit et al., CVPR 2021).
    """

    def __init__(self, cl_iters: int = 5) -> None:
        super().__init__()
        self.cl_iters = cl_iters

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        fg_logit = logits[:, 0:1].float()
        sp_logit = logits[:, 1:2].float()
        sep_logit = logits[:, 2:3].float()

        y_fg = targets[:, 0:1].float()
        y_sp = targets[:, 1:2].float()
        y_sep = targets[:, 2:3].float()
        valid = valid_mask.float()

        fg_prob = torch.sigmoid(fg_logit)
        sp_prob = torch.sigmoid(sp_logit)
        sep_prob = torch.sigmoid(sep_logit)

        sep_zone = F.max_pool2d(y_sep, kernel_size=5, stride=1, padding=2)
        w_fg = valid * (1.0 + 2.0 * sep_zone)
        bce_fg = (
            F.binary_cross_entropy_with_logits(fg_logit, y_fg, reduction="none") * w_fg
        ).sum() / w_fg.sum().clamp_min(1.0)
        dice_fg = masked_soft_dice_loss(fg_prob, y_fg, valid)

        bce_sp = (
            F.binary_cross_entropy_with_logits(sp_logit, y_sp, reduction="none") * valid
        ).sum() / valid.sum().clamp_min(1.0)
        dice_sp = masked_soft_dice_loss(sp_prob, y_sp, valid)

        sep_pos_w = torch.tensor([4.0], device=logits.device, dtype=torch.float32)
        bce_sep = (
            F.binary_cross_entropy_with_logits(sep_logit, y_sep, pos_weight=sep_pos_w, reduction="none")
            * valid
        ).sum() / valid.sum().clamp_min(1.0)
        dice_sep = masked_soft_dice_loss(sep_prob, y_sep, valid)

        cl_fg = soft_cldice_loss(fg_prob, y_fg, y_sp, valid, cl_iters=self.cl_iters)
        sp_m = sp_prob * valid
        fg_m = fg_prob * valid
        skel_pred = soft_skel(fg_m, iter_=self.cl_iters)
        tprec_sp = (sp_m * fg_m).sum() / sp_m.sum().clamp_min(1e-4)
        tsens_sp = (skel_pred * sp_m).sum() / skel_pred.sum().clamp_min(1e-4)
        cl_fg_sp = 1.0 - (2.0 * tprec_sp * tsens_sp) / (tprec_sp + tsens_sp + 1e-4)

        return (
            (bce_fg + dice_fg)
            + 0.60 * (bce_sp + dice_sp)
            + 0.50 * (0.5 * bce_sep + 0.5 * dice_sep)
            + 0.25 * (0.75 * cl_fg + 0.25 * cl_fg_sp)
        )


def infer_disk_gpu(
    models: List[nn.Module],
    prep: Dict[str, Any],
    device: torch.device,
    img_size: int = 1536,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer 3-head probabilities on GPU using tight disk crop + 4-view flip TTA."""
    x_3ch = prep["x_3ch"]
    dmask = prep["disk_mask"]
    h, w = dmask.shape
    cx, cy, R = prep["disk_params"]
    x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

    crop = x_3ch[:, y0:y1, x0:x1].transpose(1, 2, 0)
    crop_res = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    xt = torch.from_numpy(crop_res).unsqueeze(0).to(device)

    prob_acc = np.zeros((3, img_size, img_size), dtype=np.float32)
    n_views = 4 if use_tta else 1

    with torch.inference_mode():
        for model in models:
            model.eval()
            for v in range(n_views):
                xv = xt
                if v == 1:
                    xv = torch.flip(xt, dims=[3])
                elif v == 2:
                    xv = torch.flip(xt, dims=[2])
                elif v == 3:
                    xv = torch.flip(xt, dims=[2, 3])

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pv = torch.sigmoid(model(xv).float())

                if v == 1:
                    pv = torch.flip(pv, dims=[3])
                elif v == 2:
                    pv = torch.flip(pv, dims=[2])
                elif v == 3:
                    pv = torch.flip(pv, dims=[2, 3])
                prob_acc += pv.squeeze(0).cpu().numpy()

    prob_acc /= float(len(models) * n_views)
    ch, cw = y1 - y0, x1 - x0
    out_full = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        resized = cv2.resize(prob_acc[c], (cw, ch), interpolation=cv2.INTER_LINEAR)
        out_full[c, y0:y1, x0:x1] = resized * dmask[y0:y1, x0:x1]
    return out_full[0], out_full[1], out_full[2]


def infer_disk_cpu_tight(
    models: List[nn.Module],
    prep: Dict[str, Any],
    device: torch.device,
    infer_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-pass CPU inference on tight 1024x1024 solar-disk crop using
    torch.inference_mode() and channels_last memory layout, upsampled bilinearly
    back onto native 2048x2048 coordinates inside the off-limb mask.
    """
    x_3ch = prep["x_3ch"]
    dmask = prep["disk_mask"]
    h, w = dmask.shape
    cx, cy, R = prep["disk_params"]
    x0, y0, x1, y1 = disk_crop_box(cx, cy, R, h, w, pad_ratio=1.015)

    crop = x_3ch[:, y0:y1, x0:x1].transpose(1, 2, 0)
    crop_res = cv2.resize(crop, (infer_size, infer_size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    xt = torch.from_numpy(np.ascontiguousarray(crop_res)).unsqueeze(0).to(device)
    xt = xt.to(memory_format=torch.channels_last)

    prob_acc = np.zeros((3, infer_size, infer_size), dtype=np.float32)
    with torch.inference_mode():
        for model in models:
            model.eval()
            pv = torch.sigmoid(model(xt).float())
            prob_acc += pv.squeeze(0).cpu().numpy()

    prob_acc /= float(max(len(models), 1))
    ch, cw = y1 - y0, x1 - x0
    out_full = np.zeros((3, h, w), dtype=np.float32)
    for c in range(3):
        resized = cv2.resize(prob_acc[c], (cw, ch), interpolation=cv2.INTER_LINEAR)
        out_full[c, y0:y1, x0:x1] = resized * dmask[y0:y1, x0:x1]
    return out_full[0], out_full[1], out_full[2]
