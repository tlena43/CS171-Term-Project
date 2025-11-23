import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision.transforms import functional as F

# Torchvision fast image read (optional)
try:
    from torchvision.io import read_image
    _HAS_TV_READ = True
except Exception:
    _HAS_TV_READ = False

# ---------- Dataset ----------
class RCNN_Warp_Data(Dataset):
    """
    Fast dataset for YOLO-format labels -> Faster R-CNN targets.
    Caches normalized boxes; converts to xyxy per image size at __getitem__.
    Expects: <root>/<split>/images, <root>/<split>/labels, and classes.txt at <root>/.
    """
    def __init__(self, root_path, split, transforms=None, class_path="classes.txt", use_tv_read=True):
        self.root_path = str(root_path)
        self.split_path = split
        self.class_path = class_path
        self.transforms = transforms
        self.use_tv_read = (use_tv_read and _HAS_TV_READ)

        self.class_names = self.__get_classes()
        self.image_paths, self.label_paths = self.__get_paths()

        # Cache normalized targets (cx,cy,w,h) and class ids
        self._norm_targets = []
        for lp in self.label_paths:
            try:
                if not os.path.exists(lp) or os.path.getsize(lp) == 0:
                    self._norm_targets.append((None, None))
                    continue
                arr = np.loadtxt(lp, ndmin=2, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] < 5:
                    self._norm_targets.append((None, None))
                    continue
                labels = arr[:, 0].astype(np.int64)
                cxcywh = arr[:, 1:5].astype(np.float32)
                self._norm_targets.append((labels, cxcywh))
            except Exception:
                self._norm_targets.append((None, None))

    def __get_classes(self):
        with open(os.path.join(self.root_path, self.class_path), "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    def __get_paths(self):
        img_dir = os.path.join(self.root_path, self.split_path, "images")
        lbl_dir = os.path.join(self.root_path, self.split_path, "labels")
        imgs, lbls = [], []
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            imgs.append(os.path.join(img_dir, fn))
            lbls.append(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"))
        return imgs, lbls

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        if self.use_tv_read:
            # CHW uint8 -> float [0,1]
            img = read_image(img_path).float() / 255.0  # Tensor [C,H,W]
        else:
            img = Image.open(img_path).convert("RGB")    # PIL

        if isinstance(img, torch.Tensor):
            H, W = int(img.shape[1]), int(img.shape[2])
        else:
            W, H = img.size

        labels_np, cxcywh_np = self._norm_targets[index]
        if labels_np is None or cxcywh_np is None or len(labels_np) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
        else:
            cx = cxcywh_np[:, 0] * W
            cy = cxcywh_np[:, 1] * H
            ww = cxcywh_np[:, 2] * W
            hh = cxcywh_np[:, 3] * H
            x1 = np.clip(cx - ww / 2.0, 0, W - 1)
            y1 = np.clip(cy - hh / 2.0, 0, H - 1)
            x2 = np.clip(cx + ww / 2.0, x1 + 1e-3, W - 1)
            y2 = np.clip(cy + hh / 2.0, y1 + 1e-3, H - 1)

            boxes_t  = torch.from_numpy(np.stack([x1, y1, x2, y2], axis=1)).to(torch.float32)
            labels_t = torch.from_numpy(labels_np).to(torch.int64)

        area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        target = {
            "boxes":   boxes_t,
            "labels":  labels_t,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area":    area_t,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }

        # Apply transforms if provided
        if self.transforms is not None:
            try:
                img, target = self.transforms(img, target)
            except TypeError:
                img = self.transforms(img)

        # Ensure tensor image in float32 for detection models
        if isinstance(img, Image.Image):
            img = F.to_tensor(img)  # float32 [0,1]
        elif isinstance(img, torch.Tensor) and img.dtype != torch.float32:
            img = img.float()

        target["labels"] = target["labels"].to(torch.int64).reshape(-1)
        target["boxes"]  = target["boxes"].to(torch.float32).reshape(-1, 4)
        return img, target


# ---------- Collate ----------
def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)


# ---------- Custom ResNet + FPN backbone ----------
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

class ResNetFPNBackbone(nn.Module):
    def __init__(
        self,
        resnet='resnet50',
        weights='DEFAULT',            # 'DEFAULT' or None
        trainable_layers=3,           # typically train layer3+layer4
        returned_layers=(2, 3, 4),    # ResNet layers -> C3,C4,C5
        in_channels_list=(512, 1024, 2048),
        out_channels=256,
        norm_layer=nn.BatchNorm2d
    ):
        super().__init__()

        # Version-safe weights handling
        def _get_resnet(resnet_name, weights_flag, norm_layer):
            ctor = getattr(torchvision.models, resnet_name)
            # Try new API first
            try:
                weights_arg = None
                if weights_flag == 'DEFAULT':
                    enum = getattr(torchvision.models, f"{resnet_name}_Weights", None)
                    if enum is not None:
                        weights_arg = enum.DEFAULT
                return ctor(weights=weights_arg, norm_layer=norm_layer)
            except TypeError:
                # Old API: pretrained=bool
                return ctor(pretrained=(weights_flag == 'DEFAULT'), norm_layer=norm_layer)

        base = _get_resnet(resnet, weights, norm_layer)

        # Freeze early stages if desired (trainable_layers counts from deepest)
        layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']
        for i, name in enumerate(layers_to_train):
            if i >= trainable_layers:
                module = getattr(base, name, None)
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad = False

        # Map intermediate layers
        # Note: returned_layers=(2,3,4) corresponds to ResNet layers -> C3,C4,C5 logically.
        return_layers = {f"layer{l}": f"C{l+1}" for l in returned_layers}  # names C3,C4,C5
        self.body = IntermediateLayerGetter(base, return_layers=return_layers)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(in_channels_list),
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),  # adds P6
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c_feats = self.body(x)        # OrderedDict: C3,C4,C5
        p_feats = self.fpn(c_feats)   # dict of pyramid features
        return p_feats


# ---------- Training loop ----------
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datetime import datetime
import contextlib

def training_loop(
    model,
    optimizer,
    NUM_EPOCHS,
    train_loader,
    test_loader,
    printing=True,
    device=None,
    use_amp=True,
    grad_clip_norm=None,
    eval_every=5,
    file=None,
):
    if device is None:
        device = next(model.parameters()).device

    # AMP (version-agnostic import)
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
        def amp_ctx():
            return autocast(device_type="cuda")
    except Exception:
        # Fallback for older PyTorch
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
        def amp_ctx():
            return autocast()

    test_metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False).to(device)

    train_losses, test_maps = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for imgs, targets in train_loader:
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            if use_amp and device.type == "cuda":
                with amp_ctx():
                    loss_dict = model(imgs, targets)
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_train_loss += float(loss)

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        train_losses.append(avg_train_loss)

        # ---- Eval (throttled) ----
        model.eval()
        if (epoch + 1) % eval_every == 0 or epoch == NUM_EPOCHS - 1:
            test_metric.reset()
            # Either run eval in AMP on CUDA, or force float32 inputs to avoid fp16/fp32 mismatch
            ctx = amp_ctx() if (use_amp and device.type == "cuda") else contextlib.nullcontext()
            with torch.inference_mode(), ctx:
                for imgs, targets in test_loader:
                    imgs = [img.to(device, non_blocking=True) for img in imgs] \
                           if (use_amp and device.type == "cuda") \
                           else [img.to(device, dtype=torch.float32, non_blocking=True) for img in imgs]
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                    preds = model(imgs)
                    test_metric.update(preds, targets)
            te_map = test_metric.compute()["map"]
        else:
            te_map = torch.tensor(float("nan"), device=device)

        test_maps.append(te_map)

        if printing:
            te_map_val = float(te_map.detach().item()) if torch.isfinite(te_map).item() else float("nan")
            line = (
                f"Epoch {epoch+1}/{NUM_EPOCHS} "
                f"- Train Loss: {avg_train_loss:.4f} "
                f"- Test mAP: {te_map_val:.4f} "
                f"- Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            print(line)
            if file:
                print(line, file=file, flush=True)

    return train_losses, test_maps


# ---------- Main ----------
if __name__ == "__main__":
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    root = Path.cwd() / "Warp-D"
    train_ds = RCNN_Warp_Data(root, split="train")
    test_ds  = RCNN_Warp_Data(root, split="test")

    # Optional subsampling for quick experiments
    sub_frac = 0.5
    g = torch.Generator().manual_seed(42)
    train_len = int(sub_frac * len(train_ds))
    test_len  = int(sub_frac * len(test_ds))
    train_ds, _ = random_split(train_ds, [train_len, len(train_ds) - train_len], generator=g)
    test_ds, _  = random_split(test_ds,  [test_len,  len(test_ds)  - test_len],  generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=8,              # 6–10 usually ideal on a fast SSD
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,    # enable now
        prefetch_factor=4,
        timeout=120,
        multiprocessing_context="spawn",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=True,
        num_workers=8,              # 6–10 usually ideal on a fast SSD
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,    # enable now
        prefetch_factor=4,
        timeout=120,
        multiprocessing_context="spawn",
    )

    # Build custom ResNet-FPN backbone and model
    backbone = ResNetFPNBackbone(
        resnet='resnet50',
        weights='DEFAULT',
        trainable_layers=3,              # typically train layer3+layer4
        returned_layers=(2,3,4),         # taps layer2/3/4 (→ C3,C4,C5 conceptually)
        in_channels_list=(512,1024,2048),
        out_channels=256
    )
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    from torchvision.ops import MultiScaleRoIAlign

    from torchvision.models.detection import FasterRCNN
    model = FasterRCNN(backbone, num_classes=29, rpn_anchor_generator=anchor_generator).to(device)
    
    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
    featmap_names=['C3', 'C4', 'C5', 'pool'],  # <— MUST match your FPN keys
    output_size=7,
    sampling_ratio=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses, test_maps = training_loop(
        model, optimizer, NUM_EPOCHS=50,
        train_loader=train_loader, test_loader=test_loader,
        printing=True, device=device, use_amp=True, eval_every=5
    )
