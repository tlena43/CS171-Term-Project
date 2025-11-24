import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import functional as F

# Torchvision fast image read (optional)
try:
    from torchvision.io import read_image
    _HAS_TV_READ = True
except Exception:
    _HAS_TV_READ = False

# ---------- Dataset ----------
class RCNN_Data(Dataset):
    """
    Detection dataset for the 'Drinking Waste Classification' Kaggle set.
    Each sample has a JPG image and a matching .txt with YOLO-format boxes.
    Expects structure like:
        AluCan1_000.jpg
        AluCan1_000.txt
        ...
    """
    def __init__(self, root_dir, class_names=["Aluminum", "Glass", "HDP", "PET"], transforms=None, use_tv_read=True):
        self.root_dir = str(root_dir)
        self.transforms = transforms
        self.class_names = class_names
        self.use_tv_read = use_tv_read
        # gather all jpgs and their paired txts
        self.samples = []
        for fn in sorted(os.listdir(self.root_dir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                stem = os.path.splitext(fn)[0]
                txt = os.path.join(self.root_dir, stem + ".txt")
                self.samples.append((os.path.join(self.root_dir, fn), txt))

        # optional check
        print(f"Found {len(self.samples)} image/label pairs in {self.root_dir}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        # --- load image
        if self.use_tv_read:
            img = read_image(img_path).float() / 255.0  # tensor [C,H,W]
        else:
            img = Image.open(img_path).convert("RGB")
            img = F.to_tensor(img)
        _, H, W = img.shape

        # --- read YOLO txt
        boxes_t = torch.zeros((0,4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            arr = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 5:
                labels_np = arr[:, 0].astype(np.int64) + 1      # +1 to reserve 0 as background
                cxcywh_np = arr[:, 1:5].astype(np.float32)
                cx = cxcywh_np[:, 0] * W
                cy = cxcywh_np[:, 1] * H
                ww = cxcywh_np[:, 2] * W
                hh = cxcywh_np[:, 3] * H
                x1 = np.clip(cx - ww / 2, 0, W-1)
                y1 = np.clip(cy - hh / 2, 0, H-1)
                x2 = np.clip(cx + ww / 2, x1 + 1e-3, W-1)
                y2 = np.clip(cy + hh / 2, y1 + 1e-3, H-1)
                boxes_t = torch.from_numpy(np.stack([x1,y1,x2,y2],1)).float()
                labels_t = torch.from_numpy(labels_np).long()

        # --- build target
        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:,2]-boxes_t[:,0])*(boxes_t[:,3]-boxes_t[:,1]),
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
        }

        # --- apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


# ---------- Collate ----------
def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

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
    grad_clip_norm=5,
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

##basic starter to prove it works, will improve later
class Custom_Backbone(nn.Module):
    def __init__ (self, in_channels = 3, out_channels = 256, hidden_size = 128, depth = 3):
        super().__init__()

        layers = []
        in_dim = in_channels
        out_dim = hidden_size

        for i in range(depth):
            layers += [
                nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),  
                nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            ]
            in_dim = out_dim
            out_dim = min(out_dim * 2, 1024)

        layers += [
            nn.Conv2d(in_dim, out_channels, 3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),    
        ]
            
        self.body = nn.Sequential(*layers)
        # required by FasterRCNN
        self.out_channels = out_channels

    def forward(self, x):
        # return a single Tensor feature map
        return self.body(x)


# ---------- Main ----------
if __name__ == "__main__":
    print(f"- Time: {datetime.now().strftime('%H:%M:%S')}")
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    

    
    import torch
    from torchvision.tv_tensors import Image
    
    root = Path.cwd() / "Images_of_Waste" / "YOLO_imgs"
    ds = RCNN_Data(root)
    split = 0.8
    train_size = int(len(ds)  * split)
    test_size = len(ds) - train_size
    train_ds,test_ds= random_split(ds, [train_size, test_size])


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
    
    from torchvision.models.detection.rpn import AnchorGenerator

    anchor_generator = AnchorGenerator(
    sizes=((48, 64), (96, 128), (192, 256), (384, 512)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 4
    )
    
    backbone = Custom_Backbone()
    
    anchor_sizes =((32, 64, 128, 256, 512),) 
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
    sizes = anchor_sizes ,    # P6
    aspect_ratios= aspect_ratios
    )
    
    from torchvision.ops import MultiScaleRoIAlign

    from torchvision.models.detection import FasterRCNN
    model = FasterRCNN(backbone, num_classes=5, rpn_anchor_generator=anchor_generator).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    train_losses, test_maps = training_loop(
        model, optimizer, NUM_EPOCHS=102,
        train_loader=train_loader, test_loader=test_loader,
        printing=True, device=device, use_amp=False, eval_every=3
    )
