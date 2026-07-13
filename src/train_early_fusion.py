import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
import yaml
import os

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models" / "early_fusion"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SETTINGS 
EPOCHS     = 20
BATCH_SIZE = 4
IMG_SIZE   = 640
DEVICE     = 0



# STEP 1 — PAIRED DATASET
# Loads RGB + thermal image pairs and concatenates them into 4-channel input

class PairedDataset(Dataset):
    """
    Loads matching RGB and thermal images and concatenates them.
    RGB (3 channels) + Thermal (1 channel) = 4 channel input tensor.
    Both images get identical augmentation to stay aligned.
    """
    def __init__(self, dataset, split, img_size=640):
        self.img_size = img_size
        self.split    = split

        # paths to rgb and thermal image folders
        self.rgb_dir  = DATA_DIR / dataset.upper() / "visible"  / "images" / split
        self.th_dir   = DATA_DIR / dataset.upper() / "infrared" / "images" / split
        self.lbl_dir  = DATA_DIR / dataset.upper() / "visible"  / "labels" / split

        # get sorted list of filenames — sorting ensures rgb[i] matches thermal[i]
        self.files = sorted([f.name for f in self.rgb_dir.iterdir()
                             if f.suffix in [".jpg", ".jpeg", ".png"]])

        print(f"  {dataset} {split}: {len(self.files)} paired images found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        stem  = Path(fname).stem

        # load RGB image
        rgb_path = self.rgb_dir / fname
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size))

        # load thermal image
        # try same filename first, then .jpeg extension
        th_path = self.th_dir / fname
        if not th_path.exists():
            th_path = self.th_dir / (stem + ".jpeg")
        th = cv2.imread(str(th_path))
        th = cv2.cvtColor(th, cv2.COLOR_BGR2GRAY)  # thermal → single channel
        th = cv2.resize(th, (self.img_size, self.img_size))

        # concatenate: RGB(3ch) + Thermal(1ch) = 4ch
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        th_t  = torch.from_numpy(th).unsqueeze(0).float() / 255.0        # [1,H,W]
        img_4ch = torch.cat([rgb_t, th_t], dim=0)                        # [4,H,W]

        # load label 
        lbl_path = self.lbl_dir / (stem + ".txt")
        labels   = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f.readlines():
                    vals = line.strip().split()
                    if len(vals) == 5:
                        labels.append([float(v) for v in vals])

        labels = torch.tensor(labels, dtype=torch.float32) \
                 if labels else torch.zeros((0, 5))

        return img_4ch, labels, str(rgb_path)

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths = zip(*batch)
        # add batch index to labels: [batch_idx, cls, cx, cy, w, h]
        batch_labels = []
        for i, lbl in enumerate(labels):
            if lbl.shape[0] > 0:
                idx_col = torch.full((lbl.shape[0], 1), i)
                batch_labels.append(torch.cat([idx_col, lbl], dim=1))
        batch_labels = torch.cat(batch_labels, dim=0) \
                       if batch_labels else torch.zeros((0, 6))
        return torch.stack(imgs), batch_labels, list(paths)


# STEP 2 — MODIFY YOLOV8N FIRST LAYER FOR 4-CHANNEL INPUT

def modify_yolov8_for_4ch(model):
    """
    YOLOv8n's first Conv layer expects 3 input channels (RGB).
    We change it to accept 4 channels (RGB + Thermal).

    Strategy: copy the existing 3-channel weights and add a 4th channel
    initialised to the mean of the existing 3 channels. This preserves
    the pretrained knowledge while adding thermal capacity.
    """
    first_conv = model.model.model[0]  # first Conv layer

    old_weight = first_conv.conv.weight.data  # shape: [16, 3, 3, 3]
    out_ch, in_ch, kH, kW = old_weight.shape  # 16, 3, 3, 3

    # create new weight with 4 input channels
    new_weight = torch.zeros(out_ch, 4, kH, kW)
    new_weight[:, :3, :, :] = old_weight          # copy RGB weights
    new_weight[:, 3:4, :, :] = old_weight.mean(dim=1, keepdim=True)  # thermal = mean of RGB

    # replace the conv layer
    new_conv = nn.Conv2d(
        in_channels  = 4,
        out_channels = out_ch,
        kernel_size  = kH,
        stride       = first_conv.conv.stride,
        padding      = first_conv.conv.padding,
        bias         = False
    )
    new_conv.weight = nn.Parameter(new_weight)
    first_conv.conv = new_conv

    print(f"  Modified first conv: 3ch → 4ch input")
    print(f"  New weight shape: {new_weight.shape}")
    return model


# STEP 3 — TRAINING FUNCTION

def train_early_fusion(dataset_name, nc, class_names):
    print(f"\n{'='*60}")
    print(f"  Early Fusion Training: {dataset_name.upper()}")
    print(f"  Classes: {class_names}")
    print(f"{'='*60}")

    # load and modify model
    # Check if a partial training run exists — if so resume from it
    last_weights = MODELS_DIR / f"{dataset_name}_early_fusion" / "weights" / "last.pt"
    if last_weights.exists():
        print(f"  Resuming from {last_weights}")
        model = YOLO(str(last_weights))
    else:
        print(f"  Starting fresh from yolov8n.pt")
        model = YOLO("yolov8n.pt")
        model = modify_yolov8_for_4ch(model)
        model.model.nc = nc
        
    # create datasets
    train_ds = PairedDataset(dataset_name, "train", IMG_SIZE)
    val_ds   = PairedDataset(dataset_name, "test",  IMG_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = PairedDataset.collate_fn,
        num_workers = 0,
        pin_memory  = False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = PairedDataset.collate_fn,
        num_workers = 0,
        pin_memory  = False
    )

# Create a temporary YAML config for early fusion training run

    yaml_path = DATA_DIR / f"{dataset_name}_early_fusion.yaml"
    data_path = DATA_DIR / dataset_name.upper()

    yaml_content = f"""path: {data_path.as_posix()}
train: visible/images/train
val:   visible/images/test

nc: {nc}
names: {class_names}
"""
    # write the yaml file to the data directory
    yaml_path.write_text(yaml_content)
    print(f"  Created {yaml_path.name}")

    # train
    name = f"{dataset_name}_early_fusion"
    results = model.train(
        data        = str(yaml_path),
        epochs      = EPOCHS,
        batch       = BATCH_SIZE,
        imgsz       = IMG_SIZE,
        device      = DEVICE,
        name        = name,
        project     = str(MODELS_DIR),
        exist_ok    = True,
        resume      = True,
        workers     = 0,
        cache       = False,
        optimizer   = "AdamW",
        lr0         = 0.001,
        weight_decay= 0.0005,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        flipud      = 0.0,
        fliplr      = 0.5,
        save        = True,
        save_period = 10,
        verbose     = True,
    )

    print(f"\n Finished Early Fusion: {dataset_name.upper()}")
    print(f"   Best mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    return results


# MAIN

if __name__ == "__main__":
    print("\n Early Fusion Training")
    print("   Concatenating RGB + Thermal → 4-channel YOLOv8n")

    # LLVIP
    train_early_fusion(
        dataset_name = "llvip",
        nc           = 1,
        class_names  = ["person"]
    )

    # FLIR
    train_early_fusion(
        dataset_name = "flir",
        nc           = 3,
        class_names  = ["person", "car", "bicycle"]
    )

    print("\n Early Fusion Complete!")