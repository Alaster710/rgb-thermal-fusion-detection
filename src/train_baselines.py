import os
from pathlib import Path
from ultralytics import YOLO

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models" / "baselines"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training Parameters
EPOCHS     = 20        # number of times model sees full dataset
BATCH_SIZE = 4         # images per training step
IMG_SIZE   = 640       # standard YOLO input size
DEVICE     = 0         


# HELPER — modify YAML to point at correct image folder
def make_modal_yaml(dataset, modal):
    base_yaml = DATA_DIR / f"{dataset}.yaml"
    out_yaml  = DATA_DIR / f"{dataset}_{modal}.yaml"

    lines = base_yaml.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("train:"):
            new_lines.append(f"train: {modal}/images/train")
        elif line.startswith("val:"):
            new_lines.append(f"val:   {modal}/images/test")
        else:
            new_lines.append(line)

    out_yaml.write_text("\n".join(new_lines))
    print(f"  Created {out_yaml.name}")
    return str(out_yaml)


# Training Function
def train_baseline(name, yaml_path):
    """Train a single YOLOv8n baseline model."""
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"  Config:   {yaml_path}")
    print(f"{'='*60}")

    # Load fresh pretrained YOLOv8n weights from COCO
    model = YOLO("yolov8n.pt")

    # Train
    results = model.train(
        data    = yaml_path,
        epochs  = EPOCHS,
        batch   = BATCH_SIZE,
        imgsz   = IMG_SIZE,
        device  = DEVICE,
        name    = name,
        project = str(MODELS_DIR),
        exist_ok= True,
        workers = 0,
        cache   = False,

        # Optimiser settings
        optimizer = "AdamW",
        lr0       = 0.001,
        weight_decay = 0.0005,

        # Augmentation — keep simple for baselines
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        flipud = 0.0,
        fliplr = 0.5,

        # Save best model only
        save      = True,
        save_period = 10,    # also save checkpoint every 10 epochs

        # Logging
        verbose = True,
    )

    print(f"\n Finished training {name}")
    print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    return results

# Training all 4 baselines
if __name__ == "__main__":

    print("\n Starting Baseline Training")
    print("   4 models to train:")
    print("   1. LLVIP RGB-only")
    print("   2. LLVIP Thermal-only")
    print("   3. FLIR RGB-only")
    print("   4. FLIR Thermal-only")

    # 1. LLVIP RGB-only
   # yaml = make_modal_yaml("llvip", "visible")
    # train_baseline("llvip_rgb_only", yaml)

    # 2. LLVIP Thermal-only
   # yaml = make_modal_yaml("llvip", "infrared")
    # train_baseline("llvip_thermal_only", yaml)

    # 3. FLIR RGB-only
    yaml = make_modal_yaml("flir", "visible")
    train_baseline("flir_rgb_only", yaml)

    # 4. FLIR Thermal-only
    yaml = make_modal_yaml("flir", "infrared")
    train_baseline("flir_thermal_only", yaml)

    print("\n All baselines trained!")
    print(f"Results saved to: {MODELS_DIR}")