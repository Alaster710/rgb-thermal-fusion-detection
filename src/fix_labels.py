import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

datasets = ["LLVIP", "FLIR"]
modals = ["visible", "infrared"]
splits = ["train", "test"]

for dataset in datasets:
    for modal in modals:
        for split in splits:
            # Source labels folder
            src_labels = DATA_DIR / dataset / "labels" / split
            # Destination — next to images
            dst_labels = DATA_DIR / dataset / modal / "labels" / split
            dst_labels.mkdir(parents=True, exist_ok=True)

            if not src_labels.exists():
                print(f"  Skipping {dataset}/{modal}/{split} — source not found")
                continue

            # Copy all label files
            label_files = list(src_labels.glob("*.txt"))
            for lbl in label_files:
                shutil.copy2(lbl, dst_labels / lbl.name)

            print(f"{dataset}/{modal}/{split}: {len(label_files)} labels copied")

print("\n Labels fixed!")