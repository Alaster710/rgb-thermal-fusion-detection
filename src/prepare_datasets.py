import os
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# Paths to the datasets
LLVIP_ROOT  = r"C:\Users\Alaster\Downloads\Capstone Project\Datasets\LLVIP\LLVIP"
FLIR_ROOT   = r"C:\Users\Alaster\Downloads\Capstone Project\Datasets\flir_align"
PROJECT_ROOT = Path(__file__).parent.parent  # rgb-thermal-fusion-detection/

LLVIP_OUT   = PROJECT_ROOT / "data" / "LLVIP"
FLIR_OUT    = PROJECT_ROOT / "data" / "FLIR"


# Creating the output directories
def make_dirs(base):
    for split in ["train", "test"]:
        for modal in ["infrared", "visible", "labels"]:
            (base / modal / split).mkdir(parents=True, exist_ok=True)
            
            
# Convert LLVIP XML to YOLO txt
def convert_llvip_xml_to_yolo(xml_path, label_out_path, img_w=1280, img_h=1024):
    """Convert a single LLVIP Pascal-VOC XML file to a YOLO .txt label file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        img_w = int(size.find("width").text)
        img_h = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        cls = 0          # LLVIP has only one class: person
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_out_path, "w") as f:
        f.write("\n".join(lines))

# Prepare LLVIP dataset
def prepare_llvip():
    print("\n=== Preparing LLVIP ===")
    make_dirs(LLVIP_OUT)

    ann_dir = Path(LLVIP_ROOT) / "Annotations"

    # figure out which images belong to train vs test
    train_ir = set(p.stem for p in (Path(LLVIP_ROOT)/"infrared"/"train").glob("*.jpg"))
    test_ir  = set(p.stem for p in (Path(LLVIP_ROOT)/"infrared"/"test").glob("*.jpg"))

    for split, stems in [("train", train_ir), ("test", test_ir)]:
        copied_img = 0
        converted_lbl = 0

        for stem in stems:
            # ── copy infrared image ──
            src_ir = Path(LLVIP_ROOT) / "infrared" / split / f"{stem}.jpg"
            dst_ir = LLVIP_OUT / "infrared" / split / f"{stem}.jpg"
            if src_ir.exists():
                shutil.copy2(src_ir, dst_ir)
                copied_img += 1

            # ── copy visible image ──
            src_vis = Path(LLVIP_ROOT) / "visible" / split / f"{stem}.jpg"
            dst_vis = LLVIP_OUT / "visible" / split / f"{stem}.jpg"
            if src_vis.exists():
                shutil.copy2(src_vis, dst_vis)

            # ── convert XML label ──
            xml_path = ann_dir / f"{stem}.xml"
            dst_lbl  = LLVIP_OUT / "labels" / split / f"{stem}.txt"
            if xml_path.exists():
                convert_llvip_xml_to_yolo(xml_path, dst_lbl)
                converted_lbl += 1
            else:
                # create empty label file if no annotation exists
                dst_lbl.touch()

        print(f"  LLVIP {split}: {copied_img} images, {converted_lbl} labels converted")

# ══════════════════════════════════════════════════════════════════════════════
# FLIR  — COCO JSON  →  YOLO txt
# ══════════════════════════════════════════════════════════════════════════════

# FLIR classes we care about (from the aligned dataset)
FLIR_CLASS_MAP = {1: 0, 2: 1, 3: 2}   # person→0, car→1, bicycle→2
FLIR_CLASS_NAMES = {1: "person", 2: "car", 3: "bicycle"}

def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x,y,w,h] to YOLO [cx,cy,w,h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def prepare_flir():
    print("\n=== Preparing FLIR ===")
    make_dirs(FLIR_OUT)

    ann_root = Path(FLIR_ROOT) / "coco_annotations"

    for split in ["train", "test"]:
        # prefer *_new.json if it exists (cleaned version)
        json_new = ann_root / f"{split}_new.json"
        json_old = ann_root / f"{split}.json"
        json_path = json_new if json_new.exists() else json_old

        print(f"  Loading {json_path.name} ...")
        with open(json_path, "r") as f:
            coco = json.load(f)

        # build id → filename map
        id2info = {img["id"]: img for img in coco["images"]}

        # group annotations by image_id
        ann_by_img = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        copied_img = 0
        converted_lbl = 0

        for img_id, img_info in id2info.items():
            fname = Path(img_info["file_name"]).name   # e.g. FLIR_00002.jpeg
            stem  = Path(fname).stem                   # e.g. FLIR_00002
            img_w = img_info["width"]
            img_h = img_info["height"]

            # ── copy thermal image ──
            src_th = Path(FLIR_ROOT) / "thermal" / split / fname
            dst_th = FLIR_OUT / "infrared" / split / fname
            if src_th.exists():
                shutil.copy2(src_th, dst_th)
                copied_img += 1

            # ── copy visible image ──
            src_vis = Path(FLIR_ROOT) / "visible" / split / fname
            dst_vis = FLIR_OUT / "visible" / split / fname
            if src_vis.exists():
                shutil.copy2(src_vis, dst_vis)

            # ── convert annotations ──
            dst_lbl = FLIR_OUT / "labels" / split / f"{stem}.txt"
            anns = ann_by_img.get(img_id, [])
            lines = []
            for ann in anns:
                cat_id = ann["category_id"]
                if cat_id not in FLIR_CLASS_MAP:
                    continue      # skip classes we don't want
                yolo_cls = FLIR_CLASS_MAP[cat_id]
                cx, cy, w, h = coco_to_yolo(ann["bbox"], img_w, img_h)
                lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))
            converted_lbl += 1

        print(f"  FLIR {split}: {copied_img} images, {converted_lbl} labels converted")

# ══════════════════════════════════════════════════════════════════════════════
# YAML config files for YOLOv8
# ══════════════════════════════════════════════════════════════════════════════
def write_yaml_files():
    llvip_yaml = PROJECT_ROOT / "data" / "llvip.yaml"
    llvip_yaml.write_text(f"""path: {LLVIP_OUT.as_posix()}
train: infrared/train
val:   infrared/test

nc: 1
names: ['person']
""")
    print(f"\n  Written {llvip_yaml}")

    flir_yaml = PROJECT_ROOT / "data" / "flir.yaml"
    flir_yaml.write_text(f"""path: {FLIR_OUT.as_posix()}
train: infrared/train
val:   infrared/test

nc: 3
names: ['person', 'car', 'bicycle']
""")
    print(f"  Written {flir_yaml}")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    prepare_llvip()
    prepare_flir()
    write_yaml_files()
    print("\n Both datasets ready!")