import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models" / "late_fusion"
BASELINE_DIR = PROJECT_ROOT / "models" / "baselines"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# SETTINGS
EPOCHS     = 20
BATCH_SIZE = 4
IMG_SIZE   = 640
DEVICE     = 0

# STEP 1 — TRAIN SINGLE MODALITY DETECTORS
# Late fusion needs two separate detectors:
#   - one trained on RGB images only
#   - one trained on thermal images only
# We reuse the baseline weights directly instead of retraining.

def get_baseline_weights(dataset, modal):
    """
    Returns path to already trained baseline weights.
    We reuse baselines instead of retraining to save time and ensure
    fair comparison — same weights, just combined differently.
    """
    # map modal name to baseline folder name
    modal_map = {
        "visible":  "rgb_only",
        "infrared": "thermal_only"
    }
    folder = f"{dataset}_{modal_map[modal]}"
    weights = BASELINE_DIR / folder / "weights" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(
            f"Baseline weights not found: {weights}\n"
            f"Make sure train_baselines.py has been run first."
        )
    print(f"  Found baseline weights: {folder}/weights/best.pt")
    return str(weights)


# STEP 2 — WEIGHTED BOX FUSION (WBF)
# After both detectors make predictions on the same image,
# WBF combines them into a single set of final predictions.

# How WBF works:
# 1. Both detectors produce boxes with confidence scores
# 2. Boxes from both detectors are clustered by overlap (IoU)
# 3. Clustered boxes are averaged weighted by their confidence scores
# 4. This gives more accurate box positions than just taking one detector's output
#
# Why WBF instead of NMS (Non-Maximum Suppression)?
# NMS just picks the highest scoring box and throws away others.
# WBF averages all overlapping boxes together, producing better localisation, especially when the two detectors see the object
# from slightly different feature perspectives.

def run_wbf(rgb_results, thermal_results, img_size=640,
            iou_thr=0.55, skip_box_thr=0.01, weights=None):
    """
    Applies Weighted Box Fusion to combine predictions from
    RGB detector and thermal detector on the same image.

    Args:
        rgb_results:     ultralytics Results object from RGB detector
        thermal_results: ultralytics Results object from thermal detector
        img_size:        image size used during inference
        iou_thr:         IoU threshold for clustering boxes
        skip_box_thr:    minimum confidence to include a box
        weights:         importance weight for each detector [rgb_w, thermal_w]
                         default [1,1] means equal weight to both detectors

    Returns:
        fused_boxes:  numpy array of fused bounding boxes [x1,y1,x2,y2] normalised
        fused_scores: numpy array of confidence scores
        fused_labels: numpy array of class labels
    """
    if weights is None:
        weights = [1, 1]  # equal weight to both detectors by default

    boxes_list  = []
    scores_list = []
    labels_list = []

    # process RGB predictions 
    for result in [rgb_results, thermal_results]:
        if result.boxes is not None and len(result.boxes) > 0:
            # get boxes in xyxy format normalised to [0,1]
            boxes  = result.boxes.xyxyn.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            # clip to valid range just in case
            boxes = np.clip(boxes, 0, 1)
            boxes_list.append(boxes.tolist())
            scores_list.append(scores.tolist())
            labels_list.append(labels.tolist())
        else:
            # no detections from this detector — add empty lists
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

    # apply WBF 
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights    = weights,
        iou_thr    = iou_thr,
        skip_box_thr = skip_box_thr
    )

    return fused_boxes, fused_scores, fused_labels


# STEP 3 — EVALUATE LATE FUSION ON TEST SET
# Runs both detectors on every test image, applies WBF,
# then calculates mAP50 and mAP50-95 across the whole test set.


def evaluate_late_fusion(dataset, nc, class_names):
    """
    Evaluates late fusion performance on the test set.

    Process:
    1. Load RGB and thermal baseline detectors
    2. For each test image pair:
       a. Run RGB detector on visible image
       b. Run thermal detector on infrared image
       c. Apply WBF to combine predictions
    3. Calculate mAP50 and mAP50-95 using ultralytics validator
    """
    print(f"\n{'='*60}")
    print(f"  Late Fusion Evaluation: {dataset.upper()}")
    print(f"  Classes: {class_names}")
    print(f"{'='*60}")

    # load both baseline detectors 
    rgb_weights     = get_baseline_weights(dataset, "visible")
    thermal_weights = get_baseline_weights(dataset, "infrared")

    print(f"\n  Loading RGB detector...")
    rgb_model     = YOLO(rgb_weights)

    print(f"  Loading Thermal detector...")
    thermal_model = YOLO(thermal_weights)

    # get test image paths 
    rgb_test_dir = DATA_DIR / dataset.upper() / "visible"  / "images" / "test"
    th_test_dir  = DATA_DIR / dataset.upper() / "infrared" / "images" / "test"
    lbl_test_dir = DATA_DIR / dataset.upper() / "visible"  / "labels" / "test"

    # get sorted list of test images
    test_files = sorted([f for f in rgb_test_dir.iterdir()
                         if f.suffix in [".jpg", ".jpeg", ".png"]])

    print(f"\n  Running inference on {len(test_files)} test image pairs...")

    # storage for predictions and ground truth
    all_predictions = []  # list of dicts with boxes, scores, labels per image
    all_targets     = []  # list of dicts with ground truth per image

    for i, rgb_path in enumerate(test_files):
        stem    = rgb_path.stem
        th_path = th_test_dir / rgb_path.name
        if not th_path.exists():
            th_path = th_test_dir / (stem + ".jpeg")

        # run both detectors
        rgb_pred = rgb_model.predict(
            str(rgb_path),
            imgsz   = IMG_SIZE,
            conf    = 0.001,   # low threshold to get all possible detections
            iou     = 0.6,
            verbose = False,
            device  = DEVICE
        )[0]

        th_pred = thermal_model.predict(
            str(th_path),
            imgsz   = IMG_SIZE,
            conf    = 0.001,
            iou     = 0.6,
            verbose = False,
            device  = DEVICE
        )[0]

        # apply WBF
        fused_boxes, fused_scores, fused_labels = run_wbf(
            rgb_pred, th_pred,
            img_size = IMG_SIZE,
            iou_thr  = 0.55,
            weights  = [1, 1]   # equal weight — can tune this later
        )

        all_predictions.append({
            "boxes":  fused_boxes,
            "scores": fused_scores,
            "labels": fused_labels,
            "path":   str(rgb_path)
        })

        # load ground truth labels
        lbl_path = lbl_test_dir / (stem + ".txt")
        gt_boxes  = []
        gt_labels = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) == 5:
                        cls, cx, cy, w, h = map(float, vals)
                        # convert yolo format to xyxy normalised
                        x1 = cx - w/2
                        y1 = cy - h/2
                        x2 = cx + w/2
                        y2 = cy + h/2
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_labels.append(int(cls))

        all_targets.append({
            "boxes":  np.array(gt_boxes)  if gt_boxes  else np.zeros((0,4)),
            "labels": np.array(gt_labels) if gt_labels else np.zeros(0, dtype=int),
            "path":   str(rgb_path)
        })

        # print progress every 100 images
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_files)} images...")

    # ── calculate mAP using torchmetrics ──
    print(f"\n  Calculating mAP...")
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(
        box_format     = "xyxy",
        iou_type       = "bbox",
    )

    for pred, target in zip(all_predictions, all_targets):
        # format predictions for torchmetrics
        if len(pred["boxes"]) > 0:
            pred_dict = {
                "boxes":  torch.tensor(
                              np.array(pred["boxes"]),
                              dtype=torch.float32),
                "scores": torch.tensor(
                              np.array(pred["scores"]),
                              dtype=torch.float32),
                "labels": torch.tensor(
                              np.array(pred["labels"]),
                              dtype=torch.int64),
            }
        else:
            pred_dict = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0,      dtype=torch.float32),
                "labels": torch.zeros(0,      dtype=torch.int64),
            }

        # format ground truth for torchmetrics
        if len(target["boxes"]) > 0:
            target_dict = {
                "boxes":  torch.tensor(
                              target["boxes"],
                              dtype=torch.float32),
                "labels": torch.tensor(
                              target["labels"],
                              dtype=torch.int64),
            }
        else:
            target_dict = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0,      dtype=torch.int64),
            }

        metric.update([pred_dict], [target_dict])

    # compute final metrics
    results   = metric.compute()
    map50     = results["map_50"].item()
    map5095   = results["map"].item()

    print(f"  Raw torchmetrics output: {results}")

    # save results
    results_dir = MODELS_DIR / f"{dataset}_late_fusion"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.txt", "w") as f:
        f.write(f"Late Fusion Results — {dataset.upper()}\n")
        f.write(f"{'='*40}\n")
        f.write(f"RGB detector:     {rgb_weights}\n")
        f.write(f"Thermal detector: {thermal_weights}\n")
        f.write(f"WBF iou_thr:      0.55\n")
        f.write(f"WBF weights:      [1, 1]\n")
        f.write(f"{'='*40}\n")
        f.write(f"mAP50:            {map50:.4f}\n")
        f.write(f"mAP50-95:         {map5095:.4f}\n")
        f.write(f"Test images:      {len(test_files)}\n")

    print(f"   Results saved to {results_dir}/results.txt")
    return map50, map5095



# MAIN

if __name__ == "__main__":
    print("\n Late Fusion Evaluation")
    print("   Strategy: RGB detector + Thermal detector → WBF")
    print("   Reusing baseline weights — no retraining needed!")

    # LLVIP
    llvip_map50, llvip_map5095 = evaluate_late_fusion(
        dataset     = "llvip",
        nc          = 1,
        class_names = ["person"]
    )

    # FLIR
    flir_map50, flir_map5095 = evaluate_late_fusion(
        dataset     = "flir",
        nc          = 3,
        class_names = ["person", "car", "bicycle"]
    )

    # print final summary
    print(f"\n{'='*60}")
    print(f"  LATE FUSION FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  LLVIP  mAP50: {llvip_map50:.4f}  mAP50-95: {llvip_map5095:.4f}")
    print(f"  FLIR   mAP50: {flir_map50:.4f}  mAP50-95: {flir_map5095:.4f}")
    print(f"\n  Compare against baselines:")
    print(f"  LLVIP  RGB:0.885  Thermal:0.958  EarlyFusion:0.894")
    print(f"  FLIR   RGB:0.524  Thermal:0.657  EarlyFusion:0.518")
    print(f"{'='*60}")
    print(f"\n Late Fusion Complete!")