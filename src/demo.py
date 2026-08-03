import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results" / "demo_outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# DEMO runs all 3 fusion strategies on the same image pair
# and saves side by side comparison


def run_demo(num_images=5):
    """
    Runs all fusion models on sample test images and saves
    side by side comparison showing what each model detects.
    """
    print("\n Running Demo on Sample Images")

    # load all models
    print("\n  Loading models...")
    models = {
        "RGB Baseline":    YOLO(str(MODELS_DIR/"baselines"/"llvip_rgb_only"/"weights"/"best.pt")),
        "Thermal Baseline":YOLO(str(MODELS_DIR/"baselines"/"llvip_thermal_only"/"weights"/"best.pt")),
        "Early Fusion":    YOLO(str(MODELS_DIR/"early_fusion"/"llvip_early_fusion"/"weights"/"best.pt")),
        "Late Fusion":     YOLO(str(MODELS_DIR/"baselines"/"llvip_rgb_only"/"weights"/"best.pt")),
        "Inter. Fusion":   YOLO(str(MODELS_DIR/"intermediate_fusion"/"llvip_intermediate_fusion"/"weights"/"best.pt")),
    }
    thermal_model = YOLO(str(MODELS_DIR/"baselines"/"llvip_thermal_only"/"weights"/"best.pt"))

    # ── get sample test images ──
    rgb_dir = DATA_DIR / "LLVIP" / "visible"  / "images" / "test"
    th_dir  = DATA_DIR / "LLVIP" / "infrared" / "images" / "test"

    test_images = sorted(list(rgb_dir.glob("*.jpg")))[:num_images]
    print(f"  Running on {len(test_images)} sample images...")

    for img_path in test_images:
        stem    = img_path.stem
        th_path = th_dir / img_path.name

        # load images
        rgb_img = cv2.imread(str(img_path))
        th_img  = cv2.imread(str(th_path))

        if rgb_img is None or th_img is None:
            continue

        # resize for display
        h, w = 320, 426
        rgb_display = cv2.resize(rgb_img, (w, h))
        th_display  = cv2.resize(th_img,  (w, h))

        panels = []

        # run each model and draw detections
        for model_name, model in models.items():

            # use thermal image for thermal baseline and late fusion thermal
            if model_name == "Thermal Baseline":
                img_to_use = str(th_path)
            else:
                img_to_use = str(img_path)

            results = model.predict(
                img_to_use,
                conf    = 0.3,
                verbose = False,
                imgsz   = 640
            )[0]

            # draw predictions on image
            annotated = results.plot()
            annotated = cv2.resize(annotated, (w, h))

            # add model name label
            cv2.rectangle(annotated, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(annotated, model_name, (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1)

            # add detection count
            n_det = len(results.boxes) if results.boxes else 0
            cv2.putText(annotated, f"Detections: {n_det}",
                       (5, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)

            panels.append(annotated)

        # add raw RGB and thermal panels
        rgb_panel = rgb_display.copy()
        cv2.rectangle(rgb_panel, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(rgb_panel, "RGB Input", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1)

        th_panel = th_display.copy()
        cv2.rectangle(th_panel, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(th_panel, "Thermal Input", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1)

        # combine into grid
        # row 1: inputs + RGB baseline + Thermal baseline
        row1 = np.hstack([rgb_panel, th_panel, panels[0], panels[1]])
        # row 2: early fusion + intermediate fusion + late fusion + blank
        blank = np.zeros_like(panels[0])
        row2  = np.hstack([panels[2], panels[4], panels[3], blank])

        # add row labels
        label_h = 25
        def add_row_label(row, label):
            bar = np.zeros((label_h, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(bar, label, (10, 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (200, 200, 200), 1)
            return np.vstack([bar, row])

        row1 = add_row_label(row1, "Inputs and Single-Modal Baselines")
        row2 = add_row_label(row2, "Fusion Strategies")

        grid = np.vstack([row1, row2])

        # save output
        out_path = RESULTS_DIR / f"demo_{stem}.jpg"
        cv2.imwrite(str(out_path), grid)
        print(f"  Saved: {out_path.name}")

    print(f"\n Demo complete!")
    print(f"   {num_images} comparison images saved to results/demo_outputs/")
    print(f"   Open them in VS Code to see side-by-side comparisons")


if __name__ == "__main__":
    run_demo(num_images=5)