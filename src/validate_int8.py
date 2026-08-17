from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# validate FP32 vs INT8 on LLVIP thermal
print("\n=== FP32 Validation ===")
fp32_model = YOLO(str(PROJECT_ROOT / "models" / "baselines" / 
                      "llvip_thermal_only" / "weights" / "best.pt"))
fp32_results = fp32_model.val(
    data   = str(PROJECT_ROOT / "data" / "llvip_infrared.yaml"),
    imgsz  = 640,
    device = "cpu",
    verbose= False,
)
fp32_map = fp32_results.results_dict.get("metrics/mAP50(B)", 0)
print(f"FP32 mAP50: {fp32_map:.4f}")

print("\n=== INT8 Validation ===")
int8_model = YOLO(str(PROJECT_ROOT / "models" / "baselines" /
                      "llvip_thermal_only" / "weights" / "best_int8.onnx"))
int8_results = int8_model.val(
    data   = str(PROJECT_ROOT / "data" / "llvip_infrared.yaml"),
    imgsz  = 640,
    device = "cpu",
    verbose= False,
)
int8_map = int8_results.results_dict.get("metrics/mAP50(B)", 0)
print(f"INT8 mAP50: {int8_map:.4f}")

print(f"\n=== Summary ===")
print(f"FP32 mAP50:    {fp32_map:.4f}")
print(f"INT8 mAP50:    {int8_map:.4f}")
print(f"Accuracy drop: {((fp32_map - int8_map)/fp32_map)*100:.2f}%")
print(f"Size FP32:     11.7 MB")
print(f"Size INT8:     3.2 MB")
print(f"Size saving:   72.4%")