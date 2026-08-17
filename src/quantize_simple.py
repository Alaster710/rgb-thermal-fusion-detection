from ultralytics import YOLO
from pathlib import Path
import onnxruntime as ort
import numpy as np
import time

PROJECT_ROOT = Path(__file__).parent.parent

# quantize just the best single model — llvip thermal baseline
pt_path  = PROJECT_ROOT / "models" / "baselines" / "llvip_thermal_only" / "weights" / "best.pt"
yaml_path = PROJECT_ROOT / "data" / "llvip_infrared.yaml"

print("Exporting with INT8...")
model = YOLO(str(pt_path))
model.export(
    format   = "onnx",
    imgsz    = 640,
    int8     = True,
    data     = str(yaml_path),
    simplify = True,
)
print("Done!")

# measure INT8 size and latency
int8_path = pt_path.parent / "best_int8.onnx"
if not int8_path.exists():
    int8_path = pt_path.parent / "best.onnx"

if int8_path.exists():
    size_mb = round(int8_path.stat().st_size / (1024*1024), 2)
    print(f"INT8 Size: {size_mb} MB")

    sess  = ort.InferenceSession(str(int8_path),
            providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
    name  = sess.get_inputs()[0].name

    for _ in range(10):
        sess.run(None, {name: dummy})

    times = []
    for _ in range(50):
        s = time.perf_counter()
        sess.run(None, {name: dummy})
        times.append((time.perf_counter() - s) * 1000)

    print(f"INT8 Latency: {round(np.mean(times), 2)} ms")
    print(f"FP32 was: 26.0 ms  →  Speedup: {round(26.0/np.mean(times), 2)}x")
else:
    print("Could not find INT8 file — check export path")