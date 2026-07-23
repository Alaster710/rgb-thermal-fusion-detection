import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import json
from pathlib import Path
from ultralytics import YOLO

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# SETTINGS
IMG_SIZE      = 640
WARMUP_RUNS   = 10    # warmup runs before timing to stabilise GPU/CPU
TIMING_RUNS   = 100   # number of runs to average latency over

# STEP 1 — EXPORT MODEL TO ONNX
#
# Why ONNX?
# ONNX (Open Neural Network Exchange) is a universal format that
# edge runtimes understand. Converting from PyTorch .pt to .onnx:
# - Removes Python dependency — can run on any device
# - Enables hardware-specific optimisations
# - Allows fair cross-platform benchmarking
# - Is the standard format for edge AI deployment

def export_to_onnx(model_path, onnx_path, img_size=640):
    """
    Exports a trained YOLOv8 .pt model to ONNX format.

    Args:
        model_path: path to trained .pt weights file
        onnx_path:  where to save the .onnx file
        img_size:   input image size for the model
    """
    print(f"  Exporting {Path(model_path).parent.parent.name} to ONNX...")

    model = YOLO(str(model_path))
    model.export(
        format  = "onnx",
        imgsz   = img_size,
        opset   = 12,       # ONNX opset 12 has widest compatibility
        simplify= True,     # simplify graph for faster inference
        dynamic = False,    # fixed batch size for edge deployment
    )

    # ultralytics saves the onnx next to the .pt file
    exported = Path(str(model_path).replace(".pt", ".onnx"))
    if exported.exists():
        exported.rename(onnx_path)
        print(f"  Saved to {onnx_path.name}")
        return True
    else:
        print(f"  Export failed — {exported} not found")
        return False


# STEP 2 — MEASURE MODEL SIZE
#
# Model size in MB directly determines:
# - How much flash/storage is needed on edge device
# - Whether the model fits in cache memory
# - Download/update time over-the-air

def get_model_size_mb(model_path):
    """Returns model file size in megabytes."""
    size_bytes = Path(model_path).stat().st_size
    size_mb    = size_bytes / (1024 * 1024)
    return round(size_mb, 2)


# STEP 3 — MEASURE INFERENCE LATENCY

# Latency (ms per image) determines real-time capability.
# We use ONNX Runtime for benchmarking because:
# - It's the standard edge inference engine
# - More representative of real deployment than PyTorch
# - Hardware-agnostic — same results across different devices

# We measure on CPU specifically because:
# Edge devices typically don't have discrete GPUs
# CPU latency is what matters for deployment decisions

def measure_latency(onnx_path, img_size=640, warmup=10, runs=100):
    """
    Measures inference latency using ONNX Runtime on CPU.
    Simulates edge deployment conditions.

    Args:
        onnx_path: path to .onnx model file
        img_size:  input image size
        warmup:    number of warmup runs (not counted in timing)
        runs:      number of timed runs to average

    Returns:
        mean_ms:   average latency in milliseconds
        std_ms:    standard deviation of latency
    """
    # create ONNX Runtime session on CPU
    # CPUExecutionProvider simulates edge hardware
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = \
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=["CPUExecutionProvider"]
    )

    # get input name and shape from model
    input_name  = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # determine number of input channels
    # early fusion has 4 channels, others have 3
    channels = input_shape[1] if input_shape[1] != "batch" else 3

    # create dummy input image (random pixels, normalised 0-1)
    dummy_input = np.random.rand(1, channels, img_size, img_size).astype(np.float32)

    # warmup runs
    print(f"    Warming up ({warmup} runs)...")
    for _ in range(warmup):
        session.run(None, {input_name: dummy_input})

    # timed runs
    print(f"    Timing ({runs} runs)...")
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end   = time.perf_counter()
        latencies.append((end - start) * 1000)  # convert to ms

    mean_ms = round(float(np.mean(latencies)),   2)
    std_ms  = round(float(np.std(latencies)),    2)
    return mean_ms, std_ms


# STEP 4 — COUNT PARAMETERS AND ESTIMATE FLOPs

# Parameters: number of learnable weights in the model
# FLOPs: floating point operations per inference
# Both directly impact:
# Memory requirements (parameters × 4 bytes for float32)
# Computational cost (FLOPs → battery drain on edge device)

def count_params_and_flops(model_path, img_size=640):
    """
    Counts model parameters and estimates FLOPs using thop library.

    Args:
        model_path: path to .pt weights file
        img_size:   input image size

    Returns:
        params_m: number of parameters in millions
        flops_g:  estimated GFLOPs
    """
    try:
        from thop import profile
        model = YOLO(str(model_path)).model.eval()

        # determine input channels
        first_conv = list(model.parameters())[0]
        in_channels = first_conv.shape[1]

        dummy = torch.zeros(1, in_channels, img_size, img_size)

        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy,), verbose=False)

        params_m = round(params / 1e6, 2)
        flops_g  = round(flops  / 1e9, 2)
        return params_m, flops_g

    except Exception as e:
        print(f"    FLOPs calculation failed: {e}")
        return 0.0, 0.0



# STEP 5 — FULL BENCHMARK PIPELINE
# Runs all measurements for one model and returns results dict

def benchmark_model(name, pt_path, onnx_dir):
    """
    Full benchmark pipeline for one model:
    1. Export to ONNX
    2. Measure file size
    3. Measure latency
    4. Count params and FLOPs

    Args:
        name:     display name for this model
        pt_path:  path to trained .pt weights
        onnx_dir: directory to save .onnx file

    Returns:
        dict with all benchmark metrics
    """
    print(f"\n  Benchmarking: {name}")
    print(f"  {'-'*50}")

    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / f"{name}.onnx"

    # export to ONNX
    if not onnx_path.exists():
        success = export_to_onnx(pt_path, onnx_path)
        if not success:
            return None
    else:
        print(f"  ONNX already exists — skipping export")

    # measure size
    pt_size_mb   = get_model_size_mb(pt_path)
    onnx_size_mb = get_model_size_mb(onnx_path)
    print(f"  Size: {pt_size_mb} MB (PyTorch)  {onnx_size_mb} MB (ONNX)")

    # measure latency
    mean_ms, std_ms = measure_latency(onnx_path, IMG_SIZE, WARMUP_RUNS, TIMING_RUNS)
    print(f"  Latency: {mean_ms} ± {std_ms} ms (CPU)")

    # count params and FLOPs
    params_m, flops_g = count_params_and_flops(pt_path, IMG_SIZE)
    print(f"  Params: {params_m}M   FLOPs: {flops_g}G")

    return {
        "name":          name,
        "pt_size_mb":    pt_size_mb,
        "onnx_size_mb":  onnx_size_mb,
        "latency_ms":    mean_ms,
        "latency_std":   std_ms,
        "params_m":      params_m,
        "flops_g":       flops_g,
    }



# MAIN — benchmark all models

if __name__ == "__main__":
    print("\n Edge Deployment Benchmarking")
    print("   Exporting all models to ONNX and measuring:")
    print("   - Model size (MB)")
    print("   - Inference latency (ms) on CPU")
    print("   - Parameters (M)")
    print("   - FLOPs (G)")

    onnx_dir = MODELS_DIR / "onnx_exports"

    # define all models to benchmark
    # format: (display_name, path_to_best.pt)
    models_to_benchmark = [
        # baselines
        ("llvip_rgb_baseline",
         MODELS_DIR/"baselines"/"llvip_rgb_only"/"weights"/"best.pt"),
        ("llvip_thermal_baseline",
         MODELS_DIR/"baselines"/"llvip_thermal_only"/"weights"/"best.pt"),
        ("flir_rgb_baseline",
         MODELS_DIR/"baselines"/"flir_rgb_only"/"weights"/"best.pt"),
        ("flir_thermal_baseline",
         MODELS_DIR/"baselines"/"flir_thermal_only"/"weights"/"best.pt"),

        # early fusion
        ("llvip_early_fusion",
         MODELS_DIR/"early_fusion"/"llvip_early_fusion"/"weights"/"best.pt"),
        ("flir_early_fusion",
         MODELS_DIR/"early_fusion"/"flir_early_fusion"/"weights"/"best.pt"),

        # intermediate fusion
        ("llvip_intermediate_fusion",
         MODELS_DIR/"intermediate_fusion"/"llvip_intermediate_fusion"/"weights"/"best.pt"),
        ("flir_intermediate_fusion",
         MODELS_DIR/"intermediate_fusion"/"flir_intermediate_fusion"/"weights"/"best.pt"),
    ]

    all_results = []

    for name, pt_path in models_to_benchmark:
        if not pt_path.exists():
            print(f"\n  Skipping {name} — weights not found at {pt_path}")
            continue

        result = benchmark_model(name, pt_path, onnx_dir)
        if result:
            all_results.append(result)

    # late fusion note
    # Late fusion uses TWO models at inference time
    # so its effective cost = RGB model + Thermal model
    print("\n  Note: Late fusion uses two models simultaneously")
    print("  Effective late fusion cost = RGB baseline + Thermal baseline")

    # print final summary table
    print(f"\n{'='*75}")
    print(f"  EDGE BENCHMARKING RESULTS")
    print(f"{'='*75}")
    print(f"  {'Model':<35} {'Size(MB)':<10} {'Latency(ms)':<14} {'Params(M)':<12} {'FLOPs(G)'}")
    print(f"  {'-'*75}")

    for r in all_results:
        print(f"  {r['name']:<35} {r['onnx_size_mb']:<10} "
              f"{r['latency_ms']:<14} {r['params_m']:<12} {r['flops_g']}")

    # save results to JSON
    results_file = RESULTS_DIR / "edge_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # calculate late fusion combined cost
    rgb_results = next((r for r in all_results
                        if "rgb_baseline" in r["name"] and "llvip" in r["name"]), None)
    th_results  = next((r for r in all_results
                        if "thermal_baseline" in r["name"] and "llvip" in r["name"]), None)

    if rgb_results and th_results:
        late_size    = rgb_results["onnx_size_mb"] + th_results["onnx_size_mb"]
        late_latency = rgb_results["latency_ms"]   + th_results["latency_ms"]
        late_params  = rgb_results["params_m"]     + th_results["params_m"]
        late_flops   = rgb_results["flops_g"]      + th_results["flops_g"]
        print(f"\n  Late Fusion (LLVIP) combined cost:")
        print(f"  Size: {late_size:.2f} MB  "
              f"Latency: {late_latency:.2f} ms  "
              f"Params: {late_params:.2f}M  "
              f"FLOPs: {late_flops:.2f}G")

    print(f"\n Benchmarking Complete!")