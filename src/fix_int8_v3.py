import onnx
import onnx.helper as helper
import onnxruntime as ort
import numpy as np
import cv2
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

int8_path  = PROJECT_ROOT / "models" / "baselines" / \
             "llvip_thermal_only" / "weights" / "best_int8.onnx"
fp32_path  = PROJECT_ROOT / "models" / "baselines" / \
             "llvip_thermal_only" / "weights" / "best.onnx"
fixed_path = PROJECT_ROOT / "models" / "baselines" / \
             "llvip_thermal_only" / "weights" / "best_int8_v3.onnx"
lbl_dir    = PROJECT_ROOT / "data" / "LLVIP" / \
             "infrared" / "labels" / "test"
img_dir    = PROJECT_ROOT / "data" / "LLVIP" / \
             "infrared" / "images" / "test"

# ── find ALL sigmoid outputs in model.22 ──
print("Finding all model.22 Sigmoid outputs...")
model = onnx.load(str(int8_path))
graph = model.graph

sigmoid_nodes = []
for node in graph.node:
    if node.op_type == "DequantizeLinear":
        inp = str(node.input[0])
        if "model.22" in inp and "Sigmoid" in inp:
            sigmoid_nodes.append({
                "input": inp,
                "output": node.output[0]
            })
            print(f"  {inp} → {node.output[0]}")

print(f"\nFound {len(sigmoid_nodes)} sigmoid nodes in model.22")

# ── check shape of each sigmoid output ──
print("\nChecking shapes...")
test_img = sorted(list(img_dir.glob("*.jpg")))[0]
img = cv2.imread(str(test_img))
img_r = cv2.resize(img, (640, 640))
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
tensor = img_r.astype(np.float32) / 255.0
tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]

model_debug = onnx.load(str(int8_path))
for sn in sigmoid_nodes:
    model_debug.graph.output.append(
        helper.make_tensor_value_info(
            sn["output"], onnx.TensorProto.FLOAT, None))

import tempfile, os
tmp = tempfile.mktemp(suffix=".onnx")
onnx.save(model_debug, tmp)
sess_tmp = ort.InferenceSession(
    tmp, providers=["CPUExecutionProvider"])
outputs = sess_tmp.run(
    None, {sess_tmp.get_inputs()[0].name: tensor})
os.unlink(tmp)

# outputs[0] = original, rest are sigmoid outputs
print(f"Original output shape: {outputs[0].shape}")
for i, sn in enumerate(sigmoid_nodes):
    out = outputs[i+1]
    print(f"  Sigmoid {i}: shape={out.shape} "
          f"max={out.max():.3f} "
          f"above_0.25={(out > 0.25).sum()}")

# find sigmoid with shape matching 8400 detections
target_sigmoid = None
for i, sn in enumerate(sigmoid_nodes):
    out = outputs[i+1]
    if 8400 in out.shape:
        target_sigmoid = sn
        target_scores  = outputs[i+1]
        print(f"\nFound matching sigmoid: {sn['output']}")
        print(f"Shape: {out.shape}")
        break

if target_sigmoid is None:
    print("\nNo sigmoid with 8400 detections found.")
    print("Trying to find via Concat node input...")
    for node in graph.node:
        if node.op_type == "Concat":
            inputs = list(node.input)
            if any("output0" in str(n.output)
                   for n in graph.node
                   if node.output[0] in str(n.input)):
                print(f"Found Concat: {inputs}")
                for inp in inputs:
                    print(f"  Input: {inp}")
else:
    # ── build fixed model with correct outputs ──
    print("\nBuilding fixed model...")
    model_fixed = onnx.load(str(int8_path))
    graph_fixed = model_fixed.graph

    # find box output
    box_output = None
    for node in graph_fixed.node:
        if node.op_type == "DequantizeLinear" and \
           "model.22" in str(node.input) and \
           "Mul_2" in str(node.input):
            box_output = node.output[0]
            break

    while graph_fixed.output:
        graph_fixed.output.pop()

    graph_fixed.output.append(
        helper.make_tensor_value_info(
            box_output, onnx.TensorProto.FLOAT, None))
    graph_fixed.output.append(
        helper.make_tensor_value_info(
            target_sigmoid["output"],
            onnx.TensorProto.FLOAT, None))

    onnx.save(model_fixed, str(fixed_path))
    print(f"Saved: {fixed_path.name}")

    # ── test scores ──
    print("\nTesting fixed model scores...")
    sess_fixed = ort.InferenceSession(
        str(fixed_path), providers=["CPUExecutionProvider"])

    test_imgs = sorted(list(img_dir.glob("*.jpg")))[:5]
    for img_path in test_imgs:
        img_t = cv2.imread(str(img_path))
        img_t = cv2.resize(img_t, (640, 640))
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        t     = img_t.astype(np.float32)/255.0
        t     = np.transpose(t,(2,0,1))[np.newaxis]

        boxes_raw, scores_raw = sess_fixed.run(
            None, {sess_fixed.get_inputs()[0].name: t})

        print(f"\n{img_path.name}:")
        print(f"  Box shape:   {boxes_raw.shape}")
        print(f"  Score shape: {scores_raw.shape}")
        print(f"  Score max:   {scores_raw.max():.4f}")
        print(f"  Above 0.25:  {(scores_raw > 0.25).sum()}")

    print("\nThis intermediate fix found the correct Sigmoid.")
    print("Run fix_int8_v4.py for the full evaluation.")