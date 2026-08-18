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
             "llvip_thermal_only" / "weights" / "best_int8_v4.onnx"
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
img_r = cv2.resize(img, (640,640))
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
tensor = img_r.astype(np.float32)/255.0
tensor = np.transpose(tensor,(2,0,1))[np.newaxis]

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
          f"above_0.25={( out > 0.25).sum()}")

# find sigmoid with shape matching 8400 detections
target_sigmoid = None
for i, sn in enumerate(sigmoid_nodes):
    out = outputs[i+1]
    # we need shape [1, 1, 8400] or [1, 8400]
    if 8400 in out.shape:
        target_sigmoid = sn
        target_scores  = outputs[i+1]
        print(f"\nFound matching sigmoid: {sn['output']}")
        print(f"Shape: {out.shape}")
        break

if target_sigmoid is None:
    print("\nNo sigmoid with 8400 detections found.")
    print("Trying to find via Concat node input...")

    # find the concat node that feeds into the final output
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

    # ── evaluate ──
    print("\nRunning evaluation on 500 images...")
    sess_fixed = ort.InferenceSession(
        str(fixed_path), providers=["CPUExecutionProvider"])

    CONF = 0.25
    IOU  = 0.45
    tp = fp = fn = 0

    img_files = sorted(list(img_dir.glob("*.jpg")))[:500]

    for idx, img_path in enumerate(img_files):
        img_orig = cv2.imread(str(img_path))
        orig_h, orig_w = img_orig.shape[:2]
        scale_x = orig_w / 640
        scale_y = orig_h / 640

        inp = cv2.resize(img_orig, (640,640))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        t   = inp.astype(np.float32)/255.0
        t   = np.transpose(t,(2,0,1))[np.newaxis]

        boxes_raw, scores_raw = sess_fixed.run(
            None, {sess_fixed.get_inputs()[0].name: t})

        # boxes: [1,4,8400] → [8400,4]
        boxes_raw = boxes_raw[0].T
        # scores: flatten to [8400]
        scores_raw = scores_raw.flatten()

        if len(scores_raw) != len(boxes_raw):
            # try to align
            scores_raw = scores_raw[:len(boxes_raw)]

        mask = scores_raw > CONF
        if not mask.any():
            lp = lbl_dir / f"{img_path.stem}.txt"
            if lp.exists():
                fn += sum(1 for _ in open(lp))
            continue

        boxes  = boxes_raw[mask]
        scores = scores_raw[mask]

        # xywh pixel → xyxy original coords
        cx,cy,w,h = boxes[:,0],boxes[:,1],\
                    boxes[:,2],boxes[:,3]
        x1=(cx-w/2)*scale_x; y1=(cy-h/2)*scale_y
        x2=(cx+w/2)*scale_x; y2=(cy+h/2)*scale_y
        pred_boxes = np.stack([x1,y1,x2,y2],axis=1)

        keep = cv2.dnn.NMSBoxes(
            pred_boxes.tolist(), scores.tolist(),
            CONF, IOU)
        if len(keep) > 0:
            pred_boxes = pred_boxes[keep.flatten()]
        else:
            pred_boxes = np.zeros((0,4))

        # load GT
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    v = line.strip().split()
                    if len(v) == 5:
                        _,cx,cy,w,h = map(float,v)
                        gt_boxes.append([
                            (cx-w/2)*orig_w,
                            (cy-h/2)*orig_h,
                            (cx+w/2)*orig_w,
                            (cy+h/2)*orig_h])

        matched = set()
        for pb in pred_boxes:
            hit = False
            for j,gb in enumerate(gt_boxes):
                if j in matched: continue
                xi1=max(pb[0],gb[0]); yi1=max(pb[1],gb[1])
                xi2=min(pb[2],gb[2]); yi2=min(pb[3],gb[3])
                inter=max(0,xi2-xi1)*max(0,yi2-yi1)
                a1=(pb[2]-pb[0])*(pb[3]-pb[1])
                a2=(gb[2]-gb[0])*(gb[3]-gb[1])
                iou_val=inter/(a1+a2-inter+1e-6)
                if iou_val >= 0.5:
                    hit=True; matched.add(j); break
            if hit: tp+=1
            else:   fp+=1
        fn += len(gt_boxes)-len(matched)

        if (idx+1) % 100 == 0:
            print(f"  {idx+1}/500 processed...")

    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

    # latency
    dummy = np.random.rand(1,3,640,640).astype(np.float32)
    inp_n = sess_fixed.get_inputs()[0].name
    for _ in range(10):
        sess_fixed.run(None, {inp_n: dummy})
    times = []
    for _ in range(50):
        s = time.perf_counter()
        sess_fixed.run(None, {inp_n: dummy})
        times.append((time.perf_counter()-s)*1000)
    lat = round(float(np.mean(times)), 2)

    print(f"\n{'='*55}")
    print(f"  INT8 FIXED MODEL RESULTS (500 images)")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'FP32':>12} {'INT8':>12}")
    print(f"  {'-'*45}")
    print(f"  {'Precision':<20} {'0.9818':>12} {prec:>12.4f}")
    print(f"  {'Recall':<20} {'0.8718':>12} {rec:>12.4f}")
    print(f"  {'F1 Score':<20} {'0.9235':>12} {f1:>12.4f}")
    print(f"  {'Size (MB)':<20} {'11.7':>12} {'3.23':>12}")
    print(f"  {'Latency (ms)':<20} {49.15:>12} {lat:>12.2f}")
    print(f"{'='*55}")
    if prec > 0:
        print(f"  Precision drop: {((0.9818-prec)/0.9818)*100:.2f}%")
        print(f"  Recall drop:    {((0.8718-rec)/0.8718)*100:.2f}%")
    print(f"  Size reduction: 72.4%")
    print(f"  Speed change:   {((lat-49.15)/49.15)*100:+.1f}%")