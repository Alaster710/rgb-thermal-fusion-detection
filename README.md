# RGB-Thermal Fusion for Object Detection on Edge Hardware

MSc Data Analytics Capstone Project — University of Galway, 2026

## Project Overview
In this research work, comparison of three fusion approaches, which include early, intermediate, and late fusion approaches, is done on the lightweight YOLOv8n backbone and evaluated under edge deployment conditions.

## Research Question
Does early fusion (RGB + thermal at feature level) outperform late fusion (separate detection + fusion) for object detection. Fusion architectures for edge deployment.

## Key Results
| Model | LLVIP mAP50 | FLIR mAP50 | Size (MB) | Latency (ms) |
|---|---|---|---|---|
| RGB Baseline | 0.885 | 0.524 | 11.7 | 25.0 |
| Thermal Baseline | 0.958 | 0.657 | 11.7 | 26.0 |
| Early Fusion | 0.882 | 0.518 | 11.7 | 27.7 |
| Intermediate Fusion | 0.889 | 0.502 | 11.7 | 27.7 |
| Late Fusion | **0.962** | **0.690** | 23.4 | 50.8 |

## Datasets
- **LLVIP** — 15,488 aligned visible-infrared pairs, single class (person), nighttime only
- **FLIR ADAS Aligned** — 5,142 pairs, three classes (person, car, bicycle), day and night

## Project Structure
rgb-thermal-fusion-detection/
├── src/
│ ├── prepare_datasets.py # Dataset conversion to YOLO format
│ ├── fix_labels.py # Labels folder structure fix
│ ├── train_baselines.py # Single modality baseline training
│ ├── train_early_fusion.py # Early fusion model training
│ ├── train_intermediate_fusion.py # Intermediate fusion training
│ ├── train_late_fusion.py # Late fusion WBF evaluation
│ ├── edge_benchmark.py # ONNX export and benchmarking
│ ├── analyse_results.py # Results figures and tables
│ ├── demo.py # Qualitative comparison demo
│ ├── quantize_simple.py # INT8 quantization export
│ └── fix_int8_v4.py # INT8 validation script
├── data/ # Datasets and YAML configs
├── models/ # Trained weights and ONNX exports
└── results/ # Figures and benchmark results


## Setup and Installation

### Requirements
- Python 3.11
- CUDA-capable GPU (tested on NVIDIA RTX 2050)

### Install dependencies
```bash
pip install ultralytics torch torchvision onnx onnxruntime 
pip install thop ensemble-boxes torchmetrics pycocotools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Dataset Preparation
1. Download LLVIP from Google Drive (contact bupt-ai-cz@gmail.com)
2. Download FLIR ADAS Aligned from HuggingFace
3. Run dataset preparation:
```bash
python src/prepare_datasets.py
python src/fix_labels.py
```

### Training
```bash
# Train baselines
python src/train_baselines.py

# Train fusion models
python src/train_early_fusion.py
python src/train_intermediate_fusion.py
python src/train_late_fusion.py
```

### Benchmarking
```bash
python src/edge_benchmark.py
python src/analyse_results.py
```

## Key Findings
1. Late fusion achieves highest accuracy, outperforming early fusion 
   by 9.1% on LLVIP and 33.2% on FLIR
2. Early fusion offers best accuracy-efficiency trade-off — identical 
   size and FLOPs to single modality models
3. Single-model fusion architectures are more robust to complete RGB 
   modality failure than late fusion in pitch-dark conditions
4. INT8 quantization reduces late fusion size by 72.4% with only 
   0.93% F1 drop

## Environment
- Python 3.11.9
- PyTorch 2.5.1+cu121
- Ultralytics 8.4.75
- ONNX Runtime 1.27.0
- Hardware: NVIDIA GeForce RTX 2050 (4GB VRAM)

## Author
Alaster Joy Cheeramkuzhyil
MSc Data Analytics, University of Galway


