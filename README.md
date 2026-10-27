# 🍋 Smart Lemon Harvester AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-powered lemon detection, ripeness classification, and quality grading for smart harvesting systems.**

---

## 📌 Overview

The **Smart Lemon Harvester AI** is an end-to-end computer vision and deep learning framework built to automate citrus harvesting operations. By combining real-time object detection with multi-task deep feature classification and spatial size estimation, the system determines which lemons are ready for picking, classifies commercial quality grades, and outputs actionable decisions for robotic manipulators and smart agricultural machinery.

---

## 🏗️ Architecture

The pipeline processes raw camera frames through a multi-stage vision pipeline to arrive at automated harvesting decisions:

```mermaid
graph TD
    A["📷 Camera / Video Stream"] --> B["🔍 YOLOv8 Detection Engine"]
    B -->|"Bounding Boxes & Confidence"| C["✂️ Crop & Normalize Regions of Interest"]
    C --> D["🧠 EfficientNet Multi-Task Classifier"]
    D --> E["🌱 Ripeness Head<br/>(Raw / Semi-Ripe / Ripe)"]
    D --> F["⭐ Quality Head<br/>(Grade A / Grade B / Grade C)"]
    B --> G["📏 Size & Area Estimator"]
    E --> H{"⚖️ Harvest Decision Engine"}
    F --> H
    G --> H
    H -->|"Ripe + Grade >= B + Size >= Medium"| I["✅ Harvest Command (Pick Target)"]
    H -->|"Not Ready / Low Grade"| J["⏳ Skip / Monitor Target"]
    H --> K["📊 Visual Annotation & Telemetry"]
```

---

## ✨ Features

- **High-Speed Object Detection:** YOLOv8-based lemon localization optimized for dense orchard foliage and varying outdoor sunlight conditions.
- **Multi-Task Classification:** Unified EfficientNet backbone with dual prediction heads:
  - **Ripeness State:** `raw` (green), `semi_ripe` (turning/yellow-green), and `ripe` (bright yellow).
  - **Quality Grading:** `grade_a` (export/premium), `grade_b` (domestic market), and `grade_c` (processing/juice).
- **Physical Size Estimation:** Bounding box pixel density analysis and reference metric scaling to prevent harvesting undersized fruit.
- **Configurable Harvest Logic:** Fully customizable harvest criteria via YAML configuration files (ripeness requirements, quality thresholds, and size cutoffs).
- **Versatile Inference Engine:** Compatible with static images, recorded MP4/AVI videos, RTSP IP camera streams, and USB webcams.
- **Production-Ready Structure:** Fully typed code, clean modular design, and export-ready configurations for edge deployment.

---

## 📂 Project Structure

```text
smart-lemon-harvester/
├── configs/
│   ├── detection_config.yaml       # YOLOv8 detection hyperparameters & paths
│   ├── classifier_config.yaml      # EfficientNet multi-task training setup
│   └── pipeline_config.yaml        # End-to-end inference & decision rules
├── data/
│   ├── detection/                  # YOLO-format detection datasets
│   └── classification/           # Cropped lemon ripeness & quality datasets
├── models/
│   ├── __init__.py                 # Model package interfaces
│   ├── detector.py                 # LemonDetector YOLOv8 wrapper
│   ├── classifier.py               # RipenessClassifier dual-head EfficientNet
│   └── pipeline.py                 # LemonHarvestPipeline decision orchestrator
├── utils/
│   ├── __init__.py                 # Utilities package interfaces
│   ├── visualization.py            # Bounding box & telemetry overlays
│   ├── preprocessing.py            # Image transformation & crop pipeline
│   └── metrics.py                  # Evaluation & confusion matrix metrics
├── scripts/
│   ├── train_detector.py           # Training entrypoint for YOLOv8 detector
│   ├── train_classifier.py         # Training entrypoint for multi-head classifier
│   └── infer.py                    # Multi-source inference script
├── weights/                        # Model checkpoints (.pt, .pth)
├── runs/                           # Experiment outputs, logs, and plots
├── tests/                          # Automated unit and integration tests
├── requirements.txt                # Project dependencies
├── setup.py                        # Package installation specification
├── .gitignore                      # Git exclusion rules
├── LICENSE                         # MIT License
└── README.md                       # Project documentation
```

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/yogesh/smart-lemon-harvester.git
cd smart-lemon-harvester
```

### 2. Create and Activate Virtual Environment
```bash
# On Linux / macOS
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Start

Run inference on sample images or video streams with the default pipeline configuration:

```bash
# Run pipeline on a sample orchard image
python scripts/infer.py --source data/samples/tree.jpg --config configs/pipeline_config.yaml --view

# Run real-time inference on a connected webcam
python scripts/infer.py --source 0 --config configs/pipeline_config.yaml --view

# Process a video file and save output with annotated decision telemetry
python scripts/infer.py --source data/samples/orchard_video.mp4 --save-path runs/output.mp4
```

---

## 🗂️ Dataset Setup

### 1. Detection Dataset (YOLO Format)
Organize labeled lemon detection images inside `data/detection/`:
```text
data/detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```
Ensure bounding box labels conform to normalized YOLO format (`class x_center y_center width height`).

### 2. Classification Dataset
Organize cropped lemon images for multi-attribute learning under `data/classification/`:
```text
data/classification/
├── train/
│   ├── ripe/
│   ├── semi_ripe/
│   └── raw/
├── val/
└── test/
```
*(Multi-task annotations can also be supplied via metadata CSV mapping image paths to both ripeness and quality labels).*

---

## 🏋️ Training

### Train Lemon Detector (YOLOv8)
```bash
python scripts/train_detector.py --config configs/detection_config.yaml
```
Checkpoints and training curves will be logged to `runs/detection/` with best weights stored at `weights/detection/best.pt`.

### Train Ripeness & Quality Classifier (EfficientNet)
```bash
python scripts/train_classifier.py --config configs/classifier_config.yaml
```
Evaluation metrics, confusion matrices, and best weights will be saved to `weights/classifier/best.pth`.

---

## 🔍 Inference Modes

| Source Type | Command Example |
| :--- | :--- |
| **Single Image** | `python scripts/infer.py --source data/samples/test1.jpg --save-path runs/test1_out.jpg` |
| **Directory of Images** | `python scripts/infer.py --source data/samples/ --save-dir runs/batch_predictions/` |
| **Recorded Video** | `python scripts/infer.py --source data/orchard.mp4 --save-path runs/annotated_orchard.mp4` |
| **Live USB Camera** | `python scripts/infer.py --source 0 --view` |
| **RTSP Network Stream** | `python scripts/infer.py --source rtsp://admin:pass@192.168.1.50:554/stream --view` |

---

## 📊 Benchmark Results

*(Placeholder benchmarks on validation split)*

| Task | Model Architecture | Metric | Score | Inference Speed (GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | YOLOv8n (640x640) | mAP@50 | *TBD (Target: >0.92)* | ~6.5 ms / frame (RTX 4090) |
| **Detection** | YOLOv8n (640x640) | mAP@50-95 | *TBD (Target: >0.78)* | ~6.5 ms / frame (RTX 4090) |
| **Ripeness** | EfficientNet-B0 (224x224) | Accuracy / F1 | *TBD (Target: >95%)* | ~3.2 ms / crop |
| **Quality** | EfficientNet-B0 (224x224) | Accuracy / F1 | *TBD (Target: >91%)* | ~3.2 ms / crop |

---

## 🗺️ Roadmap

- [ ] **Edge Deployment Optimization:** Export trained models to ONNX and NVIDIA TensorRT engines for real-time inference on NVIDIA Jetson Nano / Jetson Orin Nano.
- [ ] **Interactive Web Dashboard:** Build a modern FastAPI + React/Streamlit monitoring dashboard for orchard telemetry, yield estimates, and harvest heatmaps.
- [ ] **Robotic Manipulator Integration:** Integrate with ROS2 (Robot Operating System) and MoveIt for 3D coordinate estimation and automated end-effector trajectory planning.
- [ ] **Multi-Spectral & Depth Sensing:** Integrate RGB-D sensors (Intel RealSense / OAK-D) for millimeter-accurate distance estimation and occlusion handling.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 **Yogesh**. All rights reserved.
