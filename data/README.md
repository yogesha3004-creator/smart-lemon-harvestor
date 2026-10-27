# Dataset Setup

This directory stores datasets for training the lemon detection and classification models.

## Quick Setup

Run the download script to fetch public datasets:

```bash
python data/download_dataset.py --dataset kaggle-lemon-quality
python data/prepare_dataset.py --task both
```

## Supported Datasets

### 1. Kaggle Lemon Quality Dataset
- **Source**: [Lemon Quality Dataset on Kaggle](https://www.kaggle.com/datasets/yusufemir/lemon-quality-dataset)
- **Size**: ~2,500 images
- **Classes**: good_quality, bad_quality, empty_background
- **Usage**: Ripeness & quality classification training

### 2. Custom Detection Dataset (Roboflow)
- **Format**: YOLOv8 (images + labels in YOLO txt format)
- **Usage**: Lemon detection training

## Directory Structure After Preparation

```
data/
├── raw/                    # Original downloaded data
│   └── lemon-quality/
├── detection/              # YOLOv8 format for detection
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── classification/         # ImageFolder format for classifier
    ├── train/
    │   ├── raw/
    │   ├── semi_ripe/
    │   └── ripe/
    ├── val/
    │   ├── raw/
    │   ├── semi_ripe/
    │   └── ripe/
    └── test/
        ├── raw/
        ├── semi_ripe/
        └── ripe/
```

## Using Your Own Dataset

1. Place detection images in `data/detection/images/` and YOLO labels in `data/detection/labels/`
2. Place classification images in `data/classification/` following the ImageFolder structure
3. Run `python data/prepare_dataset.py --task split` to create train/val/test splits
