# YOLOv8 Training Guide

## Overview

This guide explains how to train YOLOv8 on the football players dataset using Roboflow.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended for training)
- Roboflow account and API key
- Dataset downloaded from Roboflow workspace

## Quick Start

### 1. Download Dataset from Roboflow

1. Go to your Roboflow workspace
2. Navigate to the football players dataset project
3. Click **Export Dataset**
4. Select format: **YOLO v5/v7**
5. Choose a version
6. Copy the download code (curl or Python)

### 2. Set Roboflow API Key

```sh
# Linux/Mac
export ROBOFLOW_API_KEY=your_api_key_here

# Windows (Command Prompt)
set ROBOFLOW_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:ROBOFLOW_API_KEY="your_api_key_here"
```

### 3. Train the Model

```sh
# Nano model (fastest, ~3.2M params) — good for realtime
python scripts/train_yolov8.py --size n --epochs 100

# Small model (better accuracy) — recommended
python scripts/train_yolov8.py --size s --epochs 150 --imgsz 640

# Medium model (high accuracy)
python scripts/train_yolov8.py --size m --epochs 100 --imgsz 640
```

### 4. Export to TensorRT

After training, export the best model to TensorRT for fastest inference:

```sh
python scripts/export_trt.py --model models/yolov8_football.pt --imgsz 640 --half
```

## Model Comparison

| Model | Params | Speed (A100) | Speed (RTX 3050 est.) | mAP@50 |
|-------|--------|-------------|----------------------|--------|
| YOLOv8n | 3.2M | 0.99ms | ~3-5ms | ~60% |
| YOLOv8s | 11.2M | 1.20ms | ~5-8ms | ~75% |
| YOLOv8m | 25.9M | 2.40ms | ~10-15ms | ~80% |
| YOLOv8l | 43.7M | 3.80ms | ~15-20ms | ~82% |
| YOLOv5n (current) | 1.9M | ~1ms | ~3-5ms | ~55% |
| YOLOv5s (current) | 7.5M | ~1.5ms | ~5-8ms | ~72% |

## Troubleshooting

### Out of Memory during training
Reduce batch size or image size:
```sh
python scripts/train_yolov8.py --size s --epochs 100 --batch-size 8 --imgsz 416
```

### Roboflow download fails
Check your API key and internet connection. Ensure the dataset version exists in your workspace.

### Training too slow
- Use a smaller model (`--size n` or `s`)
- Reduce image size (`--imgsz 416`)
- Use mixed precision: add `--amp` flag if supported

## Using Custom Trained Models

After training, update `trackers/tracker.py` or use CLI:

```sh
# Realtime with custom model
python realtime_main.py --source 0 --model models/yolov8_football.pt --fp16

# Offline with custom model
python main.py --video demos/demo1.mp4 --tracks players ball --model models/yolov8_football.pt
```
