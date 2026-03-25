#!/usr/bin/env python3
"""
YOLOv8 Training Script for Football Player Detection
Trains YOLOv8 on the Roboflow football dataset.

Prerequisites:
    1. Get your Roboflow API key from https://app.roboflow.com/settings/api
    2. Set environment variable: export ROBOFLOW_API_KEY=your_key
       (or set it in the code below)
    3. Download dataset from Roboflow workspace

Usage:
    python scripts/train_yolov8.py                          # YOLOv8n (fastest, smallest)
    python scripts/train_yolov8.py --size s --epochs 150    # YOLOv8s (better accuracy)
    python scripts/train_yolov8.py --imgsz 416              # Smaller input
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on football dataset")
    parser.add_argument("--size", type=str, default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n=nano (fastest), s=small, m=medium, l=large, x=xlarge")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (0, 1, ... or cpu)")
    parser.add_argument("--api-key", type=str, default="", help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("ERROR: Roboflow API key required.")
        print("  Set ROBOFLOW_API_KEY environment variable or pass --api-key")
        print("  Get your key at: https://app.roboflow.com/settings/api")
        return

    from ultralytics import YOLO

    model_name = f"yolov8{args.size}.pt"
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training...")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device: {args.device}")

    results = model.train(
        data="training/football-players-detection-1/data.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        project="training",
        name="yolov8_football",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True,
    )

    print(f"Training complete!")
    print(f"  Best model: training/yolov8_football/weights/best.pt")
    print(f"  Last model: training/yolov8_football/weights/last.pt")

    # Copy best model to models/
    import shutil
    best_path = "training/yolov8_football/weights/best.pt"
    if os.path.exists(best_path):
        shutil.copy(best_path, "models/yolov8_football.pt")
        print(f"  Copied to: models/yolov8_football.pt")


if __name__ == "__main__":
    main()
