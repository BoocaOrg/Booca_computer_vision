#!/usr/bin/env python3
"""
TensorRT Engine Export Script
Exports YOLO .pt model to TensorRT .engine for optimized inference on NVIDIA GPUs.
Must be run on the TARGET machine (RTX 3050) — engines are GPU-specific.

Usage:
    python scripts/export_trt.py                          # Default: imgsz=640, FP16
    python scripts/export_trt.py --imgsz 416 --half       # Smaller/faster
    python scripts/export_trt.py --model models/best.pt   # Custom model
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT engine")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (416=fast, 640=balanced, 1280=accurate)")
    parser.add_argument("--half", action="store_true", help="Use FP16 half precision (default: enabled)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print(f"Please download the model first or check the path.")
        return

    print(f"Loading model: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    print(f"Exporting to TensorRT engine...")
    print(f"  Image size: {args.imgsz}")
    print(f"  Half precision: {args.half}")
    print(f"  Note: This export may take 5-15 minutes on first run (building CUDA kernels)")

    exported_path = model.export(format="engine", imgsz=args.imgsz, half=args.half)
    print(f"Exported to: {exported_path}")
    print(f"  Engine file is GPU-specific — cannot be shared across different GPUs.")

if __name__ == "__main__":
    main()
