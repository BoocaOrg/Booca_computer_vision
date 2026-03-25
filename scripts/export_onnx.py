#!/usr/bin/env python3
"""
ONNX Export Script with INT8 Quantization
Exports YOLO .pt model to ONNX format, with optional INT8 quantization.
Use INT8 for CPU or low-VRAM machines (~50-70% faster, 25% smaller model).

Usage:
    python scripts/export_onnx.py                          # Standard ONNX (FP32)
    python scripts/export_onnx.py --int8                   # INT8 quantized
    python scripts/export_onnx.py --imgsz 416 --int8       # Small + INT8
    python scripts/export_onnx.py --model models/best.pt   # Custom model
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to YOLO .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 quantization (faster but lower accuracy)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX model")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return

    print(f"Loading model: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)

    print(f"Exporting to ONNX...")
    print(f"  Image size: {args.imgsz}")
    print(f"  INT8 quantization: {args.int8}")

    export_kwargs = {"format": "onnx", "imgsz": args.imgsz, "simplify": args.simplify}
    if args.int8:
        export_kwargs["int8"] = True

    exported_path = model.export(**export_kwargs)
    print(f"Exported to: {exported_path}")

    # Report size
    pt_size = os.path.getsize(args.model) / (1024 * 1024)
    onnx_size = os.path.getsize(exported_path) / (1024 * 1024)
    print(f"  Original: {pt_size:.1f} MB -> Exported: {onnx_size:.1f} MB ({onnx_size/pt_size*100:.0f}%)")


if __name__ == "__main__":
    main()
