# CLAUDE.md

This file provides guidance to Claude Opus (claude.ai/code) when working with code in this repository.

## Project Overview

MatchVision is a Python-based automated football (soccer) match analysis system using Computer Vision and Deep Learning. It processes video footage to detect, track, and analyze players, referees, and the ball in real time.

**Primary stack**: Ultralytics (YOLOv5), OpenCV, PyTorch, Streamlit, ByteTrack (via Supervision library).

## Commands

```sh
# Install dependencies
pip install -r requirements.txt

# Run web interface (recommended)
streamlit run frontend/app.py

# Run CLI realtime mode
python realtime_main.py --source demos/demo1.mp4 --tracks players ball stats

# Run CLI offline processing
python main.py --video demos/demo1.mp4 --tracks players referees stats --verbose

# Performance optimization flags
python realtime_main.py --fp16 --imgsz 416 --frame-skip 2 --nvdec
python main.py --fp16 --imgsz 416 --frame-skip 2
```

Key CLI arguments: `--source` (webcam `0`, file, or URL stream), `--video` (input file), `--tracks` (`players`, `goalkeepers`, `referees`, `ball`, `stats`), `--verbose`, `--fp16` (FP16 half precision), `--imgsz` (YOLO input size, default 640), `--frame-skip` (process every Nth frame), `--nvdec` (NVDEC hardware decode), `--engine` (use TensorRT engine), `--onnx` (use ONNX model), `--model` (custom model path).

Model weights (`models/best.pt`, `models/best.engine`) are not committed and must be downloaded separately.

## Architecture

```
Video Input
    │
    ▼
┌──────────────────────┐
│ 1. OBJECT DETECTION   │  YOLOv5 via Ultralytics (4 classes: ball, goalkeeper, player, referee)
│    models/best.pt     │  Confidence: 0.15 general, 0.30 for ball
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. OBJECT TRACKING    │  ByteTrack via Supervision; unique IDs per object
│                       │  Goalkeeper merged into player category
└──────────┬───────────┘
           │
     ┌─────┴─────┬─────────────────┐
     ▼           ▼                 ▼
┌──────────┐ ┌───────────┐  ┌─────────────────┐
│3. CAMERA │ │4. TEAM    │  │5. BALL          │
│MOVEMENT  │ │ASSIGNMENT │  │POSSESSION       │
│Lucas-    │ │KMeans k=2 │  │Distance-based   │
│Kanade    │ │(color)    │  │(threshold 70px)│
└──────────┘ └───────────┘  └─────────────────┘
     └────────────┬────────────────┘
                  ▼
         ┌─────────────────┐
         │6. VISUALIZATION │
         │Annotations +    │
         │stats overlay    │
         └─────────────────┘
```

## Module Map

| Module | File | Algorithm |
|---|---|---|
| Detection + Tracking | `trackers/tracker.py` | YOLOv5 + ByteTrack + ball interpolation |
| Team Assignment | `team_assignment/team_assigner.py` | KMeans clustering on shirt color |
| Ball Possession | `player_ball_assignment/player_ball_assigner.py` | Euclidean distance |
| Camera Movement | `camera_movement/camera_movement.py` | Lucas-Kanade optical flow |
| Visualization | `utils/annotation_utils.py` | OpenCV drawing |

## Processing Modes

- **Offline** (`main.py`): Reads video into memory, batch processing, saves to `output/output.mp4`
- **Realtime** (`realtime_main.py`, `frontend/app.py`): Frame-by-frame from webcam, video file, or URL stream

## Key Conventions

- All detection/tracking/assignment modules have batch methods (for `main.py`) and single-frame methods (for `realtime_main.py`)
- Ball positions are interpolated using pandas linear interpolation to fill gaps
- Team assignment uses KMeans with k=2 on RGB color clusters; re-calibration is throttled to every 30 frames
- Device selection is automatic: CUDA > MPS > CPU (`utils/device_utils.py`)
- Model auto-detection priority: `.engine` (TensorRT) > `.onnx` > `.pt` (PyTorch)
- Ball confidence threshold: 0.30 for detection (both batch and single-frame modes)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/export_trt.py` | Export YOLO model to TensorRT engine (run on target GPU) |
| `scripts/export_onnx.py` | Export YOLO model to ONNX with optional INT8 quantization |
| `scripts/train_yolov8.py` | Train YOLOv8 model on Roboflow football dataset |
| `scripts/broadcast_server.py` | MJPEG streaming server with RTMP push support |
