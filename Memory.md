# MatchVision Memory

## Ball Possession O(1) Fix (viz-opt)
- File: `utils/annotation_utils.py`
- Added module-level counters `_team1_frames`, `_team2_frames` to replace O(n) slice + count per frame
- `ball_possession_box()` now increments counters incrementally instead of re-slicing from frame 0
- Counters reset at `frame_num == 0` to handle new video sessions
- Rendering (colors, positions, text) unchanged

## Tracker Optimization (tracker-opt)
- File: `trackers/tracker.py`
- Added `fp16=False` and `imgsz=640` parameters to `Tracker.__init__()`
- `self.fp16` is set to `fp16 and get_device() == "cuda"` (only enabled on CUDA when explicitly requested)
- Model is converted to FP16 with `self.model.half()` only when `self.fp16` is True
- Both `model.predict()` calls use `half=self.fp16` and `imgsz=self.imgsz`:
  - Batch method (`detect_frames`)
  - Single-frame method (`get_object_tracks_single_frame`)

## CLI Optimization Args (cli-opt)
- Files: `realtime_main.py`, `main.py`
- Added `--frame-skip`, `--fp16`, `--imgsz` CLI arguments to both entry points
- `main.py`: Added `_expand_tracks()` helper to expand sampled tracks back to full frame count by holding last known values; `process_video()` passes args to `Tracker` and conditionally samples frames
- `realtime_main.py`: Frame-skip skips YOLO detection every N frames, reuses `last_tracks` for visualization; `process_realtime()` passes args to `Tracker`

## Frontend Optimization Controls (viz-opt)
- File: `frontend/app.py`
- Added "Advanced / CLI" sidebar section with `use_fp16` checkbox and `imgsz` slider
- Both Tracker init calls (offline line ~145, realtime line ~575) now pass `fp16=use_fp16, imgsz=imgsz`

## Tier 3a: TensorRT Export & Auto-Load (trt-opt)
- File: `scripts/export_trt.py` (new)
  - Exports YOLO .pt to TensorRT .engine via Ultralytics `model.export(format="engine", imgsz=640, half=True)`
  - Run on target machine (RTX 3050) — engines are GPU-specific
  - Usage: `python scripts/export_trt.py --imgsz 416 --half`
- File: `trackers/tracker.py`
  - `Tracker.__init__` now auto-detects model priority: `.engine` (TensorRT) > `.onnx` > `.pt` (PyTorch)
  - Sets `self.model_type` ("tensorrt"/"onnx"/"pytorch") and `self.fp16` accordingly
  - If .engine exists, half precision is baked in at export time (fp16=False here)
  - Both `model.predict()` calls unchanged — Ultralytics auto-handles half for TRT
- Files: `realtime_main.py`, `main.py`
  - Added `--engine` flag (informational; Tracker auto-detects engine)
- File: `frontend/app.py`
  - Added model selectbox in "Advanced / CLI" section
  - Added caption explaining auto-load behavior

## Tier 3b: YOLOv8 Upgrade (yolov8-opt)
- File: `requirements.txt` — updated `ultralytics==8.2.31` to `ultralytics>=8.3.0` (supports both YOLOv5 and YOLOv8)
- File: `scripts/train_yolov8.py` (new) — YOLOv8 training script using Ultralytics `YOLO` API
  - Supports `--size` (n/s/m/l/x), `--epochs`, `--imgsz`, `--device`, `--api-key`
  - Copies best.pt to `models/yolov8_football.pt` after training
- File: `TRAINING_GUIDE.md` (new) — comprehensive guide covering dataset download, training commands, model comparison table, and troubleshooting
- Files: `realtime_main.py`, `main.py` — added `--model` argument to override model path; existing `--engine`/`--onnx` flags retained; path resolution logic prefers explicit `--model` > `--engine` > `--onnx` > default
- File: `frontend/app.py` — updated model selectbox with PyTorch/TensorRT/ONNX/Custom options; custom text input for arbitrary model path; all Tracker init calls now use `model_path` variable

## Tier 3c: ONNX/INT8 Export (onnx-opt)
- File: `scripts/export_onnx.py` (new)
  - Exports YOLO .pt to ONNX via `ultralytics.YOLO(model).export(format="onnx")`
  - `--int8` flag enables INT8 quantization (~50-70% faster, ~25% smaller model)
  - `--imgsz` (default 640), `--simplify` (default True); reports size reduction
  - Run on target machine: `python scripts/export_onnx.py --int8 --imgsz 416`
- File: `trackers/tracker.py` — added `import os`; `__init__` uses `os.path.splitext()` for clean extension detection
- Files: `realtime_main.py`, `main.py` — added `--onnx` and `--model` CLI args; `model_path` resolution: explicit `--model` > `--engine` (best.engine) > `--onnx` (best.onnx) > default (best.pt); `process_realtime()`/`process_video()` accept `model_path` parameter
- File: `frontend/app.py` — model selectbox already has ONNX option (from yolov8-opt); fixed second Tracker init (realtime path) to use `model_path` instead of hardcoded `"models/best.pt"`

## Tier 4a: NVDEC Hardware Decode (nvdec-opt)
- Files: `frontend/app.py`, `realtime_main.py`, `requirements.txt`
- Added NVDEC hardware decode toggle for NVIDIA GPUs via OpenCV's ffmpeg environment variable
- `frontend/app.py`: Added "Hardware Acceleration" sidebar section with `use_nvdec` checkbox (default True); env var `OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;cuda|video_codec;h264_cuvid` set before VideoCapture in realtime mode
- `realtime_main.py`: Added `--nvdec` CLI flag; `process_realtime()` accepts `nvdec` param and sets env var before VideoCapture; print statement updated to show nvdec status
- `requirements.txt`: Added comment documenting NVDEC requirements (NVIDIA GPU + ffmpeg built with CUDA support; no new packages needed)

## Tier 4b: UI/UX + Broadcast (FULL IMPLEMENTATION)
- Architecture: analysis loop → BroadcastPusher (HTTP POST) → broadcast_server.py (MJPEG + RTMP)
- `frontend/app.py` UI improvements:
  - Embedded MJPEG server thread on port 8503 serves frames to HTML iframe
  - HTML iframe video player: native fullscreen support (F11 or browser button)
  - Fullscreen button opens stream in new tab
  - Layout restructured: header (status + fullscreen btn) → video iframe → sidebar zoom slider → debug area
  - FPS counter in status bar
  - Zoom applied before MJPEG encoding
  - Removed all debug_expander wrapping (debug now simple inline)
- `scripts/broadcast_server.py` (rewritten):
  - `BroadcastServer` class with `push_frame()` method
  - POST `/push` endpoint receives base64 JPEG frames from frontend/CLI
  - MJPEG `/mjpeg` endpoint for browser viewing
  - RTMP pusher thread with lazy writer init and FPS throttling
  - `/health` endpoint for status check
- `scripts/push_frame.py` (rewritten):
  - `BroadcastPusher` class using `requests` for non-blocking HTTP POST
  - Thread-based async sending, fire-and-forget with auto-reconnect
  - Local queue for same-process MJPEG consumers
- `frontend/app.py`:
  - `BroadcastPusher` imported at top (with `_HAS_PUSHER` guard)
  - Pusher initialized in realtime loop when `enable_broadcast=True`
  - Every annotated frame pushed via `pusher.push(output_frame)` before display
  - Sidebar shows active status + RTMP URL when enabled
- `realtime_main.py`:
  - Added `--broadcast` CLI flag
  - `process_realtime()` accepts `broadcast` param
  - Pusher init with print confirmation; frames pushed in loop

## Bug Fixes (Comprehensive Code Review)
- `scripts/export_trt.py`: Fixed `--half` argument — removed misleading `default=True` (ignored by argparse `store_true`). Now uses `action="store_true"` correctly.
- `trackers/tracker.py`: Added ball confidence threshold `>= 0.3` in `get_object_tracks_single_frame()` to match batch mode, preventing false positive ball detections in realtime.
- `trackers/tracker.py`: Removed dead code `batch_size=batch_size` in `detect_frames()`.
- `camera_movement/camera_movement.py`: Fixed wrong dimension `frame.shape[0]` → `first_frame_grayscale.shape[1]` for right-edge mask. Removed duplicate grayscale conversion. Fixed mask now covers left edge + right edge correctly.
- `player_ball_assignment/player_ball_assigner.py`: Fixed `get_player_and_possession()` — added KeyError guard for missing ball detection; handles empty ball tracks gracefully; handles empty ball_possession list with `np.array([])`.
- `frontend/app.py`: Fixed missing `force=True` in team calibration call (was inconsistent with realtime_main.py).
- `frontend/app.py`: Fixed NVDEC env var placement — now set before all VideoCapture calls (not overwritten by m3u8 setting).
- `frontend/app.py`: Fixed `resolution_scale` display — realtime mode now rescales annotated output to max 1280px before display, while inference still uses original resolution.

## Team Assignment Optimization (team-opt)
- File: `team_assignment/team_assigner.py`
- Added `frame_counter` and `kmeans_throttle=30` to `__init__`
- Added `force` param to `assign_team_colour()` with throttle condition: KMeans re-calibration only runs every `kmeans_throttle` frames (unless `force=True`)
- Frame counter increments in `get_player_team()`
- File: `realtime_main.py`
- Calibration call now uses `force=True` to ensure first frame always runs KMeans

## Booca.online Livestream/VOD Integration (booca-stream)
- File: `frontend/app.py`
- Added `resolve_booca_url()` function: Resolves Booca URLs to direct m3u8 streams via Booca API
  - API endpoint: `https://api.booca.online/api/streams/{stream_id}`
  - Supports: `/livestream/watch/{id}` (live) and `/livestream/vod/{id}` (VOD)
  - Live streams → `data.playbackUrls.hls` (HLS m3u8 from stream.booca.online)
  - VOD recordings → `data.vod.url` (HLS m3u8 from Bunny CDN)
  - Returns stream URL + metadata (title, status, category, user, stats, thumbnail)
- **Realtime mode**: Added "Booca Stream" source type with dedicated UI
  - Stream info card with theme support (dark/light) showing title, status badge, user, views/likes
  - Thumbnail preview when available
  - Duration info for VOD
  - Auto-resolves URL via API and passes m3u8 to cv2.VideoCapture
- **Offline mode**: Added "Booca VOD" source type
  - Downloads all frames from HLS stream via cv2.VideoCapture
  - Progress bar showing download progress
  - Then feeds frames into existing offline analysis pipeline
- **resolve_stream_url()**: Added Method 0a — Booca detection before other methods
  - Detects `booca.online/livestream/` or `booca.vn/livestream/` URLs in generic URL input
  - Falls through to other methods if Booca resolution fails
- URL Stream documentation updated with Booca examples
- Instructions section updated with Booca quick-start guide
