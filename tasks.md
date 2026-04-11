# MatchVision Tasks

## Completed
- [x] tracker-opt: FP16 + imgsz optimization in trackers/tracker.py
- [x] team-opt: KMeans frame-counter throttle in team_assignment/team_assigner.py + realtime_main.py
- [x] viz-opt: Ball possession O(n) -> O(1) in utils/annotation_utils.py; FP16/imgsz controls added to frontend/app.py
- [x] cli-opt: Added --frame-skip, --fp16, --imgsz to realtime_main.py and main.py; Tracker.__init__ now accepts fp16/imgsz params
- [x] trt-opt: TensorRT engine export via scripts/export_trt.py; Tracker auto-detects .engine/.onnx/.pt; --engine flag added to CLI; model selectbox added to frontend
- [x] yolov8-opt: YOLOv8 training script (scripts/train_yolov8.py); TRAINING_GUIDE.md; --model CLI override for both entry points; custom model selection in frontend/app.py
- [x] onnx-opt: ONNX export script with INT8 quantization (scripts/export_onnx.py); --int8, --imgsz, --simplify args; ONNX auto-load in Tracker.__init__ via os.path.splitext; --onnx CLI arg for both entry points; model_path variable used consistently in both Tracker inits in frontend/app.py
- [x] nvdec-opt: NVDEC hardware decode toggle in frontend/app.py (sidebar checkbox, default True); --nvdec CLI flag in realtime_main.py; OPENCV_FFMPEG_CAPTURE_OPTIONS env var set before VideoCapture; comment added to requirements.txt
- [x] ui-fix: Replaced deprecated `use_column_width=True` with `width=960` in st.image(); fixed broadcast CORS error by showing clear server-start instructions instead of auto-linking
- [x] bugfix-comprehensive: Fixed 11 bugs from code review — export_trt.py --half default, ball confidence threshold missing in single-frame mode, wrong camera mask dimension, KeyError guard in ball possession, missing force=True in frontend team assign, duplicate grayscale conversion, dead code batch_size=, NVDEC env var overwrite
- [x] broadcast-full: Full broadcast implementation — rewritten broadcast_server.py (BroadcastServer class + POST /push + MJPEG + RTMP), rewritten push_frame.py (HTTP POST BroadcastPusher), integrated into frontend/app.py realtime loop + realtime_main.py CLI with --broadcast flag
- [x] ui-fullscreen: Embedded MJPEG server thread (port 8503) in app.py for HTML iframe video player with native fullscreen; restructured layout (header → iframe → sidebar zoom → debug); FPS counter; zoom applied before encoding
- [x] booca-stream: Booca.online livestream/VOD integration — resolve_booca_url() via API (api.booca.online/api/streams/{id}); "Booca Stream" source in realtime mode with info card + thumbnail; "Booca VOD" source in offline mode with HLS frame download; auto-detect Booca URLs in generic URL Stream input; dark/light theme support
