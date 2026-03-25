 
MatchVision
Performance Optimization Report
GPU: NVIDIA RTX 3050 Laptop · VRAM 4GB · Ampere

Báo cáo này tổng hợp các giải pháp tăng hiệu năng cho hệ thống phân tích bóng đá MatchVision, được tối ưu riêng cho GPU RTX 3050 Laptop. Mục tiêu đưa hệ thống từ baseline ~15 FPS lên 45–55 FPS ổn định cho cả video offline lẫn camera livestream.

Tổng quan các Tier tối ưu

Tier & Mục tiêu	Thời gian	FPS gain
Tier 1 — Quick wins (FP16, frame skip, resize)	1–3 ngày	~2× baseline
Tier 2 — Pipeline parallelism (threading, KMeans opt)	1–2 tuần	+30–50% thêm
Tier 3 — TensorRT + model upgrade	2–4 tuần	~3–5× baseline
Tier 4 — NVDEC + DeepStream (production livestream)	Dài hạn	Edge-grade


Tier 1 — Quick Wins
Impact cao, ít refactor. Thực hiện được trong 1–3 ngày làm việc.

Frame Skipping
Chạy YOLO detection mỗi 2–3 frame thay vì mỗi frame. ByteTrack vẫn cập nhật đủ mọi frame bằng các bounding box đã track — cầu thủ không biến mất trong 2–3 frame. Football realtime không cần detect 30 fps.
Input Resolution Resize
Resize ảnh gốc xuống 640px hoặc 416px trước khi đưa vào YOLO. Inference time giảm gần tỉ lệ bình phương với resolution. Với football, detect ở 416px vẫn đủ chính xác vì cầu thủ không quá nhỏ trong khung hình.
FP16 Half Precision
Nếu đang dùng CUDA (RTX 3050 hỗ trợ Tensor Core FP16 thật sự), thêm một dòng code để có ngay 1.5–2× throughput:
model = YOLO('models/best.pt')
model.half()  # Enable FP16 on CUDA
VRAM sử dụng giảm ~40%, cho phép batch size lớn hơn trên 4GB VRAM.


Tier 2 — Pipeline Parallelism
Tách các bước xử lý tuần tự thành pipeline song song, GPU không bị idle chờ CPU.

Async 3-Thread Pipeline
Hiện tại detect → track → team assign → render chạy tuần tự. Tách ra 3 thread với queue:
•	Thread 1: Decode frame từ video/RTSP stream vào buffer
•	Thread 2: YOLO inference + ByteTrack (GPU)
•	Thread 3: Team assign + render + display (CPU)
GPU không bị idle chờ CPU decode nữa. Đây là nơi gain nhiều nhất cho realtime stream và RTSP camera.

KMeans Team Assignment Throttling
KMeans không cần chạy mỗi frame — màu áo cầu thủ không thay đổi. Chạy mỗi 30–60 frame là đủ:
# Trong team_assigner.py
self.frame_counter = 0
if self.frame_counter % 30 == 0:
    self.run_kmeans(players)  # Chỉ chạy mỗi 30 frame
self.frame_counter += 1
Giải phóng CPU đáng kể, đặc biệt khi số cầu thủ trên sân lớn.


Tier 3 — Model Optimization
Thay model hoặc export sang runtime nhanh hơn. Gain lớn nhất trong toàn bộ roadmap.

TensorRT Engine Export
Export YOLOv5/v8 sang TensorRT engine — game changer cho GPU NVIDIA. Engine được tối ưu cho RTX 3050 cụ thể, gain 3–5× so với PyTorch baseline:
yolo export model=models/best.pt format=engine half=True imgsz=640
Lưu ý quan trọng: TRT engine bị lock vào GPU đó, không share được. Cần chạy export 1 lần trên máy mục tiêu.

Model Upgrade: YOLOv5 → YOLOv8/v9
API Ultralytics tương thích, upgrade gần như không cần refactor. YOLOv8/v9 có accuracy tốt hơn và speed nhanh hơn YOLOv5 ở cùng resolution.

ONNX + INT8 Quantization
Dành cho máy không có GPU mạnh: giảm model size xuống 25%, inference nhanh hơn 50–70%. Accuracy mất rất ít với football detection vì confidence threshold 0.15 đã rất thấp.
yolo export model=best.pt format=onnx int8=True


Tier 4 — Production Livestream
Dành cho triển khai thực tế với RTSP camera hoặc broadcast. Đầu tư thời gian dài hơn.

•	NVDEC hardware decode: Thay ffmpeg CPU decode bằng NVDEC, tiết kiệm 20–30% CPU chỉ riêng bước đọc stream RTSP
•	NVIDIA DeepStream: Pipeline hoàn chỉnh cho multi-camera, built-in tracker và analytics
•	WebRTC output: Cho phép stream kết quả phân tích trực tiếp lên browser hoặc broadcast platform


Tương thích RTX 3050 Laptop — Ưu tiên thực tế

Ưu tiên	Thay đổi cụ thể	FPS estimate
#1 — model.half()	Thêm model.half() sau khi load model	~25 FPS
#2 — Input resize	imgsz=416 trong realtime_main.py	~30 FPS
#3 — TensorRT export	yolo export model=best.pt format=engine half=True	~45–55 FPS
#4 — KMeans throttle	Chạy team assign mỗi 30 frame, thêm frame_counter	CPU -30%

Lưu ý về giới hạn phần cứng:
•	VRAM 4GB: Giữ imgsz ≤ 640, batch=1 cho realtime. Batch lớn sẽ OOM
•	Thermal throttle: Laptop GPU giảm clock sau 30+ phút tải cao. Cân nhắc FPS cap ở 30 thay vì full throttle
•	TRT engine: Phải chạy export trên chính máy RTX 3050, không dùng được trên GPU khác
•	FP16 Tensor Core: RTX 3050 Ampere hỗ trợ thật sự (không phải emulation), đây là ưu điểm lớn nhất của GPU này

Kết quả kỳ vọng:
Baseline: ~15 FPS  →  Sau Tier 1+2: ~30 FPS  →  Sau TRT export: ~45–55 FPS
Ổn định đủ cho cả video offline lẫn camera livestream với RTSP.
