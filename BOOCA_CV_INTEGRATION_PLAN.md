# 🎯 BOOCA × MatchVision — Computer Vision Integration Plan
> **Mục tiêu**: Tích hợp các tính năng Computer Vision từ `football-computer-vision` vào phần **Livestream** của BOOCA để tạo ra trải nghiệm xem bóng đá thông minh, tự động phân tích ngay trên nền tảng.

---

## 📋 Mục Lục

1. [Tổng Quan Hiện Tại](#1-tổng-quan-hiện-tại)
2. [Kiến Trúc Tích Hợp](#2-kiến-trúc-tích-hợp)
3. [Các Tính Năng CV Nên Thêm](#3-các-tính-năng-cv-nên-thêm)
4. [Implementation Phases](#4-implementation-phases)
5. [API Design](#5-api-design)
6. [Frontend Components](#6-frontend-components)
7. [Tech Stack & Dependencies](#7-tech-stack--dependencies)
8. [Timeline](#8-timeline)

---

## 1. Tổng Quan Hiện Tại

### 🎥 BOOCA Livestream (Hiện tại)

| Layer | Tech | Chức năng |
|-------|------|-----------|
| **Frontend** | Next.js + MUI | StreamPlayer (HLS/FLV), ChatBox, FloatingReactions, StreamCard |
| **Backend** | Node.js/Express | stream.controller, ownerLivestream.controller, streamSession.controller |
| **Streaming** | SRS Server | SRT ingest (OBS → SRS), HLS/FLV playback, VOD xử lý |
| **Database** | MongoDB | Stream model, StreamSession model |

### 🤖 MatchVision (football-computer-vision)

| Module | Công nghệ | Output |
|--------|-----------|--------|
| Object Detection | YOLOv5 custom | Bounding boxes: player/ball/referee |
| Object Tracking | ByteTrack | Player IDs duy nhất qua frame |
| Team Classification | KMeans | Tự động phân biệt 2 đội qua màu áo |
| Camera Movement | Lucas-Kanade Optical Flow | Bù trừ chuyển động camera |
| Ball Interpolation | Pandas | Trajectory bóng mượt mà |
| Ball Possession | Distance Algorithm | % kiểm soát bóng real-time |

### 🔗 Điểm Kết Nối Đã Có

> Memory.md ghi nhận: **`booca-stream` branch** đã resolve Booca HLS URL → CV pipeline  
> `frontend/app.py` → `resolve_booca_url()` → `api.booca.online/api/streams/{id}` → `playbackUrls.hls`

---

## 2. Kiến Trúc Tích Hợp

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BOOCA PLATFORM                               │
│                                                                       │
│  OBS/Camera ──SRT──► SRS Server ──HLS──► StreamPlayer (FE)          │
│                           │                       │                   │
│                           │                  ChatBox                  │
│                           │              FloatingReactions            │
│                           ▼                                           │
│              ┌────────────────────────┐                              │
│              │   CV Analysis Service  │ ◄─── NEW                     │
│              │   (Python FastAPI)     │                              │
│              │                        │                              │
│              │  ┌──────────────────┐  │                              │
│              │  │  YOLOv5 + ByteT │  │                              │
│              │  │  KMeans Teams   │  │                              │
│              │  │  Optical Flow   │  │                              │
│              │  │  Possession Alg │  │                              │
│              │  └──────────────┬──┘  │                              │
│              └─────────────────│──────┘                              │
│                                ▼                                      │
│              ┌────────────────────────┐                              │
│              │   CV Stats WebSocket   │ ──► CVOverlay (FE)  ◄── NEW │
│              │   / REST API           │     LiveStatsPanel           │
│              └────────────────────────┘     PlayerHeatmap            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Các Tính Năng CV Nên Thêm

### 🥇 PHASE 1 — Real-time Statistics (Ưu tiên cao)

#### 3.1 Ball Possession Widget
**Mô tả**: Thanh % kiểm soát bóng của 2 đội, cập nhật real-time trong livestream.

```
┌─────────────────────────────────────────┐
│  ⚽ KIỂM SOÁT BÓNG                     │
│                                          │
│  Đội Đỏ  ████████████░░░░░░   58%      │
│  Đội Xanh ░░░░░░░░░░░░████   42%      │
│                                          │
│  Cập nhật: frame 1247 / 1:23:45        │
└─────────────────────────────────────────┘
```

**CV Module dùng**: `PlayerBallAssigner` → distance algorithm  
**Backend**: WebSocket emit `cv:possession:update`  
**Frontend**: Component `PossessionBar.tsx`

---

#### 3.2 Player Count & Team Detection
**Mô tả**: Hiển thị số cầu thủ mỗi đội đang xuất hiện trên sân.

```
│ 👕 Đội A: 11 cầu thủ  │  👕 Đội B: 10 cầu thủ  │
```

**CV Module**: YOLOv5 detection + KMeans team assignment  
**Emit**: `cv:players:count`

---

#### 3.3 Live Event Detection — Auto Highlight
**Mô tả**: Tự động phát hiện các sự kiện đáng chú ý để push notification và tạo highlight clip.

| Sự kiện | Cách detect |
|---------|-----------|
| 🥅 Bàn thắng | Ball crosses goal line (vị trí bóng + bounding box khung thành) |
| 🔴 Thẻ đỏ/vàng | Referee + kéo arm gesture (pose estimation) |
| 🤝 Corner / Free kick | Ball out of bounds + game pause detection |

**Backend**: Emit `cv:event:detected` → push notification → NFY service  
**Frontend**: Toast notification + Auto-clip bookmark

---

### 🥈 PHASE 2 — Player Analytics (Ưu tiên trung bình)

#### 3.4 Player Heatmap
**Mô tả**: Bản đồ nhiệt vị trí di chuyển của từng cầu thủ trong trận, hiển thị khi VOD kết thúc.

```
┌─────────────────────────────────────────┐
│        🔥 HEATMAP - Cầu thủ #7         │
│                                          │
│   ╔══════════════════════╗              │
│   ║░░░░░░░░░░░░░░░░░░░░  ║              │
│   ║░░░████████░░░░░░░░░  ║              │
│   ║░░░████████░░░░░░░░░  ║              │
│   ║░░░░░░░░░░░░░░░░░░░░  ║              │
│   ╚══════════════════════╝              │
│   Nhiều nhất: Cánh phải 68%             │
└─────────────────────────────────────────┘
```

**Kỹ thuật**: Tích lũy `position_adjusted` theo player ID → Gaussian blur  
**Khi dùng**: Click vào player card trong VOD replay  
**Frontend**: `PlayerHeatmapModal.tsx` dùng canvas/SVG overlay

---

#### 3.5 Speed Estimation (Tốc độ cầu thủ)
**Mô tả**: Ước lượng tốc độ di chuyển của cầu thủ dựa trên pixel displacement + camera calibration.

```
│  Cầu thủ #10  💨 27.3 km/h  (Sprint!) │
```

**Kỹ thuật**: 
1. `position_adjusted` giữa 2 frame liên tiếp → pixel distance
2. Calibration: 1 pixel ≈ X mét (dựa trên kích thước chuẩn sân)
3. `speed = pixel_distance × scale / Δtime`

**Emit**: `cv:player:speed` → filter top 5 players  
**Frontend**: Speed leaderboard trong Creator Dashboard

---

#### 3.6 Pass Detection & Formation Analysis
**Mô tả**: Phát hiện đường chuyền bóng và đội hình chiến thuật.

```
Formation detected: 4-3-3 → 4-4-2 (phút 67)
Pass network: Cầu thủ #8 → #10 → #9 (5 lần)
```

**Kỹ thuật**: 
- Ball possession change = pass event
- Cluster player positions → formation mapping

---

### 🥉 PHASE 3 — AI-Enhanced Viewer Experience (Ưu tiên bonus)

#### 3.7 Auto-Clip Generator
**Mô tả**: Khi detect sự kiện quan trọng (bàn thắng, cơ hội nguy hiểm), tự động cắt clip ±30s và lưu vào VOD highlights.

**Flow**:
```
CV detects event → timestamp saved → 
VOD post-process → extract clip → 
save to Bunny CDN → push to highlights feed
```

**Backend**: `POST /api/cv/highlights` → trigger FFmpeg clip extract  
**Frontend**: Highlights tab trong StreamPlayer VOD section

---

#### 3.8 CV Overlay Toggle (Viewer Control)
**Mô tả**: Người xem có thể bật/tắt các lớp overlay CV ngay trên video player.

```
⚙️ CV Settings
[✓] Hiển thị bounding box cầu thủ
[✓] Bảng kiểm soát bóng
[ ] Heatmap real-time
[ ] Tốc độ cầu thủ
```

**Frontend**: Thêm vào Settings Menu trong `StreamPlayer.tsx`

---

#### 3.9 Automated Commentary (AI Text)
**Mô tả**: Tự động tạo bình luận text bằng AI dựa trên CV events, hiển thị trong ChatBox.

```
🤖 [CV Bot]: Cầu thủ #9 đang cầm bóng, sprint 
              tốc độ 29 km/h về phía vòng cấm!

🤖 [CV Bot]: Đội Đỏ đang kiểm soát bóng 65% 
              trong 5 phút vừa rồi.
```

**Backend**: CV event → format text → emit `chat:cv_bot:message`  
**Frontend**: Hiển thị trong `ChatBox.tsx` với avatar bot riêng

---

#### 3.10 Pre-Match Player Identification (Roster Sync)
**Mô tả**: Link player ID từ tracking với tên cầu thủ thực tế từ lineup roster.

**Cách hoạt động**:
1. Chủ stream nhập lineup (số áo + tên) trước khi stream
2. CV detect số áo bằng OCR (PaddleOCR / EasyOCR)
3. Map tracking ID → player name

**Backend**: `POST /api/cv/roster` → lưu mapping  
**Frontend**: Roster input form trong `CreatorDashboardView.tsx`

---

## 4. Implementation Phases

```
Phase 1 (Weeks 1-3) — Core Stats Integration
├── [BE] CV Analysis Service (FastAPI, Python)
│   ├── Wrap existing MatchVision pipeline
│   ├── WebSocket endpoint `ws://cv-service/analysis`
│   └── Docker container
├── [BE] BOOCA Backend proxy endpoints
│   ├── POST /api/cv/start-analysis
│   ├── GET  /api/cv/stats/:streamId
│   └── WS  /ws/cv/:streamId (forward)
├── [FE] PossessionBar component
├── [FE] LiveStatsPanel component
└── [FE] Player count display

Phase 2 (Weeks 4-6) — Player Analytics
├── [CV] Speed estimation module
├── [CV] Player heatmap accumulator
├── [BE] POST /api/cv/heatmap/:streamId
├── [FE] PlayerHeatmapModal
├── [FE] Speed leaderboard widget
└── [FE] Event notification toast

Phase 3 (Weeks 7-9) — AI Experience
├── [CV] Auto event detection (goal, card)
├── [BE] Highlight clip generator (FFmpeg API)
├── [BE] CV Bot chat integration
├── [FE] Auto clips tab in VOD player
├── [FE] CV Settings overlay toggle
└── [BE] Roster + OCR number detection
```

---

## 5. API Design

### 5.1 CV Analysis Service (FastAPI — mới)

```python
# cv_service/main.py

POST /start
{
  "stream_id": "abc123",
  "hls_url": "https://stream.booca.online/live/xxx.m3u8",
  "mode": "realtime" | "vod",
  "features": ["possession", "heatmap", "events"]
}

WS  /ws/{stream_id}
# Emit JSON frames:
{
  "type": "possession",
  "data": { "team1": 58.3, "team2": 41.7 }
}
{
  "type": "event",
  "data": { 
    "event": "goal", 
    "team": 1, 
    "timestamp": 1234.5,
    "frame": 37350 
  }
}
{
  "type": "players",
  "data": { "team1_count": 11, "team2_count": 10, "ball_visible": true }
}

GET /heatmap/{stream_id}/{player_id}
# Returns: PNG image (heatmap overlay)

POST /stop/{stream_id}
```

### 5.2 BOOCA Backend (Node.js — bổ sung)

```typescript
// controllers/cvAnalysis.controller.ts (NEW)

POST /api/cv/start-analysis
// Trigger CV service, store session

GET  /api/cv/stats/:streamId
// Latest possession + players data

GET  /api/cv/highlights/:streamId
// List of auto-detected highlights

GET  /api/cv/heatmap/:streamId/:playerId
// Proxy to CV service heatmap image

WS   /ws/cv/:streamId
// Forward CV service WebSocket to FE
```

### 5.3 WebSocket Events (Frontend)

```typescript
// Emitted by server → client
socket.on("cv:possession:update", (data: {
  streamId: string;
  team1: number;   // percentage 0-100
  team2: number;
  frames_processed: number;
}) => {});

socket.on("cv:event:detected", (data: {
  streamId: string;
  event: "goal" | "card" | "corner" | "foul";
  team?: number;
  timestamp: number;
}) => {});

socket.on("cv:players:update", (data: {
  team1_count: number;
  team2_count: number;
  ball_visible: boolean;
}) => {});
```

---

## 6. Frontend Components

### 6.1 CVOverlay.tsx (wrapper)

```tsx
// components/livestream/CVOverlay.tsx
interface CVOverlayProps {
  streamId: string;
  enabled: boolean;
  features: CVFeature[];
}

// Mounts on top of StreamPlayer video area
// Receives CV stats via WebSocket
```

### 6.2 PossessionBar.tsx

```tsx
// Thanh kiểm soát bóng 2 đội
// Light/Dark theme support (user rule)
// Animated progress dùng CSS transition
```

### 6.3 LiveStatsPanel.tsx

```tsx
// Panel stats phải màn hình:
// - Ball possession
// - Player count
// - Top speed hiện tại
// - Event log (bàn thắng, thẻ...)
```

### 6.4 PlayerHeatmapModal.tsx

```tsx
// Dialog hiển thị heatmap
// Canvas overlay lên ảnh pitch chuẩn
// Chọn player từ dropdown
```

### 6.5 CVSettingsMenu.tsx

```tsx
// Tích hợp vào Settings menu của StreamPlayer
// Toggle for each CV feature
// Persist preference in localStorage
```

---

## 7. Tech Stack & Dependencies

### Python CV Service (Mới)

```txt
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
websockets>=12.0
opencv-python-headless>=4.9.0
ultralytics>=8.3.0
supervision>=0.19.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
httpx>=0.27.0       # Async HTTP client
Pillow>=10.3.0      # Heatmap generation
```

### BOOCA Backend (Bổ sung)

```txt
ws-forward          # WebSocket proxy
axios               # HTTP to CV service
bull                # Job queue for VOD analysis
```

### BOOCA Frontend (Bổ sung)

```txt
# Đã có MUI, hls.js, socket.io-client
# Thêm:
canvas              # Heatmap rendering
recharts            # Charts cho stats (optional)
```

---

## 8. Timeline

| Sprint | Tuần | Mục tiêu |
|--------|------|----------|
| Sprint 1 | 1-2 | Setup CV FastAPI service, Docker, WebSocket bridge |
| Sprint 2 | 2-3 | PossessionBar + LiveStatsPanel + BE proxy endpoints |
| Sprint 3 | 4-5 | Speed estimation + Heatmap module |
| Sprint 4 | 5-6 | Event detection (goal/card) + Auto notification |
| Sprint 5 | 7-8 | Auto-clip highlight generator + VOD tab |
| Sprint 6 | 8-9 | CV Bot chat + CVSettings overlay toggle |
| Sprint 7 | 9-10 | Roster sync + OCR số áo + Polish + Testing |

---

## 9. Deployment Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  booca_fe       │────►│  booca_be        │────►│  cv_service    │
│  (Next.js)      │◄────│  (Node.js)       │◄────│  (FastAPI)     │
│  Vercel/Docker  │ WS  │  Docker + PM2    │ WS  │  Docker + GPU  │
└─────────────────┘     └──────────────────┘     └───────┬────────┘
                                                          │
                                               ┌──────────▼───────┐
                                               │  SRS Stream Srv  │
                                               │  HLS m3u8 input  │
                                               └──────────────────┘
```

> **Lưu ý GPU**: CV service nặng nhất khi chạy YOLOv5 realtime. Nên deploy trên instance có GPU (CUDA), hoặc dùng ONNX/INT8 quantization để chạy trên CPU (đã có trong MatchVision Memory.md).

---

## 10. Đánh Giá Mức Độ Khó & ROI

| Tính năng | Độ khó | Thời gian | Impact |
|-----------|--------|-----------|--------|
| Ball Possession Bar | 🟢 Dễ | 3 ngày | ⭐⭐⭐⭐⭐ |
| Player Count | 🟢 Dễ | 1 ngày | ⭐⭐⭐ |
| Live Event Toast | 🟡 Trung | 5 ngày | ⭐⭐⭐⭐⭐ |
| Player Heatmap | 🟡 Trung | 7 ngày | ⭐⭐⭐⭐ |
| Speed Estimation | 🟡 Trung | 5 ngày | ⭐⭐⭐⭐ |
| Auto Clip Highlight | 🔴 Khó | 10 ngày | ⭐⭐⭐⭐⭐ |
| CV Overlay Toggle | 🟢 Dễ | 2 ngày | ⭐⭐⭐ |
| AI Commentary Bot | 🟡 Trung | 6 ngày | ⭐⭐⭐⭐ |
| Roster + OCR | 🔴 Khó | 12 ngày | ⭐⭐⭐ |
| Pass Network | 🔴 Khó | 14 ngày | ⭐⭐⭐ |

---

## 11. Quick Start — Tích Hợp Nhanh Nhất

Nếu muốn demo nhanh, làm theo thứ tự:

```bash
# Step 1: Khởi động CV service
cd football-computer-vision
pip install fastapi uvicorn
# Tạo cv_api.py wrap MatchVision pipeline

# Step 2: Test với Booca HLS URL đã có
# Memory.md ghi nhận: resolve_booca_url() đã hoạt động!
# URL: https://api.booca.online/api/streams/{stream_id}

# Step 3: Thêm PossessionBar vào StreamPlayer.tsx
# → Đây là tính năng đơn giản nhất, impact cao nhất

# Step 4: Kết nối qua WebSocket
# Socket event: cv:possession:update
```

---

*Created: 2026-04-07*  
*Author: AI Planning Assistant*  
*Project: BOOCA SEP × MatchVision CV Integration*
