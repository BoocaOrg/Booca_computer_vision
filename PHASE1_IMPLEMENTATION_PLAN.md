# 🚀 PHASE 1 — Real-time CV Statistics: Implementation Plan

> **Scope**: Tích hợp 3 tính năng CV real-time vào BOOCA Livestream  
> **Thời gian**: ~3 tuần (15 ngày dev)  
> **Stack**: Python FastAPI (CV Service) + Node.js (BOOCA BE) + React/Next.js (BOOCA FE)

---

## 📋 Tổng Quan 3 Tính Năng

| # | Tính năng | Effort | Impact |
|---|-----------|--------|--------|
| 3.1 | Ball Possession Widget | 3 ngày | ⭐⭐⭐⭐⭐ |
| 3.2 | Player Count & Team Detection | 2 ngày | ⭐⭐⭐ |
| 3.3 | Live Event Detection + Toast | 5 ngày | ⭐⭐⭐⭐⭐ |

---

## 🏗️ Kiến Trúc Tổng Thể Phase 1

```
┌──────────────────────────────────────────────────────────────────┐
│  BOOCA FRONTEND (Next.js)                                        │
│                                                                  │
│  StreamPlayer.tsx ─────────────────────────────────────────────  │
│       │                                                         │
│       ├── CVStatsPanel.tsx    ← NEW (Possession + Players)      │
│       ├── CVEventToast.tsx    ← NEW (Goal/Card notification)    │
│       └── PossessionBar.tsx   ← NEW (Animated bar widget)      │
│                    │ socket.io events                           │
└────────────────────┼─────────────────────────────────────────────┘
                     │ cv:possession:update
                     │ cv:players:update  
                     │ cv:event:detected
                     │
┌────────────────────┼─────────────────────────────────────────────┐
│  BOOCA BACKEND (Node.js :5000)                                   │
│                                                                  │
│  socket.ts   ← thêm CV event handlers + emit functions          │
│  cv.controller.ts ← NEW REST API                                │
│  cv.routes.ts     ← NEW routes                                  │
│                    │ HTTP POST (JSON stats)                     │
└────────────────────┼─────────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────────┐
│  CV SERVICE (Python FastAPI :8000)  ← NEW SERVICE               │
│                                                                  │
│  cv_api.py (FastAPI app)                                        │
│       │                                                         │
│       ├── POST /cv/start    → khởi động phân tích               │
│       ├── POST /cv/stop     → dừng phân tích                    │
│       └── Background Task: đọc HLS → chạy CV → POST stats      │
│                    │ cv2.VideoCapture(hls_url)                  │
└────────────────────┼─────────────────────────────────────────────┘
                     │ HLS stream
┌────────────────────┼─────────────────────────────────────────────┐
│  SRS SERVER (stream.booca.online)                               │
│  /live/{streamKey}.m3u8                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📅 Timeline Chi Tiết

```
Tuần 1 (Ngày 1-5)  → CV Service + BOOCA BE endpoints
Tuần 2 (Ngày 6-10) → Frontend components + Socket integration
Tuần 3 (Ngày 11-15)→ Event detection + Polish + Testing
```

---

## 📁 Files Cần Tạo/Sửa

### CV Service (football-computer-vision/) — TẠO MỚI
```
football-computer-vision/
├── cv_service/
│   ├── __init__.py          [NEW]
│   ├── cv_api.py            [NEW] ← FastAPI main app
│   ├── cv_session.py        [NEW] ← Session manager
│   ├── event_detector.py    [NEW] ← Goal/card detection
│   └── requirements_api.txt [NEW] ← FastAPI deps
```

### BOOCA Backend (booca_be/src/) — SỬA/TẠO
```
booca_be/src/
├── services/
│   └── cvAnalysis.service.ts     [NEW] ← Business logic, gọi CV service
├── controllers/
│   └── cvAnalysis.controller.ts  [NEW] ← Chỉ parse req/res, gọi service
├── routes/
│   └── cv.routes.ts              [NEW]
├── utils/
│   └── socket.ts                 [MODIFY] ← thêm CV emit functions
└── routes/
    └── index.routes.ts           [MODIFY] ← đăng ký cv routes
```

### BOOCA Frontend (booca_fe/) — TẠO MỚI
```
booca_fe/
└── components/
    └── livestream/
        ├── CVStatsPanel.tsx    [NEW]
        ├── PossessionBar.tsx   [NEW]
        ├── PlayerCountCard.tsx [NEW]
        └── CVEventToast.tsx    [NEW]
```

---

## 🔧 TASK 1: CV Service (Python FastAPI)

### Ngày 1-2: Tạo `cv_service/cv_api.py`

**Mô tả**: FastAPI app nhận lệnh start/stop phân tích, chạy CV pipeline trong background thread, POST stats về BOOCA BE.

```python
# cv_service/cv_api.py
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import threading, cv2, numpy as np, requests, time, sys, os

# Thêm root vào path để import MatchVision modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

app = FastAPI(title="MatchVision CV Service", version="1.0.0")

# ==== Models ====
class StartRequest(BaseModel):
    stream_id: str          # BOOCA stream ID (MongoDB ObjectId)
    hls_url: str            # HLS m3u8 URL từ stream.booca.online
    booca_callback_url: str # URL BOOCA BE nhận stats (POST)
    features: list[str] = ["possession", "players", "events"]

# ==== State ====
active_sessions: dict = {}  # stream_id → thread + stop_event

# ==== Endpoints ====
@app.post("/cv/start")
async def start_analysis(req: StartRequest, background_tasks: BackgroundTasks):
    if req.stream_id in active_sessions:
        return {"status": "already_running", "stream_id": req.stream_id}
    
    stop_event = threading.Event()
    active_sessions[req.stream_id] = {"stop_event": stop_event}
    
    background_tasks.add_task(
        run_cv_pipeline,
        req.stream_id,
        req.hls_url,
        req.booca_callback_url,
        req.features,
        stop_event
    )
    return {"status": "started", "stream_id": req.stream_id}

@app.post("/cv/stop/{stream_id}")
async def stop_analysis(stream_id: str):
    if stream_id not in active_sessions:
        return {"status": "not_found"}
    active_sessions[stream_id]["stop_event"].set()
    del active_sessions[stream_id]
    return {"status": "stopped", "stream_id": stream_id}

@app.get("/cv/status")
async def get_status():
    return {"active_sessions": list(active_sessions.keys())}

@app.get("/health")
async def health():
    return {"status": "ok"}

# ==== CV Pipeline ====
def run_cv_pipeline(stream_id, hls_url, callback_url, features, stop_event):
    """Chạy trong background thread"""
    cap = cv2.VideoCapture(hls_url)
    if not cap.isOpened():
        print(f"[CV] Cannot open HLS: {hls_url}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print(f"[CV] Cannot read first frame for {stream_id}")
        return

    # Init modules (giống realtime_main.py)
    tracker  = Tracker("models/best.pt", [0,1,2,3], verbose=False)
    cam_est  = CameraMovementEstimator(frame, [0,1,2,3], verbose=False)
    t_assign = TeamAssigner()
    b_assign = PlayerBallAssigner()
    
    teams_assigned = False
    frame_count = 0
    last_post_time = 0
    POST_INTERVAL = 2.0  # POST stats mỗi 2 giây

    print(f"[CV] Started analysis for stream: {stream_id}")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)  # HLS buffer chưa sẵn, thử lại
            cap = cv2.VideoCapture(hls_url)  # Reconnect
            continue

        frame_count += 1
        
        # Chỉ detect mỗi 2 frame để giảm tải CPU
        if frame_count % 2 != 0:
            continue

        try:
            # CV Pipeline (copy từ process_realtime)
            tracks = tracker.get_object_tracks_single_frame(frame)
            tracker.add_position_to_tracks_single_frame(tracks)
            
            cam_mv = cam_est.get_camera_movement_single_frame(frame)
            cam_est.adjust_positions_to_tracks_single_frame(tracks, cam_mv)
            
            players = tracks.get("players", {})
            
            if not teams_assigned and len(players) >= 4:
                t_assign.assign_team_colour(frame, players, force=True)
                teams_assigned = True
            
            if teams_assigned:
                for pid, ptrack in players.items():
                    team = t_assign.get_player_team(frame, ptrack["bbox"], pid)
                    tracks["players"][pid]["team"] = team
            
            b_assign.assign_ball_single_frame(tracks)
            
            # ====== Stats Collection ======
            now = time.time()
            if now - last_post_time >= POST_INTERVAL:
                last_post_time = now
                
                stats = collect_stats(tracks, b_assign, features, frame_count)
                
                try:
                    requests.post(
                        callback_url,
                        json={"stream_id": stream_id, "frame": frame_count, **stats},
                        timeout=2
                    )
                except Exception as e:
                    print(f"[CV] POST failed: {e}")
                    
        except Exception as e:
            print(f"[CV] Frame error: {e}")
            continue

    cap.release()
    print(f"[CV] Stopped analysis for stream: {stream_id}")


def collect_stats(tracks, b_assign, features, frame_count):
    """Thu thập stats từ CV output"""
    stats = {}
    players = tracks.get("players", {})
    
    # === Possession ===
    if "possession" in features and b_assign.ball_possession:
        poss = b_assign.ball_possession
        if hasattr(poss, '__len__') and len(poss) > 0:
            import numpy as np
            poss_arr = np.array([x for x in poss if x is not None and x > 0])
            if len(poss_arr) > 0:
                t1 = float(np.sum(poss_arr == 1) / len(poss_arr) * 100)
                t2 = float(np.sum(poss_arr == 2) / len(poss_arr) * 100)
                stats["possession"] = {"team1": round(t1, 1), "team2": round(t2, 1)}
    
    # === Player Count ===
    if "players" in features:
        team1 = sum(1 for p in players.values() if p.get("team") == 1)
        team2 = sum(1 for p in players.values() if p.get("team") == 2)
        ball_visible = bool(tracks.get("ball", {}))
        stats["players"] = {
            "team1": team1,
            "team2": team2,
            "total": len(players),
            "ball_visible": ball_visible
        }
    
    return stats
```

### Ngày 3: Tạo `cv_service/event_detector.py`

```python
# cv_service/event_detector.py
"""
Phát hiện sự kiện: bàn thắng, thẻ (phase 1 basic)
"""
import numpy as np
from typing import Optional

class EventDetector:
    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height
        
        # Vùng khung thành (ước tính ~10% hai bên của frame)
        self.goal_zone_left  = (0, int(self.fh*0.3), int(self.fw*0.08), int(self.fh*0.7))
        self.goal_zone_right = (int(self.fw*0.92), int(self.fh*0.3), self.fw, int(self.fh*0.7))
        
        # Trạng thái
        self.ball_in_goal_cooldown = 0   # Tránh detect nhiều lần
        self.last_possession_team = None
        
    def check_goal(self, ball_bbox: Optional[list], last_team: int) -> Optional[dict]:
        """
        Kiểm tra bóng có vào vùng khung thành không.
        Returns event dict nếu có bàn thắng, None nếu không.
        """
        if self.ball_in_goal_cooldown > 0:
            self.ball_in_goal_cooldown -= 1
            return None
            
        if not ball_bbox:
            return None
        
        bx1, by1, bx2, by2 = ball_bbox
        ball_cx = (bx1 + bx2) / 2
        ball_cy = (by1 + by2) / 2
        
        in_left_goal  = self._in_zone(ball_cx, ball_cy, self.goal_zone_left)
        in_right_goal = self._in_zone(ball_cx, ball_cy, self.goal_zone_right)
        
        if in_left_goal or in_right_goal:
            self.ball_in_goal_cooldown = 150  # ~5 giây ở 30fps
            # Đội ghi bàn = đội đối diện (bóng vào khung thành đội nào thì đội kia ghi)
            scoring_team = 2 if in_left_goal else 1
            return {
                "event": "goal",
                "scoring_team": scoring_team,
                "side": "left" if in_left_goal else "right"
            }
        
        return None
    
    def _in_zone(self, cx, cy, zone):
        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2
    
    def check_referee_event(self, referees: dict, frame) -> Optional[dict]:
        """
        Phase 1: Basic — nếu referee xuất hiện sau khi game pause lâu
        Phase 2: Dùng pose estimation để detect giơ thẻ
        """
        # TODO Phase 2: MediaPipe pose estimation
        return None
```

### Ngày 3: `cv_service/requirements_api.txt`

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
requests>=2.31.0
```

---

## 🔧 TASK 2: BOOCA Backend

### Ngày 4: Tạo `src/services/cvAnalysis.service.ts`

**Mô tả**: Service layer chứa toàn bộ business logic — gọi CV Python service, forward stats qua Socket.io. Controller chỉ parse HTTP, gọi service. Theo pattern singleton class của `statsStream.service.ts`.

```typescript
// booca_be/src/services/cvAnalysis.service.ts
import axios from "axios";
import { getIO } from "../utils/socket";

const CV_SERVICE_URL = process.env.CV_SERVICE_URL || "http://localhost:8000";
const BOOCA_API_URL  = process.env.API_URL         || "http://localhost:5000/api";

class CVAnalysisService {
  // =========================
  // Gọi CV service: start job
  // =========================
  async startAnalysis(streamId: string, hlsUrl: string): Promise<any> {
    const callbackUrl = `${BOOCA_API_URL}/cv/stats-callback`;
    
    const response = await axios.post(
      `${CV_SERVICE_URL}/cv/start`,
      {
        stream_id: streamId,
        hls_url: hlsUrl,
        booca_callback_url: callbackUrl,
        features: ["possession", "players", "events"],
      },
      { timeout: 5000 }
    );
    
    console.log(`[CVService] Started analysis for stream: ${streamId}`);
    return response.data;
  }

  // =========================
  // Gọi CV service: stop job
  // =========================
  async stopAnalysis(streamId: string): Promise<void> {
    await axios.post(
      `${CV_SERVICE_URL}/cv/stop/${streamId}`,
      {},
      { timeout: 3000 }
    );
    console.log(`[CVService] Stopped analysis for stream: ${streamId}`);
  }

  // =====================================
  // Xử lý stats callback từ Python service
  // Emit Socket.io events tới viewers
  // =====================================
  handleStatsCallback(payload: {
    stream_id: string;
    frame: number;
    possession?: { team1: number; team2: number };
    players?: { team1: number; team2: number; total: number; ball_visible: boolean };
    event?: { event: string; scoring_team?: number };
  }): void {
    const { stream_id, frame, possession, players, event } = payload;
    const io = getIO();
    const room = `stream:${stream_id}`;

    if (possession) {
      io.to(room).emit("cv:possession:update", {
        streamId: stream_id,
        frame,
        team1: possession.team1,
        team2: possession.team2,
      });
    }

    if (players) {
      io.to(room).emit("cv:players:update", {
        streamId: stream_id,
        team1: players.team1,
        team2: players.team2,
        total: players.total,
        ballVisible: players.ball_visible,
      });
    }

    if (event) {
      io.to(room).emit("cv:event:detected", {
        streamId: stream_id,
        event: event.event,
        scoringTeam: event.scoring_team,
        timestamp: Date.now(),
        frame,
      });
      console.log(`[CVService] Event detected: ${event.event} in stream ${stream_id}`);
    }
  }

  // =========================
  // Check Python service health
  // =========================
  async checkHealth(): Promise<{ status: string; activeSessions?: string[] }> {
    try {
      const r = await axios.get(`${CV_SERVICE_URL}/health`, { timeout: 2000 });
      return r.data;
    } catch {
      return { status: "offline" };
    }
  }
}

// Singleton — giống statsStream.service.ts
export default new CVAnalysisService();
```

### Ngày 4: Tạo `src/controllers/cvAnalysis.controller.ts`

**Mô tả**: Chỉ parse req/res, validate input, delegate hết xuống service.

```typescript
// booca_be/src/controllers/cvAnalysis.controller.ts
import { Request, Response } from "express";
import cvAnalysisService from "../services/cvAnalysis.service";

export const startCVAnalysis = async (req: Request, res: Response) => {
  try {
    const { streamId, hlsUrl } = req.body;
    if (!streamId || !hlsUrl) {
      return res.status(400).json({ success: false, message: "Missing streamId or hlsUrl" });
    }
    const data = await cvAnalysisService.startAnalysis(streamId, hlsUrl);
    res.json({ success: true, data });
  } catch (error: any) {
    console.error("[CV Controller] Start error:", error.message);
    res.status(500).json({ success: false, message: "CV service unavailable" });
  }
};

export const stopCVAnalysis = async (req: Request, res: Response) => {
  try {
    await cvAnalysisService.stopAnalysis(req.params.streamId);
    res.json({ success: true, message: "CV analysis stopped" });
  } catch (error: any) {
    console.error("[CV Controller] Stop error:", error.message);
    res.status(500).json({ success: false, message: "Failed to stop" });
  }
};

// Internal endpoint: CV Python service POST về đây
export const receiveCVStats = async (req: Request, res: Response) => {
  try {
    if (!req.body?.stream_id) {
      return res.status(400).json({ success: false });
    }
    cvAnalysisService.handleStatsCallback(req.body);
    res.json({ success: true });
  } catch (error: any) {
    console.error("[CV Controller] Callback error:", error.message);
    res.status(500).json({ success: false });
  }
};

export const getCVServiceStatus = async (_req: Request, res: Response) => {
  const health = await cvAnalysisService.checkHealth();
  res.json({ success: health.status === "ok", cvService: health });
};

### Ngày 4: Tạo `src/routes/cv.routes.ts`

```typescript
// booca_be/src/routes/cv.routes.ts
import { Router } from "express";
import {
  startCVAnalysis,
  stopCVAnalysis,
  receiveCVStats,
  getCVServiceStatus,
} from "../controllers/cvAnalysis.controller";
import { authMiddleware } from "../middlewares/auth.middleware";

const router = Router();

// Internal: CV service gọi về (không cần auth)
router.post("/stats-callback", receiveCVStats);

// Public: Health check
router.get("/status", getCVServiceStatus);

// Protected: Chỉ owner/admin mới start/stop
router.post("/start", authMiddleware, startCVAnalysis);
router.post("/stop/:streamId", authMiddleware, stopCVAnalysis);

export default router;
```

### Ngày 4: Sửa `src/routes/index.routes.ts` — thêm CV route

```typescript
// Thêm vào cuối file index.routes.ts:
import cvRoutes from "./cv.routes";
// ...
router.use("/cv", cvRoutes);
```

### Ngày 4: Sửa `src/utils/socket.ts` — thêm CV emit functions

```typescript
// Thêm vào cuối file socket.ts:

// ================ CV Analysis Events ================

export function emitCVPossession(
  streamId: string,
  data: { team1: number; team2: number; frame: number }
) {
  if (!io) return;
  io.to(`stream:${streamId}`).emit("cv:possession:update", {
    streamId,
    ...data,
    timestamp: Date.now(),
  });
}

export function emitCVPlayers(
  streamId: string,
  data: { team1: number; team2: number; total: number; ballVisible: boolean }
) {
  if (!io) return;
  io.to(`stream:${streamId}`).emit("cv:players:update", {
    streamId,
    ...data,
  });
}

export function emitCVEvent(
  streamId: string,
  data: {
    event: "goal" | "card" | "corner" | "foul";
    scoringTeam?: number;
    timestamp: number;
  }
) {
  if (!io) return;
  io.to(`stream:${streamId}`).emit("cv:event:detected", {
    streamId,
    ...data,
  });
}
```

### Ngày 5: Cập nhật `.env.dev`

```bash
# Thêm vào .env.dev
CV_SERVICE_URL=http://localhost:8000
```

---

## 🎨 TASK 3: Frontend Components

### Ngày 6-7: Tạo `PossessionBar.tsx`

```tsx
// booca_fe/components/livestream/PossessionBar.tsx
"use client";

import React from "react";
import { Box, Typography } from "@mui/material";

interface PossessionBarProps {
  team1: number;   // 0-100
  team2: number;   // 0-100
  team1Color?: string;
  team2Color?: string;
  team1Name?: string;
  team2Name?: string;
  frame?: number;
}

const PossessionBar: React.FC<PossessionBarProps> = ({
  team1,
  team2,
  team1Color = "#ef4444",   // Đỏ default
  team2Color = "#3b82f6",   // Xanh default
  team1Name = "Đội A",
  team2Name = "Đội B",
  frame,
}) => {
  return (
    <Box
      sx={{
        // Light mode
        bgcolor: "rgba(255,255,255,0.95)",
        border: "1px solid rgba(0,0,0,0.08)",
        // Dark mode
        "@media (prefers-color-scheme: dark)": {
          bgcolor: "rgba(15,23,42,0.92)",
          border: "1px solid rgba(255,255,255,0.1)",
        },
        borderRadius: "14px",
        px: 2.5,
        py: 1.5,
        backdropFilter: "blur(12px)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.12)",
        minWidth: 280,
      }}
    >
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.2 }}>
        <Typography sx={{ fontSize: "1rem" }}>⚽</Typography>
        <Typography
          sx={{
            fontWeight: 700,
            fontSize: "0.75rem",
            letterSpacing: "0.08em",
            color: "text.secondary",
            textTransform: "uppercase",
          }}
        >
          Kiểm soát bóng
        </Typography>
      </Box>

      {/* Team 1 row */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.8 }}>
        <Typography
          sx={{
            fontSize: "0.75rem",
            fontWeight: 600,
            minWidth: 48,
            color: team1Color,
          }}
        >
          {team1Name}
        </Typography>
        <Box
          sx={{
            flex: 1,
            height: 8,
            bgcolor: "rgba(0,0,0,0.08)",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              width: `${team1}%`,
              height: "100%",
              bgcolor: team1Color,
              borderRadius: 4,
              transition: "width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }}
          />
        </Box>
        <Typography
          sx={{
            fontSize: "0.8rem",
            fontWeight: 700,
            minWidth: 36,
            textAlign: "right",
            color: team1Color,
          }}
        >
          {team1.toFixed(0)}%
        </Typography>
      </Box>

      {/* Team 2 row */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography
          sx={{
            fontSize: "0.75rem",
            fontWeight: 600,
            minWidth: 48,
            color: team2Color,
          }}
        >
          {team2Name}
        </Typography>
        <Box
          sx={{
            flex: 1,
            height: 8,
            bgcolor: "rgba(0,0,0,0.08)",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              width: `${team2}%`,
              height: "100%",
              bgcolor: team2Color,
              borderRadius: 4,
              transition: "width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)",
            }}
          />
        </Box>
        <Typography
          sx={{
            fontSize: "0.8rem",
            fontWeight: 700,
            minWidth: 36,
            textAlign: "right",
            color: team2Color,
          }}
        >
          {team2.toFixed(0)}%
        </Typography>
      </Box>

      {/* Frame counter */}
      {frame && (
        <Typography
          sx={{
            fontSize: "0.65rem",
            color: "text.disabled",
            mt: 1,
            textAlign: "right",
          }}
        >
          Frame #{frame.toLocaleString()}
        </Typography>
      )}
    </Box>
  );
};

export default PossessionBar;
```

### Ngày 7-8: Tạo `PlayerCountCard.tsx`

```tsx
// booca_fe/components/livestream/PlayerCountCard.tsx
"use client";

import React from "react";
import { Box, Typography, Divider } from "@mui/material";

interface PlayerCountCardProps {
  team1: number;
  team2: number;
  ballVisible: boolean;
  team1Color?: string;
  team2Color?: string;
}

const PlayerCountCard: React.FC<PlayerCountCardProps> = ({
  team1,
  team2,
  ballVisible,
  team1Color = "#ef4444",
  team2Color = "#3b82f6",
}) => {
  return (
    <Box
      sx={{
        bgcolor: "rgba(255,255,255,0.95)",
        "@media (prefers-color-scheme: dark)": {
          bgcolor: "rgba(15,23,42,0.92)",
        },
        borderRadius: "14px",
        px: 2,
        py: 1.2,
        backdropFilter: "blur(12px)",
        display: "flex",
        alignItems: "center",
        gap: 1.5,
        boxShadow: "0 4px 24px rgba(0,0,0,0.12)",
      }}
    >
      {/* Team 1 */}
      <Box sx={{ textAlign: "center" }}>
        <Typography sx={{ fontSize: "1.2rem", fontWeight: 800, color: team1Color }}>
          {team1}
        </Typography>
        <Typography sx={{ fontSize: "0.6rem", color: "text.secondary", fontWeight: 600 }}>
          ĐỘI A
        </Typography>
      </Box>

      <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

      {/* Ball indicator */}
      <Box sx={{ textAlign: "center" }}>
        <Typography sx={{ fontSize: "1rem" }}>
          {ballVisible ? "⚽" : "👁️"}
        </Typography>
        <Typography sx={{ fontSize: "0.55rem", color: "text.disabled" }}>
          {ballVisible ? "Thấy bóng" : "Mất bóng"}
        </Typography>
      </Box>

      <Divider orientation="vertical" flexItem sx={{ mx: 0.5 }} />

      {/* Team 2 */}
      <Box sx={{ textAlign: "center" }}>
        <Typography sx={{ fontSize: "1.2rem", fontWeight: 800, color: team2Color }}>
          {team2}
        </Typography>
        <Typography sx={{ fontSize: "0.6rem", color: "text.secondary", fontWeight: 600 }}>
          ĐỘI B
        </Typography>
      </Box>
    </Box>
  );
};

export default PlayerCountCard;
```

### Ngày 8-9: Tạo `CVEventToast.tsx`

```tsx
// booca_fe/components/livestream/CVEventToast.tsx
"use client";

import React, { useState, useEffect } from "react";
import { Box, Typography, Slide } from "@mui/material";

export interface CVEvent {
  id: string;
  event: "goal" | "card" | "corner" | "foul";
  scoringTeam?: number;
  timestamp: number;
}

interface CVEventToastProps {
  event: CVEvent | null;
}

const EVENT_CONFIG = {
  goal: { icon: "🥅", label: "BÀN THẮNG!", color: "#f59e0b", bg: "rgba(245,158,11,0.15)" },
  card: { icon: "🟨", label: "THẺ PHẠT", color: "#ef4444", bg: "rgba(239,68,68,0.15)" },
  corner: { icon: "🚩", label: "GÓC", color: "#8b5cf6", bg: "rgba(139,92,246,0.15)" },
  foul: { icon: "⚠️", label: "LỖI", color: "#6b7280", bg: "rgba(107,114,128,0.12)" },
};

const CVEventToast: React.FC<CVEventToastProps> = ({ event }) => {
  const [visible, setVisible] = useState(false);
  const [current, setCurrent] = useState<CVEvent | null>(null);

  useEffect(() => {
    if (!event) return;
    setCurrent(event);
    setVisible(true);

    const timer = setTimeout(() => setVisible(false), 5000);
    return () => clearTimeout(timer);
  }, [event]);

  if (!current) return null;

  const config = EVENT_CONFIG[current.event] || EVENT_CONFIG.foul;
  const teamLabel = current.scoringTeam ? ` — Đội ${current.scoringTeam === 1 ? "A" : "B"}` : "";

  return (
    <Slide direction="down" in={visible} mountOnEnter unmountOnExit>
      <Box
        sx={{
          position: "absolute",
          top: 16,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 100,
          bgcolor: config.bg,
          border: `2px solid ${config.color}`,
          backdropFilter: "blur(16px)",
          borderRadius: "16px",
          px: 3,
          py: 1.5,
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          boxShadow: `0 8px 32px ${config.color}40`,
          animation: current.event === "goal"
            ? "goalPulse 0.5s ease-in-out 2"
            : undefined,
          "@keyframes goalPulse": {
            "0%, 100%": { transform: "translateX(-50%) scale(1)" },
            "50%": { transform: "translateX(-50%) scale(1.08)" },
          },
        }}
      >
        <Typography sx={{ fontSize: "1.6rem" }}>{config.icon}</Typography>
        <Box>
          <Typography
            sx={{
              fontWeight: 800,
              fontSize: "0.9rem",
              color: config.color,
              letterSpacing: "0.05em",
            }}
          >
            {config.label}{teamLabel}
          </Typography>
          <Typography sx={{ fontSize: "0.65rem", color: "text.secondary" }}>
            Phát hiện bởi AI • MatchVision CV
          </Typography>
        </Box>
      </Box>
    </Slide>
  );
};

export default CVEventToast;
```

### Ngày 9-10: Tạo `CVStatsPanel.tsx` (wrapper chính)

```tsx
// booca_fe/components/livestream/CVStatsPanel.tsx
"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Box, IconButton, Tooltip, Switch, FormControlLabel } from "@mui/material";
import { Psychology as CVIcon } from "@mui/icons-material";
import { useSocket } from "@/hooks/useSocket"; // hook socket có sẵn trong BOOCA
import PossessionBar from "./PossessionBar";
import PlayerCountCard from "./PlayerCountCard";
import CVEventToast from "./CVEventToast";
import type { CVEvent } from "./CVEventToast";

interface CVStatsPanelProps {
  streamId: string;
  isLive: boolean;
}

interface PossessionData {
  team1: number;
  team2: number;
  frame: number;
}

interface PlayersData {
  team1: number;
  team2: number;
  total: number;
  ballVisible: boolean;
}

const CVStatsPanel: React.FC<CVStatsPanelProps> = ({ streamId, isLive }) => {
  const socket = useSocket();
  const [enabled, setEnabled] = useState(true);
  const [possession, setPossession] = useState<PossessionData | null>(null);
  const [players, setPlayers] = useState<PlayersData | null>(null);
  const [latestEvent, setLatestEvent] = useState<CVEvent | null>(null);

  useEffect(() => {
    if (!socket || !isLive) return;

    const handlePossession = (data: any) => {
      if (data.streamId === streamId) {
        setPossession({ team1: data.team1, team2: data.team2, frame: data.frame });
      }
    };

    const handlePlayers = (data: any) => {
      if (data.streamId === streamId) {
        setPlayers({
          team1: data.team1,
          team2: data.team2,
          total: data.total,
          ballVisible: data.ballVisible,
        });
      }
    };

    const handleEvent = (data: any) => {
      if (data.streamId === streamId) {
        setLatestEvent({
          id: `${data.timestamp}-${data.event}`,
          event: data.event,
          scoringTeam: data.scoringTeam,
          timestamp: data.timestamp,
        });
      }
    };

    socket.on("cv:possession:update", handlePossession);
    socket.on("cv:players:update", handlePlayers);
    socket.on("cv:event:detected", handleEvent);

    return () => {
      socket.off("cv:possession:update", handlePossession);
      socket.off("cv:players:update", handlePlayers);
      socket.off("cv:event:detected", handleEvent);
    };
  }, [socket, streamId, isLive]);

  if (!isLive) return null;

  return (
    <>
      {/* Event Toast — absolute, top of video */}
      <CVEventToast event={latestEvent} />

      {/* Stats Overlay — bottom-left above controls */}
      <Box
        sx={{
          position: "absolute",
          bottom: 72,  // Above player controls bar
          left: 16,
          display: "flex",
          flexDirection: "column",
          gap: 1,
          zIndex: 20,
          opacity: enabled ? 1 : 0.3,
          transition: "opacity 0.3s ease",
          pointerEvents: enabled ? "auto" : "none",
        }}
      >
        {possession && (
          <PossessionBar
            team1={possession.team1}
            team2={possession.team2}
            frame={possession.frame}
          />
        )}
        {players && (
          <PlayerCountCard
            team1={players.team1}
            team2={players.team2}
            ballVisible={players.ballVisible}
          />
        )}
      </Box>

      {/* Toggle Button — top right corner */}
      <Tooltip title={`CV Stats ${enabled ? "Bật" : "Tắt"}`}>
        <IconButton
          onClick={() => setEnabled(!enabled)}
          size="small"
          sx={{
            position: "absolute",
            top: 12,
            right: 52,
            zIndex: 30,
            bgcolor: enabled
              ? "rgba(0,224,202,0.2)"
              : "rgba(0,0,0,0.4)",
            color: enabled ? "#00e0ca" : "rgba(255,255,255,0.5)",
            border: enabled
              ? "1px solid rgba(0,224,202,0.4)"
              : "1px solid rgba(255,255,255,0.1)",
            backdropFilter: "blur(8px)",
            "&:hover": { bgcolor: "rgba(0,224,202,0.3)" },
            transition: "all 0.2s ease",
          }}
        >
          <CVIcon sx={{ fontSize: 18 }} />
        </IconButton>
      </Tooltip>
    </>
  );
};

export default CVStatsPanel;
```

### Ngày 10: Tích hợp vào `StreamPlayer.tsx`

**Thêm CVStatsPanel vào container của StreamPlayer:**

```tsx
// Trong StreamPlayer.tsx - thêm import
import CVStatsPanel from "./CVStatsPanel";

// Trong JSX, bên trong containerRef Box (sau <video> tag):
{stream.status === "live" && (
  <CVStatsPanel
    streamId={stream._id}
    isLive={stream.status === "live"}
  />
)}
```

---

## 🧪 Testing Plan

### Unit Test CV Service
```bash
cd football-computer-vision
pip install fastapi uvicorn requests
uvicorn cv_service.cv_api:app --port 8000 --reload

# Test start
curl -X POST http://localhost:8000/cv/start \
  -H "Content-Type: application/json" \
  -d '{"stream_id":"test123","hls_url":"https://stream.booca.online/live/xxx.m3u8","booca_callback_url":"http://localhost:5000/api/cv/stats-callback","features":["possession","players"]}'

# Test status
curl http://localhost:8000/cv/status

# Test stop
curl -X POST http://localhost:8000/cv/stop/test123
```

### Integration Test
```bash
# 1. Start BOOCA BE
cd booca_be && npm run dev

# 2. Start CV Service  
cd football-computer-vision && uvicorn cv_service.cv_api:app --port 8000

# 3. Open BOOCA FE livestream page
# 4. Verify socket events trong Browser DevTools:
#    Network → WS → filter "cv:"
```

### Kiểm tra UI
- [ ] PossessionBar animate mượt khi data thay đổi
- [ ] Dark mode / Light mode đúng theo user rule
- [ ] CVEventToast hiện × 5s rồi tự tắt
- [ ] Toggle button bật/tắt overlay
- [ ] Không crash khi CV service offline (graceful fallback)

---

## ⚠️ Potential Issues & Mitigation

| Issue | Mitigation |
|-------|-----------|
| HLS URL expired/rotated | CV service auto-reconnect với URL mới từ BOOCA API |
| CV service CPU overload | frame_skip=2, imgsz=416 để giảm tải |
| Teams not assigned (< 4 players) | Hiển thị "Đang phân tích..." thay vì 0%/0% |
| Socket event flood (2s interval) | Throttle ở BE: chỉ emit nếu possession thay đổi > 1% |
| CV service down | FE không show panel, không crash stream viewer |
| Goal false positive | cooldown 150 frames (~5s) sau mỗi detect |

---

## 📋 Checklist Hoàn Thành

### CV Service
- [ ] `cv_service/cv_api.py` — FastAPI endpoints
- [ ] `cv_service/cv_session.py` — Session state management
- [ ] `cv_service/event_detector.py` — Goal detection logic
- [ ] `cv_service/requirements_api.txt`
- [ ] Dockerfile cho CV service (optional)

### BOOCA Backend
- [ ] `cv.routes.ts` — routes
- [ ] `cvAnalysis.controller.ts` — controller
- [ ] `socket.ts` — thêm CV emit functions
- [ ] `index.routes.ts` — đăng ký route
- [ ] `.env.dev` — thêm `CV_SERVICE_URL`
- [ ] Lint check `npx tsc --noEmit`

### BOOCA Frontend
- [ ] `PossessionBar.tsx` — widget chính
- [ ] `PlayerCountCard.tsx` — đếm cầu thủ
- [ ] `CVEventToast.tsx` — notification sự kiện
- [ ] `CVStatsPanel.tsx` — wrapper + socket logic
- [ ] `StreamPlayer.tsx` — tích hợp CVStatsPanel
- [ ] Dark/Light theme check (user rule)
- [ ] Lint check

---

*Phase 1 Implementation Plan — Generated 2026-04-07*  
*Next: Phase 2 — Player Heatmap & Speed Estimation*
