"""
MatchVision CV Service — FastAPI app for real-time football analysis.

Reads HLS streams from BOOCA, runs the MatchVision CV pipeline,
and POSTs stats back to BOOCA backend which broadcasts via Socket.io.
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Iterator, Tuple
import threading
import cv2
import numpy as np
import requests
import time
import subprocess
import struct
import sys
import sys
import os

# Add project root to path so we can import MatchVision modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator
from cv_service.event_detector import EventDetector
from cv_service.speed_estimator import SpeedEstimator
from cv_service.tactical_analyzer import TacticalAnalyzer
from cv_service.ocr_reader import JerseyOCRReader

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="MatchVision CV Service",
    description="Real-time football analysis engine for BOOCA livestream",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class StartRequest(BaseModel):
    stream_id: str
    hls_url: str
    booca_callback_url: str
    features: List[str] = ["possession", "players", "events"]


class SessionInfo(BaseModel):
    stream_id: str
    status: str
    frame_count: int = 0
    started_at: float = 0


# ─────────────────────────────────────────────
# In-memory session registry
# ─────────────────────────────────────────────
active_sessions: Dict[str, Dict] = {}
_lock = threading.Lock()

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}


@app.get("/cv/status")
async def get_status():
    infos = []
    with _lock:
        for sid, sess in active_sessions.items():
            infos.append({
                "stream_id": sid,
                "status": "running",
                "frame_count": sess.get("frame_count", 0),
            })
    return {"active_sessions": infos}


@app.post("/cv/start")
async def start_analysis(req: StartRequest, background_tasks: BackgroundTasks):
    with _lock:
        if req.stream_id in active_sessions:
            return {"status": "already_running", "stream_id": req.stream_id}

        stop_event = threading.Event()
        active_sessions[req.stream_id] = {
            "stop_event": stop_event,
            "frame_count": 0,
            "started_at": time.time(),
        }

    background_tasks.add_task(
        _run_cv_pipeline,
        req.stream_id,
        req.hls_url,
        req.booca_callback_url,
        req.features,
        stop_event,
    )
    return {"status": "started", "stream_id": req.stream_id}


@app.post("/cv/stop/{stream_id}")
async def stop_analysis(stream_id: str):
    with _lock:
        session = active_sessions.pop(stream_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session["stop_event"].set()
    return {"status": "stopped", "stream_id": stream_id}


# ─────────────────────────────────────────────
# VOD Analysis (process entire video file)
# ─────────────────────────────────────────────
class VodAnalyzeRequest(BaseModel):
    vod_id: str
    video_url: str
    callback_url: str  # BOOCA BE endpoint to POST results back
    frame_skip: int = 3  # Process every Nth frame (higher = faster)


@app.post("/cv/analyze-vod")
async def analyze_vod(req: VodAnalyzeRequest, background_tasks: BackgroundTasks):
    with _lock:
        key = f"vod:{req.vod_id}"
        if key in active_sessions:
            return {"status": "already_processing", "vod_id": req.vod_id}

        stop_event = threading.Event()
        active_sessions[key] = {
            "stop_event": stop_event,
            "frame_count": 0,
            "started_at": time.time(),
            "type": "vod",
        }

    background_tasks.add_task(
        _run_vod_analysis,
        req.vod_id,
        req.video_url,
        req.callback_url,
        req.frame_skip,
        stop_event,
    )
    return {"status": "started", "vod_id": req.vod_id}


@app.post("/cv/cancel-vod/{vod_id}")
async def cancel_vod_analysis(vod_id: str):
    with _lock:
        session = active_sessions.pop(f"vod:{vod_id}", None)
    if session is None:
        raise HTTPException(status_code=404, detail="VOD analysis not found")
    session["stop_event"].set()
    return {"status": "cancelled", "vod_id": vod_id}


# ─────────────────────────────────────────────
# CV Pipeline (runs in background thread)
# ─────────────────────────────────────────────
def _run_cv_pipeline(
    stream_id: str,
    hls_url: str,
    callback_url: str,
    features: List[str],
    stop_event: threading.Event,
):
    """Main CV loop — reads HLS, detects, tracks, collects stats, POSTs back."""
    print(f"[CV] Starting pipeline for stream={stream_id}")
    print(f"[CV]   HLS: {hls_url}")
    print(f"[CV]   Callback: {callback_url}")

    cap = cv2.VideoCapture(hls_url)
    if not cap.isOpened():
        print(f"[CV] ERROR: Cannot open HLS stream: {hls_url}")
        _cleanup_session(stream_id)
        return

    ret, frame = cap.read()
    if not ret:
        print(f"[CV] ERROR: Cannot read first frame for {stream_id}")
        cap.release()
        _cleanup_session(stream_id)
        return

    # Get frame dimensions for event detector
    fh, fw = frame.shape[:2]

    # Init CV modules (same as realtime_main.py)
    tracker = Tracker(MODEL_PATH, [0, 1, 2, 3], verbose=False)
    cam_est = CameraMovementEstimator(frame, [0, 1, 2, 3], verbose=False)
    t_assign = TeamAssigner()
    b_assign = PlayerBallAssigner()
    ev_detect = EventDetector(fw, fh) if "events" in features else None
    speed_est = SpeedEstimator(fps=30)
    tact_analyzer = TacticalAnalyzer()
    ocr_reader = JerseyOCRReader(min_bbox_height=150)

    teams_assigned = False
    frame_count = 0
    last_post_time = 0.0
    POST_INTERVAL = 2.0  # seconds between stats posts
    reconnect_attempts = 0
    MAX_RECONNECT = 5

    print(f"[CV] Pipeline running for stream={stream_id} ({fw}x{fh})")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            reconnect_attempts += 1
            if reconnect_attempts > MAX_RECONNECT:
                print(f"[CV] Max reconnect attempts reached for {stream_id}")
                break
            print(f"[CV] Reconnecting ({reconnect_attempts}/{MAX_RECONNECT})...")
            time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(hls_url)
            continue

        reconnect_attempts = 0
        frame_count += 1

        # Update session frame count
        with _lock:
            if stream_id in active_sessions:
                active_sessions[stream_id]["frame_count"] = frame_count

        # Skip odd frames to reduce CPU load
        if frame_count % 2 != 0:
            continue

        try:
            # ── Detection & Tracking ──
            tracks = tracker.get_object_tracks_single_frame(frame)
            tracker.add_position_to_tracks_single_frame(tracks)

            # ── Camera Movement ──
            cam_mv = cam_est.get_camera_movement_single_frame(frame)
            cam_est.adjust_positions_to_tracks_single_frame(tracks, cam_mv)

            # ── Team Assignment ──
            players = tracks.get("players", {})
            if not teams_assigned and len(players) >= 4:
                t_assign.assign_team_colour(frame, players, force=True)
                teams_assigned = True
                print(f"[CV] Teams calibrated for {stream_id}")

            if teams_assigned:
                for pid, ptrack in players.items():
                    team = t_assign.get_player_team(frame, ptrack["bbox"], pid)
                    tracks["players"][pid]["team"] = team
                    tracks["players"][pid]["team_colour"] = t_assign.team_colours[team]

            # ── Ball Possession ──
            b_assign.assign_ball_single_frame(tracks)

            # ── Collect & POST stats ──
            now = time.time()
            
            # Speed Estimation (Update every frame to build history)
            top_speeds = speed_est.update_speeds(tracks, frame_count)
            
            # Pass Detection 
            pass_event = tact_analyzer.detect_passes(tracks, b_assign.player_has_ball, frame_count)

            # OCR Jersey Numbers (runs only on large bboxes)
            jersey_map = ocr_reader.process_frame(frame, tracks, frame_count)

            if now - last_post_time >= POST_INTERVAL:
                last_post_time = now

                stats = _collect_stats(tracks, b_assign, ev_detect, features, frame_count, top_speeds, pass_event)

                try:
                    requests.post(
                        callback_url,
                        json={"stream_id": stream_id, "frame": frame_count, **stats},
                        timeout=2,
                    )
                except Exception as e:
                    print(f"[CV] POST callback failed: {e}")

        except Exception as e:
            print(f"[CV] Frame {frame_count} error: {e}")
            continue

    cap.release()
    _cleanup_session(stream_id)
    print(f"[CV] Pipeline stopped for stream={stream_id} (processed {frame_count} frames)")


def _collect_stats(
    tracks: Dict,
    b_assign: PlayerBallAssigner,
    ev_detect: Optional[EventDetector],
    features: List[str],
    frame_count: int,
    top_speeds: list = None,
    pass_event: dict = None
) -> Dict:
    """Collect stats from current CV state."""
    stats: Dict = {}
    players = tracks.get("players", {})

    # ── Possession ──
    if "possession" in features and b_assign.ball_possession:
        poss = b_assign.ball_possession
        if hasattr(poss, "__len__") and len(poss) > 0:
            poss_arr = np.array([x for x in poss if x is not None and x > 0])
            if len(poss_arr) > 0:
                t1 = float(np.sum(poss_arr == 1) / len(poss_arr) * 100)
                t2 = float(np.sum(poss_arr == 2) / len(poss_arr) * 100)
                stats["possession"] = {"team1": round(t1, 1), "team2": round(t2, 1)}

    # ── Player Count ──
    if "players" in features:
        team1 = sum(1 for p in players.values() if p.get("team") == 1)
        team2 = sum(1 for p in players.values() if p.get("team") == 2)
        ball_visible = bool(tracks.get("ball", {}))
        stats["players"] = {
            "team1": team1,
            "team2": team2,
            "total": len(players),
            "ball_visible": ball_visible,
        }

    # ── Events ──
    if "events" in features and ev_detect is not None:
        event = ev_detect.check_events(tracks, frame_count)
        if event:
            stats["event"] = event
            
        if pass_event:
            stats["event"] = pass_event

    # ── Speeds ──
    if top_speeds:
        stats["speeds"] = top_speeds

    return stats


def _cleanup_session(stream_id: str):
    """Remove session from active registry."""
    with _lock:
        active_sessions.pop(stream_id, None)


# ─────────────────────────────────────────────
# FFmpeg Frame Reader (supports HLS + MP4)
# ─────────────────────────────────────────────
def _get_ffmpeg_reader(video_url: str) -> Tuple[Optional[cv2.VideoCapture], Optional[dict]]:
    """
    Create a VideoCapture using FFmpeg pipe for HLS/M3U8, or direct cv2 for MP4.
    Returns (cap, metadata) where metadata has fps, width, height, total_frames.
    Falls back to plain cv2.VideoCapture if FFmpeg pipe fails.
    """
    is_hls = video_url.endswith(".m3u8") or ".m3u8?" in video_url

    if is_hls:
        # Use FFmpeg pipe: decode HLS to raw frames
        ffmpeg_cmd = [
            "ffmpeg", "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5", "-i", video_url,
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
        ]
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8,
            )
            # Get metadata via ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-of", "default=noprint_wrappers=1", video_url
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            meta = {}
            for line in probe_result.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k] = v

            width = int(meta.get("width", 1920))
            height = int(meta.get("height", 1080))
            fps_str = meta.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str) or 30.0
            total_frames = int(meta.get("nb_frames", 0)) or 0

            metadata = {
                "width": width, "height": height,
                "fps": fps, "total_frames": total_frames,
                "proc": proc,
            }
            # Return a dummy cap with metadata attached; real reading via proc
            cap = cv2.VideoCapture(video_url)  # dummy, will be overridden
            cap._ffmpeg_proc = proc  # type: ignore
            cap._ffmpeg_meta = metadata  # type: ignore
            cap._is_ffmpeg_pipe = True  # type: ignore
            return cap, metadata
        except Exception as e:
            print(f"[CV-VOD] FFmpeg pipe failed for HLS: {e}, falling back to cv2")
            cap = cv2.VideoCapture(video_url)
            if cap.isOpened():
                meta = _get_cv2_metadata(cap)
                return cap, meta
            return None, None
    else:
        cap = cv2.VideoCapture(video_url)
        if cap.isOpened():
            return cap, _get_cv2_metadata(cap)
        return None, None


def _get_cv2_metadata(cap: cv2.VideoCapture) -> dict:
    """Extract metadata from an open cv2.VideoCapture."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return {"width": width, "height": height, "fps": fps, "total_frames": total}


# ─────────────────────────────────────────────
# VOD Analysis Pipeline (processes full video)
# ─────────────────────────────────────────────
def _run_vod_analysis(
    vod_id: str,
    video_url: str,
    callback_url: str,
    frame_skip: int,
    stop_event: threading.Event,
):
    """Process entire VOD video and POST aggregated stats back."""
    session_key = f"vod:{vod_id}"
    print(f"[CV-VOD] Starting analysis for vod={vod_id}")
    print(f"[CV-VOD]   URL: {video_url}")
    print(f"[CV-VOD]   frame_skip: {frame_skip}")

    cap, meta = _get_ffmpeg_reader(video_url)
    if cap is None:
        print(f"[CV-VOD] ERROR: Cannot open video: {video_url}")
        _post_vod_result(callback_url, vod_id, None, error="Cannot open video URL")
        _cleanup_session(session_key)
        return

    is_ffmpeg_pipe = getattr(cap, "_is_ffmpeg_pipe", False)

    # Read first frame
    if is_ffmpeg_pipe:
        proc: subprocess.Popen = cap._ffmpeg_proc  # type: ignore
        raw = proc.stdout.read(meta["width"] * meta["height"] * 3)
        if len(raw) < meta["width"] * meta["height"] * 3:
            print(f"[CV-VOD] ERROR: Cannot read first frame from FFmpeg pipe")
            proc.terminate()
            _post_vod_result(callback_url, vod_id, None, error="Cannot read video frames")
            _cleanup_session(session_key)
            return
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(meta["height"], meta["width"], 3)
        fh, fw = frame.shape[:2]
        fps = meta["fps"]
        total_video_frames = meta["total_frames"]
    else:
        ret, frame = cap.read()
        if not ret:
            print(f"[CV-VOD] ERROR: Cannot read first frame")
            cap.release()
            _post_vod_result(callback_url, vod_id, None, error="Cannot read video frames")
            _cleanup_session(session_key)
            return
        fh, fw = frame.shape[:2]
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Init CV modules
    tracker = Tracker(MODEL_PATH, [0, 1, 2, 3], verbose=False)
    cam_est = CameraMovementEstimator(frame, [0, 1, 2, 3], verbose=False)
    t_assign = TeamAssigner()
    b_assign = PlayerBallAssigner()
    ev_detect = EventDetector(fw, fh)
    speed_est = SpeedEstimator(fps=fps)
    tact_analyzer = TacticalAnalyzer()

    teams_assigned = False
    frame_count = 0

    # Aggregation accumulators
    all_possession = []       # list of team IDs (1 or 2)
    all_player_counts = []    # list of {team1: N, team2: N}
    all_events = []           # list of detected events

    print(f"[CV-VOD] Video: {fw}x{fh}, {total_video_frames} frames, {fps:.1f} fps")

    # Unified frame reader: handles both cv2 and FFmpeg pipe
    def read_frame(cap: cv2.VideoCapture, is_pipe: bool, meta: dict, proc: subprocess.Popen = None) -> Tuple[bool, any]:
        if is_pipe and proc:
            frame_size = meta["width"] * meta["height"] * 3
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                return False, None
            return True, np.frombuffer(raw, dtype=np.uint8).reshape(meta["height"], meta["width"], 3)
        else:
            return cap.read()

    # Reset to beginning (only for cv2, FFmpeg pipe always reads from start)
    if not is_ffmpeg_pipe:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    proc = getattr(cap, "_ffmpeg_proc", None) if is_ffmpeg_pipe else None

    while not stop_event.is_set():
        ret, frame = read_frame(cap, is_ffmpeg_pipe, meta, proc)
        if not ret:
            break

        frame_count += 1

        # Update progress
        with _lock:
            if session_key in active_sessions:
                active_sessions[session_key]["frame_count"] = frame_count

        # Skip frames for performance
        if frame_count % frame_skip != 0:
            continue

        try:
            # Detection & Tracking
            tracks = tracker.get_object_tracks_single_frame(frame)
            tracker.add_position_to_tracks_single_frame(tracks)

            cam_mv = cam_est.get_camera_movement_single_frame(frame)
            cam_est.adjust_positions_to_tracks_single_frame(tracks, cam_mv)

            # Team Assignment
            players = tracks.get("players", {})
            if not teams_assigned and len(players) >= 4:
                t_assign.assign_team_colour(frame, players, force=True)
                teams_assigned = True

            if teams_assigned:
                for pid, ptrack in players.items():
                    team = t_assign.get_player_team(frame, ptrack["bbox"], pid)
                    tracks["players"][pid]["team"] = team

            # Ball Possession
            b_assign.assign_ball_single_frame(tracks)

            # Collect per-frame data
            if teams_assigned:
                t1_count = sum(1 for p in players.values() if p.get("team") == 1)
                t2_count = sum(1 for p in players.values() if p.get("team") == 2)
                all_player_counts.append({"team1": t1_count, "team2": t2_count})

            # Events
            event = ev_detect.check_events(tracks, frame_count)
            if event:
                event["frame"] = frame_count
                event["timestamp"] = round(frame_count / fps, 1)
                all_events.append(event)
                
            # Speed Estimation 
            speed_est.update_speeds(tracks, frame_count)
            
            # Pass Detection 
            pass_event = tact_analyzer.detect_passes(tracks, b_assign.player_has_ball, frame_count)
            if pass_event:
                pass_event["timestamp"] = round(frame_count / fps, 1)
                all_events.append(pass_event)

        except Exception as e:
            print(f"[CV-VOD] Frame {frame_count} error: {e}")
            continue

        # Log progress every 500 processed frames
        if (frame_count // frame_skip) % 500 == 0:
            pct = round(frame_count / max(total_video_frames, 1) * 100, 1)
            print(f"[CV-VOD] Progress: {pct}% ({frame_count}/{total_video_frames})")

    # Cleanup
    if is_ffmpeg_pipe and proc:
        proc.terminate()
        proc.wait(timeout=5)
    cap.release()

    # ── Aggregate final stats ──
    result = _aggregate_vod_stats(b_assign, all_player_counts, all_events, frame_count, speed_est, tact_analyzer)

    print(f"[CV-VOD] Completed vod={vod_id}: {frame_count} frames, "
          f"possession={result.get('possession')}, events={len(all_events)}")

    # POST results back to BOOCA BE
    _post_vod_result(callback_url, vod_id, result)
    _cleanup_session(session_key)


def _aggregate_vod_stats(
    b_assign: PlayerBallAssigner,
    player_counts: List,
    events: List,
    total_frames: int,
    speed_est: SpeedEstimator = None,
    tact_analyzer: TacticalAnalyzer = None
) -> Dict:
    """Aggregate all per-frame data into final VOD stats."""
    result: Dict = {"totalFrames": total_frames}

    # Possession
    if b_assign.ball_possession:
        poss = b_assign.ball_possession
        poss_arr = np.array([x for x in poss if x is not None and x > 0])
        if len(poss_arr) > 0:
            t1 = float(np.sum(poss_arr == 1) / len(poss_arr) * 100)
            t2 = float(np.sum(poss_arr == 2) / len(poss_arr) * 100)
            result["possession"] = {"team1": round(t1, 1), "team2": round(t2, 1)}

    # Average player counts
    if player_counts:
        avg_t1 = sum(p["team1"] for p in player_counts) / len(player_counts)
        avg_t2 = sum(p["team2"] for p in player_counts) / len(player_counts)
        result["players"] = {
            "team1Avg": round(avg_t1, 1),
            "team2Avg": round(avg_t2, 1),
        }

    # Events
    if events:
        result["events"] = events
        # Count passes vs other events 
        passes = [e for e in events if e.get("event") == "pass"]
        result["tactics"] = {
            "totalPasses": len(passes)
        }
        
    # Speeds 
    if speed_est:
        # Get overall max speeds across all tracked players
        max_speeds = []
        for pid, data in speed_est.current_speeds.items():
            if data > 5.0: # Meaningful speed 
                max_speeds.append({"player_id": int(pid), "speed": data})
        
        max_speeds.sort(key=lambda x: x["speed"], reverse=True)
        result["speedStats"] = {
            "topSpeeds": max_speeds[:5]
        }

    return result


def _post_vod_result(callback_url: str, vod_id: str, result: Optional[Dict], error: str = None):
    """POST VOD analysis results back to BOOCA backend."""
    payload: Dict = {"vod_id": vod_id}
    if error:
        payload["status"] = "failed"
        payload["error"] = error
    else:
        payload["status"] = "completed"
        payload["result"] = result or {}

    try:
        requests.post(callback_url, json=payload, timeout=10)
        print(f"[CV-VOD] Results posted for vod={vod_id}")
    except Exception as e:
        print(f"[CV-VOD] POST results failed: {e}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("cv_service.cv_api:app", host="0.0.0.0", port=8000, reload=True)
