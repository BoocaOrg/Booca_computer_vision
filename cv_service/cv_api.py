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
from urllib.parse import urlparse
import random

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
from cv_service.calibration import PitchCalibrator

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
# Mount Highlight Worker (so both services run on one port)
# ─────────────────────────────────────────────
from cv_service.highlight_worker import app as highlight_app
app.mount("/highlight-worker", highlight_app)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class StartRequest(BaseModel):
    stream_id: str
    hls_url: str
    booca_callback_url: str
    features: List[str] = ["possession", "players", "events"]
    # Optional: protect /cv/start in production without changing callers that don't pass it.
    # When CV_START_TOKEN is set, requests MUST include this token.
    token: Optional[str] = None


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

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")

def _download_file(url: str, dest_path: str, timeout_sec: float = 30.0) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_sec) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def _resolve_model_path() -> str:
    """
    Resolve YOLO weights path for runtime environments like Render.

    - If CV_MODEL_PATH points to an existing file, use it.
    - Else if CV_MODEL_URL is set, download weights to /tmp and use them.
    - Else if repo default exists, use it (local dev).
    - Else fall back to a built-in Ultralytics model name so service still runs.
    """
    env_path = os.getenv("CV_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        print(f"[CV] Using CV_MODEL_PATH={env_path}")
        return env_path

    if os.path.exists(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    url = os.getenv("CV_MODEL_URL")
    if url:
        dest = os.getenv("CV_MODEL_DOWNLOAD_PATH") or "/tmp/matchvision/models/best.pt"
        if not os.path.exists(dest):
            print(f"[CV] Downloading model weights from CV_MODEL_URL to {dest}")
            _download_file(url, dest, timeout_sec=float(os.getenv("CV_MODEL_DOWNLOAD_TIMEOUT_SEC", "60")))
        else:
            print(f"[CV] Using cached downloaded model at {dest}")
        return dest

    fallback = os.getenv("CV_MODEL_FALLBACK", "yolov8n.pt")
    print(
        f"[CV] WARNING: Model weights not found at {DEFAULT_MODEL_PATH} and CV_MODEL_URL not set. "
        f"Falling back to {fallback} (accuracy may be reduced)."
    )
    return fallback

def _read_exact_with_timeout(stream, nbytes: int, timeout_sec: float) -> bytes:
    """
    Read exactly nbytes from a stream within timeout_sec.
    Returns bytes read (may be shorter if timed out).

    Prevents deadlock when ffmpeg stdout produces no frames.
    """
    import select
    buf = bytearray()
    deadline = time.time() + float(timeout_sec)
    while len(buf) < nbytes and time.time() < deadline:
        remaining = nbytes - len(buf)
        # Wait until fd is readable
        rlist, _, _ = select.select([stream], [], [], max(0.0, deadline - time.time()))
        if not rlist:
            break
        chunk = stream.read(remaining)
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


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
    required = os.getenv("CV_START_TOKEN")
    if required and (req.token != required):
        raise HTTPException(status_code=401, detail="Unauthorized")

    max_sessions = int(os.getenv("CV_MAX_ACTIVE_SESSIONS", "2"))
    with _lock:
        if len(active_sessions) >= max_sessions:
            raise HTTPException(status_code=429, detail="Too many active sessions")
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
    try:
        _run_cv_pipeline_inner(stream_id, hls_url, callback_url, features, stop_event)
    except Exception as e:
        import traceback
        print(f"[CV] FATAL ERROR in pipeline for stream={stream_id}: {e}")
        traceback.print_exc()
        _cleanup_session(stream_id)


def _run_cv_pipeline_inner(
    stream_id: str,
    hls_url: str,
    callback_url: str,
    features: List[str],
    stop_event: threading.Event,
):
    """Inner pipeline logic — wrapped by _run_cv_pipeline for error handling."""
    parsed = urlparse(hls_url)
    hls_host = parsed.netloc or parsed.hostname or ""
    print(f"[CV] Starting pipeline for stream={stream_id}")
    print(f"[CV]   HLS: {hls_url}")
    if hls_host:
        print(f"[CV]   HLS host: {hls_host}")
    print(f"[CV]   Callback: {callback_url}")

    is_hls = hls_url.endswith(".m3u8") or ".m3u8?" in hls_url

    # ── Check if FFmpeg binary is available ──
    ffmpeg_available = _check_ffmpeg_available()
    print(f"[CV] FFmpeg binary available: {ffmpeg_available}")

    # ── Open stream ──
    if is_hls and ffmpeg_available:
        # Use FFmpeg pipe for HLS stability
        print(f"[CV] Using FFmpeg pipe for HLS stream")
        cap, meta = _get_ffmpeg_reader(hls_url)
        if cap is None:
            print(f"[CV] FFmpeg reader failed, falling back to cv2.VideoCapture")
            cap, meta = _open_hls_via_cv2(hls_url)
    elif is_hls:
        # No FFmpeg binary — use cv2.VideoCapture directly (OpenCV has built-in FFmpeg)
        print(f"[CV] No FFmpeg binary — using cv2.VideoCapture directly for HLS")
        cap, meta = _open_hls_via_cv2(hls_url)
    else:
        cap = cv2.VideoCapture(hls_url)
        meta = _get_cv2_metadata(cap) if cap.isOpened() else None

    if cap is None or (not getattr(cap, '_is_ffmpeg_pipe', False) and not cap.isOpened()):
        print(f"[CV] ERROR: Cannot open stream: {hls_url}")
        _cleanup_session(stream_id)
        return

    is_ffmpeg_pipe = getattr(cap, "_is_ffmpeg_pipe", False)
    proc = getattr(cap, "_ffmpeg_proc", None) if is_ffmpeg_pipe else None
    fps = meta.get("fps", 30) if meta else 30
    fw = meta.get("width", 1920) if meta else 1920
    fh = meta.get("height", 1080) if meta else 1080
    frame_size = fw * fh * 3

    # ── Read first frame ──
    if is_ffmpeg_pipe and proc:
        raw = _read_exact_with_timeout(proc.stdout, frame_size, timeout_sec=float(os.getenv("CV_FFMPEG_FIRST_FRAME_TIMEOUT_SEC", "12")))
        if len(raw) < frame_size:
            print(f"[CV] ERROR: Cannot read first frame from FFmpeg pipe")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            print(f"[CV] Falling back to cv2.VideoCapture for stream={stream_id}")
            cap, meta = _open_hls_via_cv2(hls_url)
            if cap is None or not cap.isOpened():
                _cleanup_session(stream_id)
                return
            ret, frame = cap.read()
            if not ret:
                print(f"[CV] ERROR: Cannot read first frame via cv2 fallback for {stream_id}")
                cap.release()
                _cleanup_session(stream_id)
                return
            fh, fw = frame.shape[:2]
            is_ffmpeg_pipe = False
            proc = None
            meta = meta or {}
            fps = meta.get("fps", fps)
            _cleanup_session(stream_id)
        else:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw, 3)
    else:
        ret, frame = cap.read()
        if not ret:
            print(f"[CV] ERROR: Cannot read first frame for {stream_id}")
            cap.release()
            _cleanup_session(stream_id)
            return
        fh, fw = frame.shape[:2]

    print(f"[CV] First frame read successfully: {fw}x{fh}")

    # Init CV modules (same as realtime_main.py)
    model_path = _resolve_model_path()
    tracker = Tracker(model_path, [0, 1, 2, 3], verbose=False)
    cam_est = CameraMovementEstimator(frame, [0, 1, 2, 3], verbose=False)
    t_assign = TeamAssigner()
    b_assign = PlayerBallAssigner()
    ev_detect = EventDetector(fw, fh) if "events" in features else None
    # Auto-calibrate pixel_to_meter_scale from first frame (#3)
    calibrator = PitchCalibrator()
    px_to_m_scale = calibrator.calibrate_from_frame(frame)
    print(f"[CV] Calibrated pixel_to_meter_scale = {px_to_m_scale:.5f} (calibrated={calibrator.is_calibrated})")

    speed_est = SpeedEstimator(fps=fps, pixel_to_meter_scale=px_to_m_scale)
    tact_analyzer = TacticalAnalyzer()
    ocr_reader = JerseyOCRReader(min_bbox_height=150)

    teams_assigned = False
    frame_count = 0
    last_post_time = 0.0
    # Outbound control knobs
    POST_INTERVAL = float(os.getenv("CV_POST_INTERVAL_SEC", "2.0"))  # seconds between stats posts
    FRAME_MOD = int(os.getenv("CV_PROCESS_EVERY_N_FRAMES", "2"))  # process every Nth frame (>=1)
    FRAME_MOD = max(1, FRAME_MOD)
    reconnect_attempts = 0
    # Reconnect/backoff controls (reduce bursty outbound when stream is flaky)
    MAX_RECONNECT = int(os.getenv("CV_HLS_MAX_RECONNECT", "6"))
    BACKOFF_BASE_SEC = float(os.getenv("CV_HLS_RECONNECT_BASE_SLEEP", "2.0"))
    BACKOFF_MAX_SEC = float(os.getenv("CV_HLS_RECONNECT_MAX_SLEEP", "30.0"))

    print(f"[CV] Pipeline running for stream={stream_id} ({fw}x{fh}, {fps:.1f}fps, ffmpeg={is_ffmpeg_pipe})")

    while not stop_event.is_set():
        # ── Read frame (FFmpeg pipe or cv2) ──
        read_ok = False
        if is_ffmpeg_pipe and proc:
            raw = proc.stdout.read(frame_size)
            if len(raw) >= frame_size:
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(fh, fw, 3)
                read_ok = True
        elif cap is not None:
            ret, frame = cap.read()
            read_ok = ret

        if not read_ok:
            reconnect_attempts += 1
            if reconnect_attempts > MAX_RECONNECT:
                print(f"[CV] Max reconnect attempts reached for stream={stream_id}. Stopping session to avoid outbound flood.")
                break

            # Exponential backoff + jitter
            sleep_s = min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * (2 ** (reconnect_attempts - 1)))
            sleep_s = sleep_s + random.uniform(0, min(1.0, sleep_s * 0.1))
            print(
                f"[CV] Reconnecting stream={stream_id} ({reconnect_attempts}/{MAX_RECONNECT}) "
                f"after {sleep_s:.1f}s (host={hls_host or '?'})..."
            )
            # Cleanup old resources
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                proc = None
            elif cap is not None:
                cap.release()
            time.sleep(sleep_s)
            # Reconnect using FFmpeg for HLS, cv2 otherwise
            if is_hls:
                new_cap, new_meta = _get_ffmpeg_reader(hls_url)
                if new_cap is not None:
                    cap = new_cap
                    is_ffmpeg_pipe = getattr(cap, "_is_ffmpeg_pipe", False)
                    proc = getattr(cap, "_ffmpeg_proc", None) if is_ffmpeg_pipe else None
                    if new_meta:
                        fw = new_meta.get("width", fw)
                        fh = new_meta.get("height", fh)
                        frame_size = fw * fh * 3
            else:
                cap = cv2.VideoCapture(hls_url)
            continue

        reconnect_attempts = 0
        frame_count += 1

        # Update session frame count
        with _lock:
            if stream_id in active_sessions:
                active_sessions[stream_id]["frame_count"] = frame_count

        # Process only every Nth frame to reduce CPU + outbound callback volume
        if frame_count % FRAME_MOD != 0:
            continue

        try:
            # ── Periodic recalibration (#3) ──
            new_scale = calibrator.recalibrate_if_needed(frame, frame_count, interval=3000)
            if new_scale != speed_est.pixel_to_meter_scale:
                speed_est.update_scale(new_scale)

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
            
            # Tactics & Passes & Heatmap
            tact_analyzer.update_heatmap(tracks)
            pass_event = tact_analyzer.detect_passes(tracks, b_assign.player_has_ball, frame_count)
            tactics_info = tact_analyzer.analyze_tactics(tracks)

            # OCR Jersey Numbers (runs only on large bboxes)
            jersey_map = ocr_reader.process_frame(frame, tracks, frame_count)

            if now - last_post_time >= POST_INTERVAL:
                last_post_time = now

                stats = _collect_stats(tracks, b_assign, ev_detect, features, frame_count, top_speeds, pass_event, tactics_info, jersey_map, fps)

                try:
                    r = requests.post(
                        callback_url,
                        json={"stream_id": stream_id, "frame": frame_count, **stats},
                        timeout=2,
                    )
                    if frame_count <= 6 or r.status_code >= 400:
                        print(f"[CV] Callback → {callback_url} HTTP {r.status_code}")
                except Exception as e:
                    print(f"[CV] POST callback failed ({callback_url}): {e}")

        except Exception as e:
            print(f"[CV] Frame {frame_count} error: {e}")
            continue

    # Cleanup resources
    if is_ffmpeg_pipe and proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    try:
        cap.release()
    except Exception:
        pass
    _cleanup_session(stream_id)
    print(f"[CV] Pipeline stopped for stream={stream_id} (processed {frame_count} frames)")


def _collect_stats(
    tracks: Dict,
    b_assign: PlayerBallAssigner,
    ev_detect: Optional[EventDetector],
    features: List[str],
    frame_count: int,
    top_speeds: list = None,
    pass_event: dict = None,
    tactics_info: dict = None,
    jersey_map: dict = None,
    fps: float = 30.0,
) -> Dict:
    """Collect stats from current CV state."""
    stats: Dict = {}
    # Match elapsed time in seconds (#10 — accurate timing instead of wall clock)
    stats["match_elapsed"] = round(frame_count / fps, 1)
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

    # ── Ball + Possessor (for minimap) ──
    # Provide normalized 0..1 coordinates from observed player field bounds (same bounds TacticalAnalyzer uses).
    try:
        ball = tracks.get("ball", {})
        if ball and 1 in ball:
            bx1, by1, bx2, by2 = ball[1].get("bbox", [0, 0, 0, 0])
            bx = float((bx1 + bx2) / 2.0)
            by = float((by1 + by2) / 2.0)

            xs = [p["position_adjusted"][0] for p in players.values() if "position_adjusted" in p]
            ys = [p["position_adjusted"][1] for p in players.values() if "position_adjusted" in p]
            if xs and ys:
                min_x, max_x = float(min(xs)), float(max(xs))
                min_y, max_y = float(min(ys)), float(max(ys))
                rx = max(1.0, max_x - min_x)
                ry = max(1.0, max_y - min_y)
                stats["ball"] = {
                    "x": round((bx - min_x) / rx, 4),
                    "y": round((by - min_y) / ry, 4),
                    "visible": True,
                }
            else:
                stats["ball"] = {"visible": True}
        else:
            stats["ball"] = {"visible": False}

        if getattr(b_assign, "player_has_ball", None) is not None:
            pid = int(b_assign.player_has_ball)
            pdata = players.get(pid, {}) if isinstance(players, dict) else {}
            stats["possessor"] = {
                "player_id": pid,
                "team": int(pdata.get("team", 0) or 0),
            }
    except Exception:
        # Never break callback due to minimap extras
        pass

    # ── Events ──
    if "events" in features and ev_detect is not None:
        event = ev_detect.check_events(tracks, frame_count)
        if event:
            stats["event"] = event
            
        if pass_event:
            stats["event"] = pass_event

    # ── Speeds & Distance ──
    if top_speeds:
        stats["speeds"] = top_speeds

    # ── Tactics & Space Control ──
    if tactics_info:
        stats["tactics"] = tactics_info

    # ── Jersey Number Mapping (OCR) ──
    if jersey_map:
        # Convert int keys to strings for JSON serialization
        stats["jerseys"] = {str(k): v for k, v in jersey_map.items()}

    return stats


def _cleanup_session(stream_id: str):
    """Remove session from active registry."""
    with _lock:
        active_sessions.pop(stream_id, None)


# ─────────────────────────────────────────────
# FFmpeg availability check
# ─────────────────────────────────────────────
def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg binary is available in PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─────────────────────────────────────────────
# Direct cv2 HLS reader (fallback when FFmpeg binary unavailable)
# ─────────────────────────────────────────────
def _open_hls_via_cv2(video_url: str) -> Tuple[Optional[cv2.VideoCapture], Optional[dict]]:
    """
    Open an HLS stream directly via cv2.VideoCapture.
    OpenCV's built-in FFmpeg backend can read .m3u8 URLs.
    Sets OPENCV_FFMPEG_CAPTURE_OPTIONS for referer header if needed.
    """
    import os
    # Set referer header for BunnyCDN streams
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'referer;https://booca.online/'
    print(f"[CV] Opening HLS via cv2.VideoCapture: {video_url}")
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        meta = _get_cv2_metadata(cap)
        print(f"[CV] cv2.VideoCapture opened successfully: {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps")
        return cap, meta
    # Try without explicit backend
    print(f"[CV] CAP_FFMPEG failed, trying default backend...")
    cap = cv2.VideoCapture(video_url)
    if cap.isOpened():
        meta = _get_cv2_metadata(cap)
        print(f"[CV] Default backend opened successfully: {meta['width']}x{meta['height']}, {meta['fps']:.1f}fps")
        return cap, meta
    print(f"[CV] ERROR: cv2.VideoCapture cannot open HLS: {video_url}")
    return None, None


# ─────────────────────────────────────────────
# FFmpeg Frame Reader (supports HLS + MP4)
# ─────────────────────────────────────────────
def _get_ffmpeg_reader(video_url: str) -> Tuple[Optional[cv2.VideoCapture], Optional[dict]]:
    """
    Create a VideoCapture using FFmpeg pipe for HLS/M3U8, or direct cv2 for MP4.
    Returns (cap, metadata) where metadata has fps, width, height, total_frames.
    Falls back to plain cv2.VideoCapture if FFmpeg pipe fails.
    """
    from cv_service.hls_proxy import get_hls_proxy
    is_hls = video_url.endswith(".m3u8") or ".m3u8?" in video_url

    if is_hls:
        orig_video_url = video_url
        # Pass to local proxy to append Referer to all manifest and TS segment requests
        proxy_url = get_hls_proxy().get_proxy_url(video_url)
        headers_str = "Referer: https://booca.online/\r\nUser-Agent: Mozilla/5.0\r\n"
        ffmpeg_cmd = [
            "ffmpeg", "-headers", headers_str,
            "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5", "-i", proxy_url,
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
                "-headers", headers_str,
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                "-of", "default=noprint_wrappers=1", proxy_url
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
            cap = cv2.VideoCapture(orig_video_url)  # dummy, will be overridden
            cap._ffmpeg_proc = proc  # type: ignore
            cap._ffmpeg_meta = metadata  # type: ignore
            cap._is_ffmpeg_pipe = True  # type: ignore
            return cap, metadata
        except Exception as e:
            print(f"[CV] FFmpeg pipe failed for HLS: {e}, falling back to cv2")
            return _open_hls_via_cv2(orig_video_url)
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
        raw = _read_exact_with_timeout(
            proc.stdout,
            meta["width"] * meta["height"] * 3,
            timeout_sec=float(os.getenv("CV_FFMPEG_FIRST_FRAME_TIMEOUT_SEC", "12")),
        )
        if len(raw) < meta["width"] * meta["height"] * 3:
            print(f"[CV-VOD] ERROR: Cannot read first frame from FFmpeg pipe")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
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
    model_path = _resolve_model_path()
    tracker = Tracker(model_path, [0, 1, 2, 3], verbose=False)
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
