"""
Highlight Worker — Standalone Python service for video clip extraction.

Receives requests from BOOCA Backend, downloads VOD segment,
extracts highlight clips using FFmpeg, uploads to Cloudinary, and POSTs CDN URLs back.

Run as: python -m cv_service.highlight_worker
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import subprocess
import os
import time
import requests
import threading
import uuid

# Load .env for CLOUDINARY credentials
from dotenv import load_dotenv
load_dotenv()

# Cloudinary SDK
import cloudinary
import cloudinary.uploader
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
)

app = FastAPI(
    title="Highlight Worker",
    description="FFmpeg-based video clip extraction service for BOOCA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temp directory for downloaded VODs and extracted clips
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_highlights")
os.makedirs(TEMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class HighlightEvent(BaseModel):
    event: str
    timestamp: float  # seconds from start of video
    frame: int = 0
    scoringTeam: Optional[int] = None


class ExtractRequest(BaseModel):
    vod_id: str
    video_url: str  # URL to the full VOD video
    events: List[HighlightEvent]
    callback_url: str  # BOOCA BE endpoint to POST results
    clip_padding: int = 15  # seconds before/after event


class ClipResult(BaseModel):
    event: str
    timestamp: float
    clip_path: str
    clip_url: str
    duration: float


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    # Check ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        ffmpeg_ok = result.returncode == 0
    except Exception:
        ffmpeg_ok = False

    return {
        "status": "ok",
        "ffmpeg_available": ffmpeg_ok,
        "temp_dir": TEMP_DIR,
    }


@app.post("/highlights/extract")
async def extract_highlights(req: ExtractRequest):
    """Trigger highlight clip extraction in a background thread."""
    if not req.events:
        raise HTTPException(status_code=400, detail="No events provided")

    # Run in background
    thread = threading.Thread(
        target=_process_highlights,
        args=(req.vod_id, req.video_url, req.events, req.callback_url, req.clip_padding),
        daemon=True,
    )
    thread.start()

    return {
        "status": "processing",
        "vod_id": req.vod_id,
        "event_count": len(req.events),
    }


# ─────────────────────────────────────────────
# Background Processing
# ─────────────────────────────────────────────
def _process_highlights(
    vod_id: str,
    video_url: str,
    events: List[HighlightEvent],
    callback_url: str,
    clip_padding: int,
):
    print(f"[Highlight] Starting extraction for VOD={vod_id}, {len(events)} events")
    results = []

    for i, event in enumerate(events):
        try:
            clip_path = _extract_clip(
                vod_id,
                video_url,
                event.timestamp,
                clip_padding,
                event.event,
                i,
            )
            if clip_path:
                # ── Upload to Cloudinary ──
                clip_url = _upload_to_cloudinary(clip_path, vod_id, event.event, i)
                results.append({
                    "event": event.event,
                    "timestamp": event.timestamp,
                    "scoringTeam": event.scoringTeam,
                    "clip_path": clip_path,
                    "clip_url": clip_url,
                    "duration": clip_padding * 2,
                })
                print(f"[Highlight] Clip {i+1}/{len(events)}: {event.event} @ {event.timestamp}s → {clip_url}")

                # Cleanup local temp file after successful upload
                _safe_delete(clip_path)

        except Exception as e:
            print(f"[Highlight] Error extracting clip {i}: {e}")

    # POST results back to BOOCA Backend
    payload = {
        "vod_id": vod_id,
        "status": "completed" if results else "failed",
        "highlights": results,
        "error": None if results else "No clips could be extracted",
    }

    try:
        requests.post(callback_url, json=payload, timeout=15)
        print(f"[Highlight] Results posted for VOD={vod_id}: {len(results)} clips")
    except Exception as e:
        print(f"[Highlight] Failed to POST results: {e}")

    # Cleanup temp files (optional — keep for a while for debugging)
    # _cleanup_temp(vod_id)


def _extract_clip(
    vod_id: str,
    video_url: str,
    event_timestamp: float,
    padding: int,
    event_name: str,
    index: int,
) -> Optional[str]:
    """
    Use FFmpeg to extract a clip from the video.
    FFmpeg can read directly from HTTP URL — no need to download the full video!
    """
    start_time = max(0, event_timestamp - padding)
    duration = padding * 2

    # Output filename
    clip_filename = f"{vod_id}_{event_name}_{index}_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(TEMP_DIR, clip_filename)

    # FFmpeg command — read from URL, seek, extract clip
    # Using -ss before -i for fast seeking on HTTP sources
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-ss", str(start_time),
        "-i", video_url,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]

    print(f"[Highlight] FFmpeg: extracting {duration}s from {start_time}s")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 min max per clip
        )

        if result.returncode != 0:
            print(f"[Highlight] FFmpeg error: {result.stderr[-500:]}")
            return None

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None

    except subprocess.TimeoutExpired:
        print(f"[Highlight] FFmpeg timeout for clip {index}")
        return None


def _cleanup_temp(vod_id: str):
    """Remove temp files for a completed VOD."""
    for f in os.listdir(TEMP_DIR):
        if f.startswith(vod_id):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except Exception:
                pass


# ─────────────────────────────────────────────
# Cloudinary Upload
# ─────────────────────────────────────────────
def _upload_to_cloudinary(
    local_path: str,
    vod_id: str,
    event_name: str,
    index: int,
) -> str:
    """
    Upload extracted clip to Cloudinary and return the CDN URL.
    Falls back to returning the local path if Cloudinary credentials are not configured.
    """
    has_creds = all([
        os.getenv("CLOUDINARY_CLOUD_NAME"),
        os.getenv("CLOUDINARY_API_KEY"),
        os.getenv("CLOUDINARY_API_SECRET"),
    ])

    if not has_creds:
        # No Cloudinary configured — return local path (backwards compat)
        print("[Highlight] Cloudinary not configured, using local path as clip_url")
        return local_path

    try:
        folder = f"booca/highlights/{vod_id[:12]}"
        public_id = f"{vod_id}_{event_name}_{index}"

        result = cloudinary.uploader.upload(
            local_path,
            resource_type="video",
            folder=folder,
            public_id=public_id,
            overwrite=True,
            chunk_size=60_000_000,  # 60MB chunk for large video files
        )
        cdn_url = result.get("secure_url", local_path)
        print(f"[Highlight] Uploaded to Cloudinary: {cdn_url}")
        return cdn_url

    except Exception as e:
        print(f"[Highlight] Cloudinary upload failed: {e} — falling back to local path")
        return local_path


def _safe_delete(path: str):
    """Safely delete a temp file, logging any errors."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[Highlight] Failed to delete temp file {path}: {e}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "cv_service.highlight_worker:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
