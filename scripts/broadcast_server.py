#!/usr/bin/env python3
"""
MatchVision Broadcast Server
Receives annotated frames from the analysis pipeline and:
1. Serves them via MJPEG to browsers (open http://localhost:8502)
2. Pushes to RTMP endpoint (YouTube Live, Twitch, Facebook Live, custom RTMP server)

Architecture:
    frontend/app.py (or realtime_main.py)
          │
          │ POST /push (HTTP + JPEG)
          ▼
    broadcast_server.py
          │
          ├──▶ MJPEG stream  →  Browser (http://localhost:8502/mjpeg)
          └──▶ RTMP pusher  →  YouTube/Twitch/Facebook

Usage:
    # Step 1: Start broadcast server (separate terminal)
    python scripts/broadcast_server.py

    # Step 2: Enable broadcast in frontend/app.py sidebar, fill RTMP URL if needed

    # With RTMP (YouTube Live):
    python scripts/broadcast_server.py --rtmp rtmp://a.rtmp.youtube.com/live2/YOUR_STREAM_KEY

    # Custom port:
    python scripts/broadcast_server.py --port 8502 --rtmp "your rtmp url"
"""
import argparse
import base64
import queue
import threading
import time
import cv2
import numpy as np
import sys


class BroadcastServer:
    """Broadcast server that receives frames and streams them via MJPEG + RTMP."""

    def __init__(self, rtmp_url=None, port=8502, fps=30, verbose=True):
        self.rtmp_url = rtmp_url
        self.port = port
        self.fps = fps
        self.verbose = verbose
        self._queue = queue.Queue(maxsize=3)
        self._rtmp_writer = None
        self._last_frame = None
        self._running = False
        self._rtmp_lock = threading.Lock()
        self._last_rtmp_push = 0

    def _ensure_rtmp_writer(self, frame):
        """Lazily create RTMP writer on first frame."""
        if self._rtmp_writer is not None:
            return True
        if not self.rtmp_url:
            return False
        try:
            h, w = frame.shape[:2]
            # FLV container works universally; use 'mp4' only if target supports it
            fourcc = cv2.VideoWriter_fourcc(*'flv')
            writer = cv2.VideoWriter(
                self.rtmp_url,
                cv2.CAP_FFMPEG,
                fourcc,
                self.fps,
                (w, h)
            )
            if writer.isOpened():
                self._rtmp_writer = writer
                self._log(f"RTMP connected: {self.rtmp_url} ({w}x{h} @ {self.fps}fps)")
                return True
            else:
                self._log(f"RTMP open failed — check URL and network")
                return False
        except Exception as e:
            self._log(f"RTMP init error: {e}")
            return False

    def _push_rtmp(self, frame):
        """Push frame to RTMP. Thread-safe, skips frames if behind."""
        if not self.rtmp_url:
            return

        with self._rtmp_lock:
            now = time.time()
            # Throttle: skip frames to stay near target FPS
            min_interval = 1.0 / (self.fps + 5)
            if now - self._last_rtmp_push < min_interval and self._rtmp_writer is not None:
                return
            self._last_rtmp_push = now

            try:
                if self._ensure_rtmp_writer(frame):
                    self._rtmp_writer.write(frame)
                else:
                    self._rtmp_writer = None
            except Exception as e:
                self._log(f"RTMP write error: {e}")
                self._rtmp_writer = None

    def push_frame(self, frame):
        """Push a frame into the broadcast pipeline (call from main analysis loop)."""
        if frame is None or frame.size == 0:
            return
        self._last_frame = frame.copy()

        # Enqueue for MJPEG viewers
        try:
            self._queue.put_nowait(frame)
        except queue.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except queue.QueueFull:
                pass

        # Push to RTMP
        self._push_rtmp(frame)

    def get_queue(self):
        return self._queue

    def get_last_frame(self):
        return self._last_frame

    def _log(self, msg):
        if self.verbose:
            print(f"[Broadcast] {msg}", flush=True)

    # ─── FastAPI app ───────────────────────────────────────────────

    def create_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, StreamingResponse

        app = FastAPI(title="MatchVision Broadcast")

        rtmp_status = "Enabled" if self.rtmp_url else "Disabled (MJPEG only)"

        @app.get("/")
        async def index():
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MatchVision Broadcast</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center;
                           background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
                    h1 {{ color: #58a6ff; margin-bottom: 5px; }}
                    .subtitle {{ color: #8b949e; margin-bottom: 20px; }}
                    img {{ max-width: 100%; width: 960px; border: 2px solid #30363d;
                           border-radius: 8px; background: #161b22; }}
                    .info {{ background: #161b22; padding: 15px; border-radius: 8px;
                             margin: 20px auto; max-width: 600px; text-align: left; }}
                    .info p {{ margin: 8px 0; }}
                    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                              font-size: 12px; font-weight: bold; }}
                    .badge-live {{ background: #238636; color: #fff; }}
                    .badge-wait {{ background: #6e7681; color: #fff; }}
                    #status {{ transition: color 0.3s; }}
                </style>
            </head>
            <body>
                <h1>MatchVision Live</h1>
                <p class="subtitle">Football AI Analysis Stream</p>
                <img src="/mjpeg" alt="Live stream" />
                <div class="info">
                    <p><strong>MJPEG:</strong> <a href="/mjpeg" style="color:#58a6ff">/mjpeg</a></p>
                    <p><strong>RTMP:</strong> {rtmp_status}</p>
                    <p><strong>Status:</strong> <span id="status" class="badge badge-wait">Waiting...</span></p>
                </div>
                <script>
                    function update() {{
                        fetch('/health').then(r => r.json()).then(d => {{
                            var el = document.getElementById('status');
                            if (d.has_frame) {{
                                el.textContent = '● LIVE';
                                el.className = 'badge badge-live';
                            }} else {{
                                el.textContent = '○ No frames';
                                el.className = 'badge badge-wait';
                            }}
                        }}).catch(() => {{}});
                    }}
                    update();
                    setInterval(update, 2000);
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html, media_type="text/html")

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "has_frame": self._queue.qsize() > 0 or self._last_frame is not None,
                "rtmp": "connected" if self._rtmp_writer is not None else ("disabled" if not self.rtmp_url else "disconnected"),
                "queue_size": self._queue.qsize()
            }

        @app.get("/mjpeg")
        async def mjpeg_stream():
            """MJPEG streaming endpoint — works in any browser without plugins."""
            async def generate():
                while True:
                    try:
                        frame = self._queue.get(timeout=3.0)
                    except queue.Empty:
                        continue

                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame",
                                     headers={"Cache-Control": "no-cache"})

        @app.post("/push")
        async def push_frame(data: dict):
            """
            Receive a JPEG frame from the analysis pipeline and broadcast it.
            Expected body: {{ "image": "<base64 jpeg>" }}
            """
            try:
                if "image" not in data:
                    raise HTTPException(status_code=400, detail="Missing 'image' field (base64 JPEG)")
                img_bytes = base64.b64decode(data["image"])
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise HTTPException(status_code=400, detail="Invalid image data")
                self.push_frame(frame)
                return {"status": "ok", "queue": self._queue.qsize()}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app


# ─── Entry point ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MatchVision Broadcast Server")
    parser.add_argument("--rtmp", type=str, default="",
                        help="RTMP URL (e.g. rtmp://a.rtmp.youtube.com/live2/YOUR_KEY)")
    parser.add_argument("--port", type=int, default=8502, help="Server port (default: 8502)")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS for RTMP (default: 30)")
    args = parser.parse_args()

    server = BroadcastServer(rtmp_url=args.rtmp or None, port=args.port, fps=args.fps)

    if args.rtmp:
        print(f"[Broadcast] RTMP target: {args.rtmp}")
    else:
        print("[Broadcast] RTMP: disabled (MJPEG only)")

    print(f"[Broadcast] Server: http://localhost:{args.port}")
    print(f"[Broadcast] MJPEG:  http://localhost:{args.port}/mjpeg")
    print(f"[Broadcast] Health: http://localhost:{args.port}/health")
    print("[Broadcast] Push:   POST /push with {\"image\": \"<base64>\"}")
    print("[Broadcast] Press Ctrl+C to stop")

    app = server.create_app()
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("[Broadcast] Stopped")


if __name__ == "__main__":
    main()
