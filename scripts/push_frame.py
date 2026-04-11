#!/usr/bin/env python3
"""
BroadcastPusher — lightweight HTTP client that POSTs annotated frames to the broadcast server.

Usage in frontend/app.py:
    from scripts.push_frame import BroadcastPusher
    pusher = BroadcastPusher("http://localhost:8502")
    pusher.push(frame)   # non-blocking, fire-and-forget

Usage in realtime_main.py:
    from scripts.push_frame import BroadcastPusher
    pusher = BroadcastPusher("http://localhost:8502")
    pusher.push(annotated_frame)

Requirements:
    1. Start broadcast server first:  python scripts/broadcast_server.py [--rtmp URL]
    2. Enable "Enable Broadcast Stream" in the frontend sidebar
    3. Optionally provide RTMP URL for YouTube/Twitch/Facebook Live
"""
import queue
import threading
import time
import base64
import cv2
import numpy as np

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class BroadcastPusher:
    """
    Non-blocking frame pusher for the MatchVision broadcast pipeline.

    Pushes JPEG frames to broadcast_server.py via HTTP POST /push.
    The server handles MJPEG streaming and RTMP forwarding internally.
    """

    def __init__(self, url="http://localhost:8502", enabled=True):
        self.url = url.rstrip("/")
        self.enabled = enabled
        self._connected = False
        self._last_connect_attempt = 0
        self._connect_interval = 5.0  # retry connection every 5s

        # For same-process MJPEG viewer (no server needed)
        self._queue = queue.Queue(maxsize=2)

    def _check_connection(self):
        """Ping the server to check if it's running."""
        if not _HAS_REQUESTS:
            return False
        try:
            r = requests.get(f"{self.url}/health", timeout=1.0)
            self._connected = r.status_code == 200
        except Exception:
            self._connected = False
        return self._connected

    def push(self, frame: np.ndarray):
        """
        Push an annotated frame to the broadcast server.
        Non-blocking — drops frames silently if server is unreachable.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return

        # Enqueue locally for same-process consumers (e.g., MJPEG in same app)
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame.copy())
            except queue.Full:
                pass

        if not _HAS_REQUESTS:
            return

        # Skip if not connected (avoid flooding)
        now = time.time()
        if not self._connected and (now - self._last_connect_attempt) < self._connect_interval:
            return

        # Thread the HTTP POST to keep it non-blocking
        def _send():
            try:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    return
                b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
                resp = requests.post(
                    f"{self.url}/push",
                    json={"image": b64},
                    timeout=2.0
                )
                self._connected = resp.status_code == 200
            except Exception:
                self._connected = False
                self._last_connect_attempt = time.time()

        threading.Thread(target=_send, daemon=True).start()

    def is_connected(self) -> bool:
        """Check if broadcast server is reachable."""
        if not self.enabled:
            return False
        return self._check_connection()

    def is_enabled(self) -> bool:
        return self.enabled

    def get_queue(self) -> queue.Queue:
        """Return local queue for same-process MJPEG consumers."""
        return self._queue
