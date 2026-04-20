import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import time
import numpy as np
import os
import sys
import base64

# Add parent directory to path
sys.path.append(os.path.abspath("."))

# Load .env file so BOOCA_API_URL etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import threading
import queue

from utils import options, read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator
from cv_service.speed_estimator import SpeedEstimator
from cv_service.tactical_analyzer import TacticalAnalyzer
from cv_service.event_detector import EventDetector
from cv_service.minimap import render_minimap_rgb
from cv_service.match_sim import render_match_sim_rgb

# --- Proxy Server for BunnyCDN/HLS streams ---
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        target_url = self.path[1:]
        headers = {
            "Referer": "https://booca.online/",
            "User-Agent": "Mozilla/5.0",
        }
        try:
            resp = requests.get(target_url, headers=headers, timeout=10)
            self.send_response(resp.status_code)
            self.send_header("Content-Length", str(len(resp.content)))
            self.send_header("Content-Type", resp.headers.get("Content-Type", "video/MP2T"))
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception:
            self.send_error(500)

class SimpleProxy:
    def __init__(self):
        # find open port
        sock = socket.socket()
        sock.bind(('', 0))
        self.port = sock.getsockname()[1]
        sock.close()
        
        self.server = HTTPServer(('127.0.0.1', self.port), ProxyHTTPRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.5)
        
    def get_proxy_url(self, target_url):
        if "127.0.0.1" in target_url or "localhost" in target_url:
            return target_url # don't proxy locals
        return f"http://127.0.0.1:{self.port}/{target_url}"

@st.cache_resource
def get_hls_proxy():
    return SimpleProxy()
# ---------------------------------------------

# Broadcast frame pusher (non-blocking HTTP POST to broadcast_server.py)
try:
    from scripts.push_frame import BroadcastPusher
    _HAS_PUSHER = True
except Exception:
    _HAS_PUSHER = False


def get_broadcast_server_url():
    """Return the broadcast server URL based on user settings."""
    # Default broadcast server port
    return "http://localhost:8502"


def get_custom_video_player_html(video_path, theme="dark"):
    """Generate a custom HTML5 video player with YouTube-like controls.
    Supports: zoom in/out, volume, fullscreen, progress bar, dark/light theme.
    """
    # Read video file and encode to base64 for embedding
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode()
    
    is_dark = theme == "dark"
    bg_color = "#0e1117" if is_dark else "#ffffff"
    text_color = "#fafafa" if is_dark else "#1a1a1a"
    control_bg = "rgba(0,0,0,0.85)" if is_dark else "rgba(30,30,30,0.9)"
    slider_track = "#555" if is_dark else "#bbb"
    accent_color = "#ff4b4b"
    hover_bg = "rgba(255,255,255,0.1)" if is_dark else "rgba(255,255,255,0.2)"
    container_bg = "#1a1a2e" if is_dark else "#f0f0f5"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: {bg_color}; font-family: 'Segoe UI', Roboto, sans-serif; }}
        
        .player-wrapper {{
            position: relative;
            max-width: 100%;
            background: {container_bg};
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        
        .video-container {{
            position: relative;
            width: 100%;
            overflow: hidden;
            cursor: pointer;
            background: #000;
        }}
        
        video {{
            width: 100%;
            display: block;
            transform-origin: center center;
            transition: transform 0.3s ease;
        }}
        
        /* Controls overlay */
        .controls {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, {control_bg});
            padding: 40px 12px 10px 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 10;
        }}
        .player-wrapper:hover .controls {{
            opacity: 1;
        }}
        
        /* Progress bar */
        .progress-container {{
            width: 100%;
            height: 4px;
            background: {slider_track};
            border-radius: 2px;
            cursor: pointer;
            margin-bottom: 8px;
            position: relative;
            transition: height 0.15s;
        }}
        .progress-container:hover {{
            height: 7px;
        }}
        .progress-bar {{
            height: 100%;
            background: {accent_color};
            border-radius: 2px;
            width: 0%;
            position: relative;
        }}
        .progress-bar::after {{
            content: '';
            position: absolute;
            right: -6px;
            top: 50%;
            transform: translateY(-50%) scale(0);
            width: 13px;
            height: 13px;
            background: {accent_color};
            border-radius: 50%;
            transition: transform 0.15s;
        }}
        .progress-container:hover .progress-bar::after {{
            transform: translateY(-50%) scale(1);
        }}
        
        /* Control buttons row */
        .controls-row {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .ctrl-btn {{
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 6px 8px;
            border-radius: 6px;
            font-size: 18px;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 36px;
            height: 36px;
        }}
        .ctrl-btn:hover {{
            background: {hover_bg};
        }}
        
        /* Volume section */
        .volume-section {{
            display: flex;
            align-items: center;
            gap: 4px;
        }}
        .volume-slider {{
            width: 0;
            transition: width 0.2s, opacity 0.2s;
            opacity: 0;
            overflow: hidden;
        }}
        .volume-section:hover .volume-slider {{
            width: 70px;
            opacity: 1;
        }}
        input[type="range"] {{
            -webkit-appearance: none;
            height: 4px;
            background: {slider_track};
            border-radius: 2px;
            outline: none;
            cursor: pointer;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        /* Time display */
        .time-display {{
            color: white;
            font-size: 13px;
            margin-left: 8px;
            font-variant-numeric: tabular-nums;
            user-select: none;
        }}
        
        .spacer {{ flex: 1; }}
        
        /* Zoom display badge */
        .zoom-badge {{
            color: white;
            font-size: 12px;
            background: rgba(255,75,75,0.7);
            padding: 2px 8px;
            border-radius: 10px;
            user-select: none;
            display: none;
        }}
        .zoom-badge.visible {{
            display: inline-block;
        }}
        
        /* Center play overlay */
        .play-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 60px;
            color: rgba(255,255,255,0.8);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 5;
        }}
        .play-overlay.show {{
            opacity: 1;
        }}
    </style>
    </head>
    <body>
    <div class="player-wrapper">
        <div class="video-container" id="videoContainer">
            <video id="video" preload="metadata">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <div class="play-overlay" id="playOverlay">▶</div>
        </div>
        
        <div class="controls">
            <div class="progress-container" id="progressContainer">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            <div class="controls-row">
                <button class="ctrl-btn" id="playBtn" title="Play/Pause">▶</button>
                
                <div class="volume-section">
                    <button class="ctrl-btn" id="muteBtn" title="Mute">🔊</button>
                    <div class="volume-slider">
                        <input type="range" id="volumeSlider" min="0" max="100" value="100">
                    </div>
                </div>
                
                <span class="time-display" id="timeDisplay">0:00 / 0:00</span>
                
                <div class="spacer"></div>
                
                <span class="zoom-badge" id="zoomBadge">1.0x</span>
                <button class="ctrl-btn" id="zoomOut" title="Zoom Out">➖</button>
                <button class="ctrl-btn" id="zoomIn" title="Zoom In">➕</button>
                <button class="ctrl-btn" id="zoomReset" title="Reset Zoom">↺</button>
                <button class="ctrl-btn" id="fullscreenBtn" title="Fullscreen">⛶</button>
            </div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        const playBtn = document.getElementById('playBtn');
        const muteBtn = document.getElementById('muteBtn');
        const volumeSlider = document.getElementById('volumeSlider');
        const progressBar = document.getElementById('progressBar');
        const progressContainer = document.getElementById('progressContainer');
        const timeDisplay = document.getElementById('timeDisplay');
        const zoomIn = document.getElementById('zoomIn');
        const zoomOut = document.getElementById('zoomOut');
        const zoomReset = document.getElementById('zoomReset');
        const zoomBadge = document.getElementById('zoomBadge');
        const fullscreenBtn = document.getElementById('fullscreenBtn');
        const videoContainer = document.getElementById('videoContainer');
        const playOverlay = document.getElementById('playOverlay');
        const playerWrapper = document.querySelector('.player-wrapper');
        
        let currentZoom = 1.0;
        const ZOOM_STEP = 0.25;
        const MAX_ZOOM = 3.0;
        const MIN_ZOOM = 0.5;
        
        // Format time
        function formatTime(s) {{
            if (isNaN(s)) return '0:00';
            const m = Math.floor(s / 60);
            const sec = Math.floor(s % 60);
            return m + ':' + (sec < 10 ? '0' : '') + sec;
        }}
        
        // Play / Pause
        function togglePlay() {{
            if (video.paused) {{
                video.play();
                playBtn.textContent = '⏸';
                playOverlay.classList.remove('show');
            }} else {{
                video.pause();
                playBtn.textContent = '▶';
                playOverlay.classList.add('show');
            }}
        }}
        playBtn.addEventListener('click', togglePlay);
        videoContainer.addEventListener('click', togglePlay);
        
        // Volume
        volumeSlider.addEventListener('input', function() {{
            video.volume = this.value / 100;
            updateMuteIcon();
        }});
        
        muteBtn.addEventListener('click', function() {{
            video.muted = !video.muted;
            updateMuteIcon();
        }});
        
        function updateMuteIcon() {{
            if (video.muted || video.volume === 0) muteBtn.textContent = '🔇';
            else if (video.volume < 0.5) muteBtn.textContent = '🔉';
            else muteBtn.textContent = '🔊';
        }}
        
        // Progress
        video.addEventListener('timeupdate', function() {{
            const pct = (video.currentTime / video.duration) * 100;
            progressBar.style.width = pct + '%';
            timeDisplay.textContent = formatTime(video.currentTime) + ' / ' + formatTime(video.duration);
        }});
        
        progressContainer.addEventListener('click', function(e) {{
            const rect = this.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            video.currentTime = pct * video.duration;
        }});
        
        // Zoom
        function setZoom(level) {{
            currentZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, level));
            video.style.transform = 'scale(' + currentZoom + ')';
            zoomBadge.textContent = currentZoom.toFixed(1) + 'x';
            zoomBadge.classList.toggle('visible', currentZoom !== 1.0);
        }}
        zoomIn.addEventListener('click', function(e) {{ e.stopPropagation(); setZoom(currentZoom + ZOOM_STEP); }});
        zoomOut.addEventListener('click', function(e) {{ e.stopPropagation(); setZoom(currentZoom - ZOOM_STEP); }});
        zoomReset.addEventListener('click', function(e) {{ e.stopPropagation(); setZoom(1.0); }});
        
        // Fullscreen
        fullscreenBtn.addEventListener('click', function(e) {{
            e.stopPropagation();
            if (!document.fullscreenElement) {{
                playerWrapper.requestFullscreen().catch(err => {{}});
                fullscreenBtn.textContent = '⛶';
            }} else {{
                document.exitFullscreen();
                fullscreenBtn.textContent = '⛶';
            }}
        }});
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            switch(e.key) {{
                case ' ': e.preventDefault(); togglePlay(); break;
                case 'ArrowRight': video.currentTime += 5; break;
                case 'ArrowLeft': video.currentTime -= 5; break;
                case 'ArrowUp': e.preventDefault(); video.volume = Math.min(1, video.volume + 0.1); volumeSlider.value = video.volume * 100; updateMuteIcon(); break;
                case 'ArrowDown': e.preventDefault(); video.volume = Math.max(0, video.volume - 0.1); volumeSlider.value = video.volume * 100; updateMuteIcon(); break;
                case 'f': fullscreenBtn.click(); break;
                case 'm': muteBtn.click(); break;
                case '+': case '=': setZoom(currentZoom + ZOOM_STEP); break;
                case '-': setZoom(currentZoom - ZOOM_STEP); break;
                case '0': setZoom(1.0); break;
            }}
        }});
        
        // Show play overlay initially
        playOverlay.classList.add('show');
        
        // Video ended
        video.addEventListener('ended', function() {{
            playBtn.textContent = '▶';
            playOverlay.classList.add('show');
        }});
    </script>
    </body>
    </html>
    """
    return html


def get_theme_mode():
    """Detect Streamlit theme (dark or light)."""
    try:
        theme = st.get_option("theme.base")
        if theme:
            return theme
    except Exception:
        pass
    return "dark"  # Default to dark


# Page Config
st.set_page_config(
    page_title="Football AI Analysis", 
    layout="wide",
    page_icon="⚽"
)

st.title("⚽ Football AI Analysis")
st.markdown("### Automated Match Analysis using Computer Vision & Deep Learning")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Query params for auto-configuration
query_params = st.query_params
injected_stream_id = query_params.get("streamId")

# Mode Selection
st.sidebar.subheader("📊 Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Select Mode",
    ["Offline Processing", "Realtime Analysis"],
    index=1 if injected_stream_id else 0,
    help="Offline: Process entire video and save output. Realtime: Analyze live video stream."
)

# Visualization Options
st.sidebar.subheader("🎨 Visualization Options")
show_players = st.sidebar.checkbox("Show Players", True)
show_goalkeepers = st.sidebar.checkbox("Show Goalkeepers", True)
show_ball = st.sidebar.checkbox("Show Ball", True)
show_referees = st.sidebar.checkbox("Show Referees", True)
show_stats = st.sidebar.checkbox("Show Possession Stats", True)

# Performance Options
st.sidebar.subheader("⚡ Performance")
frame_skip = st.sidebar.slider(
    "Frame Skip (process every Nth frame)",
    min_value=1, max_value=5, value=3 if injected_stream_id else 2,
    help="Higher = faster but less smooth. 1 = process every frame, 3 = process every 3rd frame"
)
resolution_scale = st.sidebar.slider(
    "Resolution Scale",
    min_value=0.25, max_value=1.0, value=0.25 if injected_stream_id else 0.5, step=0.25,
    help="Lower = faster. 0.5 = half resolution"
)

# Hardware Acceleration
st.sidebar.subheader("🚀 Hardware Acceleration")
use_nvdec = st.sidebar.checkbox(
    "Enable NVDEC (NVIDIA GPU decode)",
    value=True,
    help="Use NVIDIA NVDEC for hardware video decoding. Saves 20-30%% CPU on RTSP streams. Requires NVIDIA GPU."
)

# Advanced / CLI Options
st.sidebar.subheader("🔧 Advanced / CLI")
use_fp16 = st.sidebar.checkbox(
    "Enable FP16 (CUDA only)",
    value=True,
    help="Enable half-precision inference. Requires NVIDIA GPU. ~2x throughput."
)
imgsz = st.sidebar.slider(
    "YOLO Input Size",
    min_value=320, max_value=1280, value=448 if injected_stream_id else 640, step=64,
    help="Lower = faster inference. 416 recommended for RTX 3050."
)
model_choice = st.sidebar.selectbox(
    "Model",
    [
        "models/best.pt (YOLOv5 PyTorch)",
        "models/best.engine (TensorRT)",
        "models/best.onnx (ONNX)",
        "Custom model path...",
    ],
    help="TensorRT/ONNX require running export_trt.py / export_onnx.py first on this machine"
)
if model_choice == "Custom model path...":
    custom_model = st.sidebar.text_input("Model path", value="models/best.pt")
    model_path = custom_model
elif model_choice == "models/best.engine (TensorRT)":
    model_path = "models/best.engine"
elif model_choice == "models/best.onnx (ONNX)":
    model_path = "models/best.onnx"
else:
    model_path = "models/best.pt"
st.sidebar.caption(f"Active model: {model_path}")
use_cache = st.sidebar.checkbox(
    "Use Cached Tracks",
    value=True,
    help="Save and load tracks locally to avoid re-running DL models on same video."
)

# Broadcast Controls
st.sidebar.subheader("📡 Broadcast")
enable_broadcast = st.sidebar.checkbox(
    "Enable Broadcast Stream",
    value=False,
    help="Stream processed video via MJPEG (/mjpeg) and RTMP. Requires: python scripts/broadcast_server.py [--rtmp URL]"
)
broadcast_url = st.sidebar.text_input(
    "RTMP URL (optional)",
    value="",
    help="Leave empty for MJPEG only. YouTube: rtmp://a.rtmp.youtube.com/live2/<KEY>"
)

# Build classes list
classes = []
if show_ball: classes.append("ball")
if show_goalkeepers: classes.append("goalkeepers")
if show_players: classes.append("players")
if show_referees: classes.append("referees")
if show_stats: classes.append("stats")

def _get_class_ids(selected_classes):
    return [value for key, value in options.items() if key in selected_classes]

class_ids = _get_class_ids(classes)

st.sidebar.markdown("---")

# ============================================================================
# OFFLINE PROCESSING MODE
# ============================================================================
if analysis_mode == "Offline Processing":
    st.sidebar.subheader("📹 Video Source")
    
    source_type = st.sidebar.selectbox(
        "Select Source",
        ["Demo Video", "Upload Video", "Booca VOD"]
    )
    
    video_data = None
    video_name = None
    
    if source_type == "Booca VOD":
        st.sidebar.markdown("### 📹 Booca VOD Analysis")
        
        theme_mode = get_theme_mode()
        is_dark = theme_mode == "dark"
        
        with st.sidebar.expander("ℹ️ How to use", expanded=False):
            st.markdown("""
            **Paste a Booca VOD URL to process the full recorded video:**
            
            📹 `https://booca.online/livestream/vod/{id}`
            
            The app will download and analyze the entire video offline.
            """)
        
        booca_vod_input = st.sidebar.text_input(
            "Booca VOD URL or Stream ID",
            value="",
            placeholder="https://booca.online/livestream/vod/...",
            help="Paste the full Booca VOD URL or just the 24-character stream ID",
            key="offline_booca_vod"
        )
        
        if booca_vod_input:
            import re as _re
            import requests as _req
            
            _vod_url = booca_vod_input.strip()
            _stream_id = None
            
            # Extract stream ID
            _match = _re.search(r'booca\.(?:online|vn)/(?:livestream|streaming)/(?:watch|vod|live)/([a-f0-9]{24})', _vod_url)
            if _match:
                _stream_id = _match.group(1)
            elif _re.match(r'^[a-f0-9]{24}$', _vod_url):
                _stream_id = _vod_url
            
            if _stream_id:
                try:
                    api_base = os.getenv("BOOCA_API_URL", "http://localhost:5000/api")
                    _api_resp = _req.get(
                        f"{api_base}/streams/{_stream_id}",
                        headers={
                            'User-Agent': 'Mozilla/5.0',
                            'Accept': 'application/json',
                            'Origin': 'https://booca.online',
                            'Referer': 'https://booca.online/'
                        },
                        timeout=10
                    )
                    _api_data = _api_resp.json()
                    
                    if _api_data.get('success'):
                        _sd = _api_data['data']
                        _vod_info = _sd.get('vod', {})
                        _title = _sd.get('title', 'Unknown')
                        _status = _sd.get('status', 'unknown')
                        _user = _sd.get('userId', {})
                        _stats = _sd.get('stats', {})
                        _category = _sd.get('category', '')
                        _thumbnail = _sd.get('thumbnail', '') or _vod_info.get('thumbnailUrl', '')
                        
                        _status_badge = "📹 VOD" if _status == 'vod_ready' else f"⚪ {_status}"
                        _user_name = f"{_user.get('lastName', '')} {_user.get('firstName', '')}".strip()
                        
                        # Stream info card with theme support
                        card_bg = "#1e293b" if is_dark else "#f1f5f9"
                        card_border = "#334155" if is_dark else "#e2e8f0"
                        card_text = "#e2e8f0" if is_dark else "#1e293b"
                        accent = "#00e0ca"
                        
                        st.sidebar.markdown(f"""
                        <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:10px; padding:12px; margin:8px 0;">
                            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                                <span style="background:#3b82f6; color:white; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600;">{_status_badge}</span>
                                <span style="color:{accent}; font-size:12px;">⚽ {_category}</span>
                            </div>
                            <div style="color:{card_text}; font-size:14px; font-weight:600; margin-bottom:6px; line-height:1.3;">{_title[:80]}</div>
                            <div style="color:#94a3b8; font-size:12px;">👤 {_user_name}</div>
                            <div style="color:#94a3b8; font-size:11px; margin-top:4px;">👁 {_stats.get('totalViews', 0)} views · ❤️ {_stats.get('likes', 0)} likes</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if _thumbnail:
                            try:
                                st.sidebar.image(_thumbnail, width="stretch")
                            except Exception:
                                pass
                        
                        _duration = _vod_info.get('duration', 0)
                        if _duration > 0:
                            st.sidebar.info(f"⏱️ Duration: {_duration // 60}m {_duration % 60}s")
                        
                        # Get VOD m3u8 URL
                        _m3u8_url = _vod_info.get('url', '') or _sd.get('playbackUrls', {}).get('hls', '')
                        
                        if _m3u8_url and _status == 'vod_ready':
                            st.sidebar.success(f"✅ VOD ready for analysis")
                            st.sidebar.caption(f"🔗 {_m3u8_url[:60]}...")
                            
                            # Download VOD via cv2 and convert to frames
                            # We store the URL; read_video can't handle m3u8,
                            # so we download via cv2 into a temp file
                            st.session_state['booca_vod_url'] = _m3u8_url
                            st.session_state['booca_vod_title'] = _title
                            video_name = _title
                        elif _status == 'live':
                            st.sidebar.warning("⚠️ This stream is currently LIVE. Use **Realtime Analysis** mode instead.")
                        else:
                            st.sidebar.warning(f"⚠️ VOD not ready yet (status: {_status})")
                    else:
                        st.sidebar.error("❌ Stream not found")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)[:60]}")
            else:
                st.sidebar.error("❌ Invalid Booca URL format")
    
    elif source_type == "Demo Video":
        demo_videos = {
            "Demo 1": "demos/demo1.mp4",
            "Demo 2": "demos/demo2.mp4"
        }
        
        selected_demo = st.sidebar.selectbox("Select Demo", list(demo_videos.keys()))
        video_path = demo_videos[selected_demo]
        
        if os.path.exists(video_path):
            st.sidebar.video(video_path)
            with open(video_path, "rb") as f:
                video_data = f.read()
            video_name = selected_demo
        else:
            st.sidebar.error(f"Demo video not found: {video_path}")
    
    elif source_type == "Upload Video":
        st.sidebar.markdown("### 📁 Upload Your Video")
        st.sidebar.markdown("*Drag & drop or click to browse*")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose video file",
            type=["mp4", "avi", "mov", "mkv", "flv"],
            help="Supported formats: MP4, AVI, MOV, MKV, FLV (Max 500MB recommended)"
        )
        
        if uploaded_file is not None:
            # File size info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.sidebar.info(f"📊 File size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 500:
                st.sidebar.warning("⚠️ Large file detected. Processing may take a while.")
            
            st.sidebar.success(f"✅ {uploaded_file.name}")
            
            # Preview
            with st.sidebar.expander("🎬 Preview Video"):
                st.video(uploaded_file)
            
            video_data = uploaded_file.read()
            video_name = uploaded_file.name
    
    # Process Button — support regular video_data or Booca VOD URL
    _booca_vod_url = st.session_state.get('booca_vod_url', '')
    _can_process = video_data is not None or bool(_booca_vod_url)
    
    if _can_process:
        if st.sidebar.button("🚀 Start Analysis", type="primary", use_container_width="always"):
            with st.spinner("🔄 Processing video... This may take several minutes."):
                try:
                    if video_data:
                        # Regular video file — use read_video
                        frames, fps, _, _ = read_video(video_data, verbose=False)
                    elif _booca_vod_url:
                        # Booca VOD — download frames via cv2.VideoCapture from HLS
                        st.info(f"📥 Downloading Booca VOD: {st.session_state.get('booca_vod_title', 'Unknown')}")
                        _cap = cv2.VideoCapture(_booca_vod_url, cv2.CAP_FFMPEG)
                        if not _cap.isOpened():
                            # Try with Booca referer
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                                "protocol_whitelist;file,http,https,tcp,tls,crypto"
                                "|headers;User-Agent: Mozilla/5.0\\r\\n"
                                "Referer: https://booca.online/\\r\\n"
                                "Origin: https://booca.online"
                            )
                            _cap = cv2.VideoCapture(_booca_vod_url, cv2.CAP_FFMPEG)
                        
                        # Fix BunnyCDN Referer drop bug using our background HTTP proxy for VOD
                        if not _cap.isOpened():
                            proxy = get_hls_proxy()
                            proxied_url = proxy.get_proxy_url(_booca_vod_url)
                            _cap = cv2.VideoCapture(proxied_url, cv2.CAP_FFMPEG)
                        
                        if not _cap.isOpened():
                            raise RuntimeError(f"Cannot open VOD stream: {_booca_vod_url[:60]}")
                        
                        fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
                        frames = []
                        _dl_progress = st.progress(0, text="Downloading VOD frames...")
                        _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                        _frame_idx = 0
                        
                        while True:
                            ret, frame = _cap.read()
                            if not ret:
                                break
                            frames.append(frame)
                            _frame_idx += 1
                            if _total_frames > 0 and _frame_idx % 30 == 0:
                                _dl_progress.progress(
                                    min(_frame_idx / _total_frames, 0.99),
                                    text=f"Downloaded {_frame_idx}/{_total_frames} frames..."
                                )
                        
                        _cap.release()
                        _dl_progress.progress(1.0, text=f"Downloaded {len(frames)} frames")
                        
                        if len(frames) == 0:
                            raise RuntimeError("No frames downloaded from VOD")
                        
                        # Clear the stored URL
                        st.session_state.pop('booca_vod_url', None)
                    else:
                        raise RuntimeError("No video source available")
                    
                    progress_bar = st.progress(0, text="Initializing models...")
                    import pickle
                    import re
                    
                    safe_video_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', video_name) if video_name else "unknown_video"
                    os.makedirs("stubs", exist_ok=True)
                    tracks_stub_path = f"stubs/tracks_{safe_video_name}.pkl"
                    camera_stub_path = f"stubs/camera_movement_{safe_video_name}.pkl"
                    
                    tracks = None
                    camera_movement_per_frame = None
                    
                    # Initialize Models
                    tracker = Tracker(model_path, class_ids, verbose=False, fp16=use_fp16, imgsz=imgsz)
                    camera_movement_estimator = CameraMovementEstimator(frames[0], class_ids, verbose=False)

                    if use_cache and os.path.exists(tracks_stub_path) and os.path.exists(camera_stub_path):
                        progress_bar.progress(20, text="Loading cached tracks...")
                        with open(tracks_stub_path, "rb") as f:
                            tracks = pickle.load(f)
                        with open(camera_stub_path, "rb") as f:
                            camera_movement_per_frame = pickle.load(f)
                    
                    if tracks is None or camera_movement_per_frame is None:
                        progress_bar.progress(10, text="Tracking objects...")
                        
                        tracks = tracker.get_object_tracks(frames)
                        tracker.add_position_to_tracks(tracks)
                        
                        progress_bar.progress(30, text="Estimating camera movement...")
                        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames)
                        
                        if use_cache:
                            with open(tracks_stub_path, "wb") as f:
                                pickle.dump(tracks, f)
                            with open(camera_stub_path, "wb") as f:
                                pickle.dump(camera_movement_per_frame, f)
                    
                    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)
                    progress_bar.progress(50, text="Interpolating ball positions...")
                    
                    # Ball interpolation
                    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
                    progress_bar.progress(60, text="Assigning teams...")
                    
                    # Team assignment
                    team_assigner = TeamAssigner()
                    team_assigner.get_teams(frames, tracks)
                    progress_bar.progress(75, text="Calculating possession...")
                    
                    # Ball possession
                    player_assigner = PlayerBallAssigner()
                    player_assigner.get_player_and_possession(tracks)
                    progress_bar.progress(80, text="Running advanced analytics...")
                    
                    # ============================================================
                    # ADVANCED ANALYTICS — SpeedEstimator, TacticalAnalyzer, EventDetector
                    # ============================================================
                    frame_h, frame_w = frames[0].shape[:2]
                    speed_estimator = SpeedEstimator(fps=fps)
                    tactical_analyzer = TacticalAnalyzer()
                    event_detector = EventDetector(frame_width=frame_w, frame_height=frame_h)
                    
                    all_events = []
                    all_top_speeds = []
                    
                    for f_idx in range(len(frames)):
                        # Build single-frame tracks dict for the analyzers
                        sf_tracks = {
                            "players": tracks["players"][f_idx] if f_idx < len(tracks["players"]) else {},
                            "referees": tracks["referees"][f_idx] if f_idx < len(tracks["referees"]) else {},
                            "ball": tracks["ball"][f_idx] if f_idx < len(tracks["ball"]) else {},
                        }
                        
                        # Speed & distance estimation
                        top_players = speed_estimator.update_speeds(sf_tracks, f_idx)
                        if top_players:
                            all_top_speeds = top_players  # keep latest
                        
                        # Heatmap tracking
                        tactical_analyzer.update_heatmap(sf_tracks)
                        
                        # Pass detection
                        bp = player_assigner.ball_possession
                        current_possessor = None
                        if bp is not None and f_idx < len(bp) and bp[f_idx] in [1, 2]:
                            # Find which player has the ball
                            for pid, pdata in sf_tracks["players"].items():
                                if pdata.get("has_ball", False):
                                    current_possessor = pid
                                    break
                        
                        pass_event = tactical_analyzer.detect_passes(sf_tracks, current_possessor, f_idx)
                        if pass_event:
                            all_events.append(pass_event)
                        
                        # Event detection (goals)
                        goal_event = event_detector.check_events(sf_tracks, f_idx)
                        if goal_event:
                            goal_event["frame"] = f_idx
                            all_events.append(goal_event)
                    
                    # Tactical analysis (formations + space control)
                    last_frame_tracks = {
                        "players": tracks["players"][-1] if tracks["players"] else {},
                        "referees": tracks["referees"][-1] if tracks["referees"] else {},
                        "ball": tracks["ball"][-1] if tracks["ball"] else {},
                    }
                    tactical_result = tactical_analyzer.analyze_tactics(last_frame_tracks)
                    
                    # Collect all player stats
                    all_player_stats = {}
                    for pid in speed_estimator.total_distances:
                        all_player_stats[pid] = {
                            "total_distance_m": round(speed_estimator.total_distances[pid], 1),
                            "sprint_count": speed_estimator.sprint_counts.get(pid, 0),
                            "current_speed": speed_estimator.current_speeds.get(pid, 0),
                        }
                        # Add team info
                        for frame_tracks in reversed(tracks["players"]):
                            if pid in frame_tracks and "team" in frame_tracks[pid]:
                                all_player_stats[pid]["team"] = frame_tracks[pid]["team"]
                                break
                    
                    progress_bar.progress(88, text="Drawing annotations...")
                    
                    # Draw annotations
                    output = tracker.draw_annotations(frames, tracks, player_assigner.ball_possession)
                    output = camera_movement_estimator.draw_camera_movement(output, camera_movement_per_frame)
                    progress_bar.progress(95, text="Saving output...")
                    
                    # Save output
                    output_path = "output/output.mp4"
                    save_video(output, output_path, fps, verbose=False)
                    progress_bar.progress(100, text="Complete!")
                    
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    st.success("✅ Processing complete!")

                    # Display result with custom YouTube-like player
                    st.subheader("📹 Processed Video")
                    theme_mode = get_theme_mode()
                    player_html = get_custom_video_player_html(output_path, theme=theme_mode)
                    # The embedded iframe height must be large enough, otherwise the video can appear "cut off"
                    # (common on 16:9 sources with custom controls).
                    offline_player_height = int(os.getenv("OFFLINE_PLAYER_HEIGHT", "780"))
                    v_col, mm_col = st.columns([3, 1], vertical_alignment="top")
                    with v_col:
                        components.html(player_html, height=offline_player_height, scrolling=True)

                    with mm_col:
                        st.markdown("#### 🗺️ Minimap")
                        try:
                            total_frames_count = len(frames)
                            default_idx = max(0, total_frames_count - 1)
                            sel_idx = st.slider(
                                "Frame",
                                min_value=0,
                                max_value=max(0, total_frames_count - 1),
                                value=default_idx,
                                step=1,
                                help="Kéo để xem minimap/mô phỏng theo thời gian (bóng sẽ di chuyển theo frame).",
                            )

                            sel_players = tracks["players"][sel_idx] if isinstance(tracks, dict) and tracks.get("players") else {}
                            sel_ball = tracks["ball"][sel_idx] if isinstance(tracks, dict) and tracks.get("ball") else {}

                            # Normalize ball track shape to what minimap expects: {1: {"bbox": ...}}
                            if isinstance(sel_ball, list):
                                # If it's a list of ball detections, pick first one as id=1
                                sel_ball = {1: sel_ball[0]} if len(sel_ball) > 0 and isinstance(sel_ball[0], dict) else {}
                            elif isinstance(sel_ball, dict) and "bbox" in sel_ball and 1 not in sel_ball:
                                # If it's a single ball dict, wrap it
                                sel_ball = {1: sel_ball}

                            sel_tracks = {"players": sel_players or {}, "ball": sel_ball or {}}

                            # Build short history for a nicer "simulation" trail
                            trail_len = int(os.getenv("OFFLINE_SIM_TRAIL_LEN", "20"))
                            start_i = max(0, sel_idx - trail_len)
                            hist = []
                            if isinstance(tracks, dict) and tracks.get("ball"):
                                for i in range(start_i, sel_idx + 1):
                                    b = tracks["ball"][i]
                                    if isinstance(b, list):
                                        b = {1: b[0]} if len(b) > 0 and isinstance(b[0], dict) else {}
                                    elif isinstance(b, dict) and "bbox" in b and 1 not in b:
                                        b = {1: b}
                                    hist.append({"ball": b or {}})

                            mm = render_minimap_rgb(sel_tracks, width=360, height=200)
                            st.image(mm, channels="RGB", use_container_width=True)
                            st.caption("Bóng (vàng) & cầu thủ cầm bóng (viền theo đội)")

                            st.markdown("#### 🎮 Mô phỏng")
                            sim = render_match_sim_rgb(sel_tracks, ball_history=hist, width=360, height=220)
                            st.image(sim, channels="RGB", use_container_width=True)
                        except Exception as e:
                            st.info("Minimap (offline) chưa render được cho video này.")
                            with st.expander("Xem lỗi minimap"):
                                st.code(str(e))
                    
                    # ============================================================
                    # 📊 ANALYTICS DASHBOARD
                    # ============================================================
                    st.markdown("---")
                    st.subheader("📊 Match Analytics Dashboard")
                    
                    is_dark = theme_mode == "dark"
                    _card_bg = "#1e293b" if is_dark else "#f8fafc"
                    _card_border = "#334155" if is_dark else "#e2e8f0"
                    _text_primary = "#f1f5f9" if is_dark else "#1e293b"
                    _text_secondary = "#94a3b8" if is_dark else "#64748b"
                    _accent_1 = "#3b82f6"
                    _accent_2 = "#ef4444"
                    _accent_green = "#22c55e"
                    
                    # --- Row 1: Key Metrics ---
                    total_frames_count = len(frames)
                    match_duration_sec = total_frames_count / fps if fps > 0 else 0
                    match_duration_min = match_duration_sec / 60
                    
                    bp_arr = player_assigner.ball_possession
                    t1_poss = int(np.sum(bp_arr == 1)) if bp_arr is not None and len(bp_arr) > 0 else 0
                    t2_poss = int(np.sum(bp_arr == 2)) if bp_arr is not None and len(bp_arr) > 0 else 0
                    total_poss = t1_poss + t2_poss
                    t1_poss_pct = round(t1_poss / total_poss * 100) if total_poss > 0 else 50
                    t2_poss_pct = 100 - t1_poss_pct
                    
                    t1_passes = sum(1 for e in all_events if e.get("event") == "pass" and e.get("scoring_team", e.get("scoringTeam")) == 1)
                    t2_passes = sum(1 for e in all_events if e.get("event") == "pass" and e.get("scoring_team", e.get("scoringTeam")) == 2)
                    t1_goals = sum(1 for e in all_events if e.get("event") == "goal" and e.get("scoring_team") == 1)
                    t2_goals = sum(1 for e in all_events if e.get("event") == "goal" and e.get("scoring_team") == 2)
                    
                    num_players_tracked = len(all_player_stats)
                    
                    def _metric_card(label, value, icon, accent_color):
                        return f"""
                        <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; 
                                    padding:20px 16px; text-align:center; position:relative; overflow:hidden;">
                            <div style="position:absolute; top:0; left:0; right:0; height:3px; background:{accent_color};"></div>
                            <div style="font-size:28px; margin-bottom:6px;">{icon}</div>
                            <div style="font-size:28px; font-weight:700; color:{_text_primary}; margin-bottom:4px;">{value}</div>
                            <div style="font-size:12px; color:{_text_secondary}; text-transform:uppercase; letter-spacing:1px;">{label}</div>
                        </div>
                        """
                    
                    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                    with m_col1:
                        st.markdown(_metric_card("Duration", f"{match_duration_min:.1f} min", "⏱️", _accent_1), unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(_metric_card("Frames", f"{total_frames_count:,}", "🎬", _accent_green), unsafe_allow_html=True)
                    with m_col3:
                        st.markdown(_metric_card("Players Tracked", str(num_players_tracked), "👥", "#a855f7"), unsafe_allow_html=True)
                    with m_col4:
                        st.markdown(_metric_card("Total Passes", str(t1_passes + t2_passes), "🔄", "#f59e0b"), unsafe_allow_html=True)
                    with m_col5:
                        st.markdown(_metric_card("Goals Detected", str(t1_goals + t2_goals), "⚽", _accent_2), unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # --- Row 2: Possession & Space Control ---
                    poss_col, space_col = st.columns(2)
                    
                    with poss_col:
                        st.markdown(f"""
                        <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; padding:24px;">
                            <h4 style="color:{_text_primary}; margin:0 0 16px 0; font-size:16px;">⚽ Ball Possession</h4>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                                <div style="text-align:center;">
                                    <div style="font-size:32px; font-weight:700; color:{_accent_1};">{t1_poss_pct}%</div>
                                    <div style="color:{_text_secondary}; font-size:13px;">Team 1</div>
                                </div>
                                <div style="font-size:20px; color:{_text_secondary};">vs</div>
                                <div style="text-align:center;">
                                    <div style="font-size:32px; font-weight:700; color:{_accent_2};">{t2_poss_pct}%</div>
                                    <div style="color:{_text_secondary}; font-size:13px;">Team 2</div>
                                </div>
                            </div>
                            <div style="background:{_card_border}; border-radius:6px; height:12px; overflow:hidden;">
                                <div style="background:linear-gradient(90deg, {_accent_1}, #60a5fa); height:100%; width:{t1_poss_pct}%; border-radius:6px;"></div>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:12px;">
                                <div style="color:{_text_secondary}; font-size:12px;">Passes: {t1_passes}</div>
                                <div style="color:{_text_secondary}; font-size:12px;">Passes: {t2_passes}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with space_col:
                        sc = tactical_result.get("space_control", {"team1": 50.0, "team2": 50.0})
                        formations = tactical_result.get("formations", {})
                        f1 = formations.get(1, "N/A")
                        f2 = formations.get(2, "N/A")
                        
                        st.markdown(f"""
                        <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; padding:24px;">
                            <h4 style="color:{_text_primary}; margin:0 0 16px 0; font-size:16px;">🗺️ Space Control & Formations</h4>
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                                <div style="text-align:center;">
                                    <div style="font-size:32px; font-weight:700; color:{_accent_1};">{sc['team1']}%</div>
                                    <div style="color:{_text_secondary}; font-size:13px;">Team 1</div>
                                </div>
                                <div style="font-size:20px; color:{_text_secondary};">vs</div>
                                <div style="text-align:center;">
                                    <div style="font-size:32px; font-weight:700; color:{_accent_2};">{sc['team2']}%</div>
                                    <div style="color:{_text_secondary}; font-size:13px;">Team 2</div>
                                </div>
                            </div>
                            <div style="background:{_card_border}; border-radius:6px; height:12px; overflow:hidden;">
                                <div style="background:linear-gradient(90deg, {_accent_1}, #60a5fa); height:100%; width:{sc['team1']}%; border-radius:6px;"></div>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:12px;">
                                <div style="color:{_text_secondary}; font-size:12px;">Formation: <b style="color:{_text_primary}">{f1}</b></div>
                                <div style="color:{_text_secondary}; font-size:12px;">Formation: <b style="color:{_text_primary}">{f2}</b></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # --- Row 3: Player Performance Table ---
                    st.markdown(f"""
                    <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; padding:24px;">
                        <h4 style="color:{_text_primary}; margin:0 0 16px 0; font-size:16px;">🏃 Player Performance Rankings</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if all_player_stats:
                        # Sort by total distance
                        sorted_players = sorted(
                            all_player_stats.items(),
                            key=lambda x: x[1].get("total_distance_m", 0),
                            reverse=True
                        )[:15]  # Top 15
                        
                        perf_data = []
                        for rank, (pid, stats) in enumerate(sorted_players, 1):
                            team = stats.get("team", "?")
                            team_label = f"🔵 Team 1" if team == 1 else f"🔴 Team 2" if team == 2 else "⚪ N/A"
                            perf_data.append({
                                "Rank": rank,
                                "Player ID": int(pid),
                                "Team": team_label,
                                "Distance (m)": f"{stats['total_distance_m']:.1f}",
                                "Top Speed (km/h)": f"{stats['current_speed']:.1f}",
                                "Sprints": stats["sprint_count"],
                            })
                        
                        import pandas as _pd
                        perf_df = _pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No player performance data available.")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # --- Row 4: Team Heatmaps ---
                    st.markdown(f"""
                    <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; padding:24px;">
                        <h4 style="color:{_text_primary}; margin:0 0 16px 0; font-size:16px;">🔥 Team Heatmaps</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    heatmaps = tactical_result.get("heatmaps", {})
                    hm_col1, hm_col2 = st.columns(2)
                    
                    with hm_col1:
                        st.markdown(f"<p style='color:{_accent_1}; font-weight:600; text-align:center;'>🔵 Team 1 Activity</p>", unsafe_allow_html=True)
                        t1_hm = np.array(heatmaps.get("team1", np.zeros((10, 10))))
                        if np.max(t1_hm) > 0:
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            fig1, ax1 = plt.subplots(figsize=(5, 3))
                            fig1.patch.set_facecolor('#0e1117' if is_dark else '#ffffff')
                            ax1.set_facecolor('#0e1117' if is_dark else '#ffffff')
                            # 10x10 grid → use nearest to avoid "blurry" artifacts
                            im1 = ax1.imshow(
                                t1_hm,
                                cmap='Blues',
                                interpolation='nearest',
                                aspect='equal',
                                origin='lower',
                                vmin=0.0,
                                vmax=1.0,
                            )
                            ax1.set_title("Team 1 Heatmap", color=_text_primary, fontsize=11)
                            ax1.set_xticks([])
                            ax1.set_yticks([])
                            for spine in ax1.spines.values():
                                spine.set_visible(False)
                            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                            st.pyplot(fig1)
                            plt.close(fig1)
                        else:
                            st.info("No heatmap data for Team 1")
                    
                    with hm_col2:
                        st.markdown(f"<p style='color:{_accent_2}; font-weight:600; text-align:center;'>🔴 Team 2 Activity</p>", unsafe_allow_html=True)
                        t2_hm = np.array(heatmaps.get("team2", np.zeros((10, 10))))
                        if np.max(t2_hm) > 0:
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            fig2, ax2 = plt.subplots(figsize=(5, 3))
                            fig2.patch.set_facecolor('#0e1117' if is_dark else '#ffffff')
                            ax2.set_facecolor('#0e1117' if is_dark else '#ffffff')
                            im2 = ax2.imshow(
                                t2_hm,
                                cmap='Reds',
                                interpolation='nearest',
                                aspect='equal',
                                origin='lower',
                                vmin=0.0,
                                vmax=1.0,
                            )
                            ax2.set_title("Team 2 Heatmap", color=_text_primary, fontsize=11)
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            for spine in ax2.spines.values():
                                spine.set_visible(False)
                            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
                            st.pyplot(fig2)
                            plt.close(fig2)
                        else:
                            st.info("No heatmap data for Team 2")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # --- Row 5: Event Timeline ---
                    if all_events:
                        st.markdown(f"""
                        <div style="background:{_card_bg}; border:1px solid {_card_border}; border-radius:12px; padding:24px;">
                            <h4 style="color:{_text_primary}; margin:0 0 16px 0; font-size:16px;">📋 Event Timeline</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for evt in all_events[:30]:  # Show up to 30 events
                            evt_frame = evt.get("frame", 0)
                            evt_time_sec = evt_frame / fps if fps > 0 else 0
                            evt_min = int(evt_time_sec // 60)
                            evt_sec = int(evt_time_sec % 60)
                            
                            if evt.get("event") == "goal":
                                scoring = evt.get("scoring_team", "?")
                                team_color = _accent_1 if scoring == 1 else _accent_2
                                st.markdown(f"""
                                <div style="background:{_card_bg}; border-left:4px solid {team_color}; padding:10px 16px; margin:4px 0; border-radius:0 8px 8px 0;">
                                    <span style="color:{team_color}; font-weight:700;">⚽ GOAL</span>
                                    <span style="color:{_text_secondary}; margin-left:12px;">{evt_min}:{evt_sec:02d}</span>
                                    <span style="color:{_text_primary}; margin-left:12px;">Team {scoring} scored ({evt.get('side', '')} side)</span>
                                </div>
                                """, unsafe_allow_html=True)
                            elif evt.get("event") == "pass":
                                scoring_team = evt.get("scoring_team", evt.get("scoringTeam", "?"))
                                team_color = _accent_1 if scoring_team == 1 else _accent_2
                                st.markdown(f"""
                                <div style="background:{_card_bg}; border-left:4px solid {_accent_green}; padding:8px 16px; margin:2px 0; border-radius:0 8px 8px 0;">
                                    <span style="color:{_accent_green}; font-weight:600;">🔄 Pass</span>
                                    <span style="color:{_text_secondary}; margin-left:12px;">{evt_min}:{evt_sec:02d}</span>
                                    <span style="color:{_text_primary}; margin-left:12px;">Player {evt.get('from_player', '?')} → Player {evt.get('to_player', '?')} (Team {scoring_team})</span>
                                </div>
                                """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# ============================================================================
# REALTIME ANALYSIS MODE
# ============================================================================
else:  # Realtime Analysis
    st.sidebar.subheader("📹 Video Source")
    
    source_type = st.sidebar.selectbox(
        "Select Source",
        ["Booca Stream", "Webcam", "Video File", "URL Stream"]
    )
    
    source_path = None
    
    # Helper function
    def _get_available_cameras(max_indices=5):
        available_indices = []
        for i in range(max_indices):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_indices.append(i)
                cap.release()
        return available_indices
    
    def resolve_booca_url(url):
        """
        Resolve Booca livestream/VOD URL to direct m3u8 stream.
        Supports:
          - https://booca.online/livestream/watch/{id}  (live)
          - https://booca.online/livestream/vod/{id}    (recorded)
          - Direct stream ID (just the hex id)
        Returns: (stream_url, stream_info_dict) or (None, error_msg)
        """
        import re
        import requests as req
        
        url = url.strip()
        
        # Extract stream ID from URL
        stream_id = None
        is_vod = False
        
        # Pattern: /livestream/watch/{id} or /livestream/vod/{id} or /streaming/live/{id} or /streaming/vod/{id}
        match = re.search(r'booca\.(?:online|vn)/(?:livestream|streaming)/(?:watch|vod|live)/([a-f0-9]{24})', url)
        if match:
            stream_id = match.group(1)
            is_vod = '/vod/' in url
        # Pattern: just a 24-char hex ID
        elif re.match(r'^[a-f0-9]{24}$', url):
            stream_id = url
        
        if not stream_id:
            return None, "Invalid Booca URL. Use format: https://booca.online/livestream/watch/{id} or /vod/{id}"
        
        # Call Booca API
        api_base = os.getenv("BOOCA_API_URL", "http://localhost:5000/api")
        api_url = f"{api_base}/streams/{stream_id}"
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Origin': 'https://booca.online',
                'Referer': 'https://booca.online/'
            }
            resp = req.get(api_url, headers=headers, timeout=10)
            data = resp.json()
            
            if not data.get('success'):
                return None, f"API error: stream not found (ID: {stream_id})"
            
            stream_data = data['data']
            status = stream_data.get('status', 'unknown')
            title = stream_data.get('title', 'Unknown')
            category = stream_data.get('category', '')
            playback = stream_data.get('playbackUrls', {})
            vod_info = stream_data.get('vod', {})
            
            info = {
                'id': stream_id,
                'title': title,
                'status': status,
                'category': category,
                'is_live': status == 'live',
                'is_vod': status == 'vod_ready',
                'is_ended': status == 'ended',
                'user': stream_data.get('userId', {}),
                'stats': stream_data.get('stats', {}),
                'thumbnail': stream_data.get('thumbnail', '') or vod_info.get('thumbnailUrl', ''),
            }
            
            # For VOD-ready streams, prefer the VOD CDN URL
            if status == 'vod_ready' and vod_info.get('url'):
                info['duration'] = vod_info.get('duration', 0)
                return vod_info['url'], info
            
            # For ended streams, check if VOD is available or being processed
            if status == 'ended':
                vod_status = vod_info.get('status', '')
                if vod_info.get('url') and vod_status == 'ready':
                    # VOD is ready even though status says ended
                    info['is_vod'] = True
                    info['duration'] = vod_info.get('duration', 0)
                    return vod_info['url'], info
                elif vod_status == 'pending':
                    return None, f"Stream đã kết thúc. VOD đang được xử lý, vui lòng thử lại sau vài phút."
                else:
                    return None, f"Stream đã kết thúc và không có bản ghi VOD. (status: ended, vod: {vod_status or 'none'})"
            
            # For live streams, use HLS playback URL
            if status == 'live' and playback.get('hls'):
                return playback['hls'], info
            
            # Fallback to FLV for live streams
            if status == 'live' and playback.get('flv'):
                return playback['flv'], info
            
            # For other statuses (idle, etc.)
            if playback.get('hls'):
                return playback['hls'], info
            
            return None, f"Không có URL phát (status: {status})"
            
        except req.exceptions.RequestException as e:
            return None, f"Network error: {str(e)[:60]}"
        except Exception as e:
            return None, f"Error: {str(e)[:60]}"
    
    def resolve_stream_url(url):
        """
        Resolve webpage URL to direct video stream.
        Tries multiple methods: player extraction -> pafy -> yt-dlp -> streamlink -> direct URL
        """
        if not url or not url.strip():
            return url
        
        url = url.strip()
        
        # If it's already a direct stream URL, return as-is
        if (url.endswith((".m3u8", ".mp4", ".flv", ".ts")) or 
            url.startswith("rtsp://") or 
            url.startswith("http://192.168") or
            url.startswith("http://10.0")):
            st.sidebar.info(f"ℹ️ Using direct URL")
            return url
        
        # Method 0a: Booca.online livestream/VOD
        if 'booca.online/livestream/' in url or 'booca.vn/livestream/' in url or 'streaming/live/' in url:
            st.sidebar.text("🔍 Resolving Booca stream...")
            stream_url, info = resolve_booca_url(url)
            if stream_url:
                if isinstance(info, dict):
                    status_emoji = "🔴" if info.get('is_live') else "📹"
                    st.sidebar.success(f"{status_emoji} {info.get('title', '')[:50]}")
                    st.session_state['booca_stream_info'] = info
                return stream_url
            else:
                st.sidebar.warning(f"⚠️ Booca: {info}")
        
        # Method 0b: Extract m3u8 from embed player pages (rkplayer, cakhia, etc.)
        player_domains = ["rkplayer", "cakhia", "xoilac", "vebo", "socolive", "xem"]
        if any(domain in url for domain in player_domains):
            try:
                import requests
                import re
                st.sidebar.text("🔍 Extracting from player page...")
                
                # Determine the correct referer based on URL
                if "rkplayer" in url:
                    referer = url  # Use the player page itself as referer
                elif "cakhia" in url or "xem" in url:
                    referer = "https://watch.rkplayer.xyz/"
                else:
                    referer = url
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Referer': referer
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                html = response.text
                
                # Look for m3u8 URLs in the page source
                m3u8_patterns = [
                    r'(https?://[^\s\'"]+\.m3u8[^\s\'"]*)',
                    r'file:\s*[\'"]([^\'"]+\.m3u8[^\'"]*)[\'"]',
                    r'source:\s*[\'"]([^\'"]+\.m3u8[^\'"]*)[\'"]',
                    r'src:\s*[\'"]([^\'"]+\.m3u8[^\'"]*)[\'"]',
                ]
                
                for pattern in m3u8_patterns:
                    matches = re.findall(pattern, html)
                    if matches:
                        m3u8_url = matches[0]
                        # Clean up the URL
                        m3u8_url = m3u8_url.replace('\\/', '/').replace('\\"', '')
                        st.sidebar.success(f"✅ Found m3u8 stream!")
                        # Store the referer for later use
                        st.session_state['stream_referer'] = referer
                        st.session_state['original_player_url'] = url
                        return m3u8_url
                
                st.sidebar.warning("⚠️ No m3u8 found in page source")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Player extraction failed: {str(e)[:40]}")
        
        # Method 1: Try pafy for YouTube (best for YouTube)
        if "youtube.com" in url or "youtu.be" in url:
            try:
                import pafy
                st.sidebar.text("🔍 Trying pafy (YouTube)...")
                video = pafy.new(url)
                best = video.getbest(preftype="mp4")
                if best:
                    st.sidebar.success(f"✅ Resolved via pafy ({best.resolution})")
                    return best.url
            except ImportError:
                st.sidebar.warning("⚠️ Pafy not installed")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Pafy failed: {str(e)[:50]}")
        
        # Method 2: Try yt-dlp (best for most platforms)
        try:
            import yt_dlp
            st.sidebar.text("🔍 Trying yt-dlp...")
            
            # Facebook uses non-standard format IDs (dash_sd_src, dash_hd_src)
            # Try multiple format strings from strict to relaxed
            is_facebook = "facebook.com" in url or "fb.watch" in url
            
            if is_facebook:
                format_attempts = [
                    'best[ext=mp4]',       # Best MP4
                    'sd',                   # Facebook SD
                    'best',                 # Any best
                    None,                   # No format filter (let yt-dlp decide)
                ]
            else:
                format_attempts = [
                    'best[height<=720][ext=mp4]/best[height<=720]/best',
                ]
            
            resolved_url = None
            for fmt in format_attempts:
                try:
                    ydl_opts = {
                        'quiet': True,
                        'no_warnings': True,
                        'extract_flat': False,
                        'socket_timeout': 30,
                    }
                    if fmt:
                        ydl_opts['format'] = fmt
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        if info and 'url' in info:
                            resolved_url = info['url']
                            break
                        elif info and 'formats' in info and len(info['formats']) > 0:
                            # Get formats that have video
                            formats = [f for f in info['formats'] if f.get('url') and f.get('vcodec') != 'none']
                            if not formats:
                                # Fallback: any format with a URL
                                formats = [f for f in info['formats'] if f.get('url')]
                            if formats:
                                # Prefer lower resolution for stability
                                for f in formats:
                                    if f.get('height') and f['height'] <= 720:
                                        resolved_url = f['url']
                                        break
                                if not resolved_url:
                                    resolved_url = formats[-1]['url']
                                break
                except Exception:
                    continue
            
            if resolved_url:
                st.sidebar.success(f"✅ Resolved via yt-dlp" + (" (Facebook)" if is_facebook else ""))
                return resolved_url
            else:
                st.sidebar.warning("⚠️ yt-dlp: no playable format found")
                
        except ImportError:
            st.sidebar.warning("⚠️ yt-dlp not installed. Install with: `pip install yt-dlp`")
        except Exception as e:
            st.sidebar.warning(f"⚠️ yt-dlp failed: {str(e)[:50]}")
        
        # Method 3: Try streamlink
        try:
            import streamlink
            st.sidebar.text("🔍 Trying streamlink...")
            streams = streamlink.streams(url)
            if streams:
                # Try different qualities (prefer lower for stability)
                for quality in ["480p", "360p", "worst", "best"]:
                    if quality in streams:
                        resolved_url = streams[quality].url
                        st.sidebar.success(f"✅ Resolved via streamlink ({quality})")
                        return resolved_url
                # Fallback to first available
                resolved_url = list(streams.values())[0].url
                st.sidebar.success("✅ Resolved via streamlink")
                return resolved_url
        except ImportError:
            st.sidebar.warning("⚠️ Streamlink not installed")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Streamlink failed: {str(e)[:50]}")
        
        # Method 4: Return original URL and let OpenCV try
        st.sidebar.warning("⚠️ Could not resolve stream. Trying direct connection...")
        return url
    
    # Source configuration
    if source_type == "Webcam":
        st.sidebar.markdown("### 📷 Webcam Setup")
        
        with st.sidebar.expander("ℹ️ How to connect phone camera"):
            st.markdown("""
            **Using Phone as Webcam:**
            1. Install **DroidCam** or **IP Webcam** app
            2. Connect phone to same WiFi as computer
            3. Note the IP address shown in app
            4. Use 'URL Stream' option with: `http://YOUR_PHONE_IP:8080/video`
            
            **Example:** `http://192.168.1.5:8080/video`
            """)
        
        available_cams = _get_available_cameras()
        if not available_cams:
            st.sidebar.error("❌ No webcams detected!")
            source_path = None
        else:
            cam_index = st.sidebar.selectbox(
                "Select Camera",
                available_cams,
                format_func=lambda x: f"Camera {x}" + (" (Default)" if x == 0 else "")
            )
            source_path = cam_index
            st.sidebar.success(f"✅ Camera {source_path} selected")
    
    elif source_type == "Video File":
        st.sidebar.markdown("### 📁 Upload Video")
        st.sidebar.markdown("*Drag & drop or click to browse*")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose video file",
            type=["mp4", "avi", "mov", "mkv", "flv"],
            help="Supported: MP4, AVI, MOV, MKV, FLV"
        )
        
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.sidebar.info(f"📊 {file_size_mb:.2f} MB")
            st.sidebar.success(f"✅ {uploaded_file.name}")
            
            # Save temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
            tfile.write(uploaded_file.read())
            source_path = tfile.name
            
            with st.sidebar.expander("🎬 Preview"):
                st.video(uploaded_file)
    
    elif source_type == "Booca Stream":
        st.sidebar.markdown("### 🟢 Booca Livestream / VOD")
        
        theme_mode = get_theme_mode()
        is_dark = theme_mode == "dark"
        
        with st.sidebar.expander("ℹ️ How to use", expanded=False):
            st.markdown("""
            **Paste a Booca livestream or VOD URL:**
            
            🔴 **Live:** `https://booca.online/livestream/watch/{id}`
            📹 **VOD:** `https://booca.online/livestream/vod/{id}`
            
            The app will automatically extract the stream URL via Booca API.
            """)
        
        booca_url_input = st.sidebar.text_input(
            "Booca URL or Stream ID",
            value=injected_stream_id if injected_stream_id else "",
            placeholder="https://booca.online/livestream/watch/...",
            help="Paste the full Booca URL or just the 24-character stream ID"
        )
        
        if booca_url_input:
            with st.sidebar.spinner("Resolving Booca stream..."):
                stream_url, info = resolve_booca_url(booca_url_input)
            
            if stream_url and isinstance(info, dict):
                # Show stream info card
                status = info.get('status', 'unknown')
                is_live = info.get('is_live', False)
                title = info.get('title', 'Unknown')
                category = info.get('category', '')
                user_data = info.get('user', {})
                stats = info.get('stats', {})
                thumbnail = info.get('thumbnail', '')
                
                status_badge = "🔴 LIVE" if is_live else "📹 VOD" if info.get('is_vod') else f"⚪ {status}"
                user_name = f"{user_data.get('lastName', '')} {user_data.get('firstName', '')}".strip()
                
                # Stream info card with theme support
                card_bg = "#1e293b" if is_dark else "#f1f5f9"
                card_border = "#334155" if is_dark else "#e2e8f0"
                card_text = "#e2e8f0" if is_dark else "#1e293b"
                accent = "#00e0ca"
                live_color = "#ef4444" if is_live else "#3b82f6"
                
                st.sidebar.markdown(f"""
                <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:10px; padding:12px; margin:8px 0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="background:{live_color}; color:white; padding:2px 8px; border-radius:12px; font-size:12px; font-weight:600;">{status_badge}</span>
                        <span style="color:{accent}; font-size:12px;">⚽ {category}</span>
                    </div>
                    <div style="color:{card_text}; font-size:14px; font-weight:600; margin-bottom:6px; line-height:1.3;">{title[:80]}</div>
                    <div style="color:#94a3b8; font-size:12px;">👤 {user_name}</div>
                    <div style="color:#94a3b8; font-size:11px; margin-top:4px;">👁 {stats.get('totalViews', 0)} views · ❤️ {stats.get('likes', 0)} likes</div>
                </div>
                """, unsafe_allow_html=True)
                
                if thumbnail:
                    try:
                        st.sidebar.image(thumbnail, width="stretch")
                    except Exception:
                        pass
                
                # Show resolved URL (truncated)
                st.sidebar.caption(f"🔗 {stream_url[:60]}...")
                
                source_path = stream_url
                st.session_state['booca_stream_info'] = info
                st.session_state['stream_referer'] = 'https://booca.online/'
                
                if not is_live and info.get('is_vod'):
                    duration = info.get('duration', 0)
                    if duration > 0:
                        mins = duration // 60
                        secs = duration % 60
                        st.sidebar.info(f"⏱️ Duration: {mins}m {secs}s")
                    st.sidebar.info("💡 Tip: For VOD, you can also use **Offline Processing** mode for full video analysis.")
            elif stream_url is None:
                error_msg = info if isinstance(info, str) else "Unknown error"
                st.sidebar.error(f"❌ {error_msg}")
    
    elif source_type == "URL Stream":
        st.sidebar.markdown("### 🌐 Stream URL")
        
        with st.sidebar.expander("ℹ️ Supported Platforms & Examples"):
            st.markdown("""
            **Supported:**
            - YouTube Live
            - Twitch
            - Facebook Live
            - Booca.online (Live & VOD)
            - Direct streams (HLS, RTSP, MP4)
            
            **Examples:**
            - Booca Live: `https://booca.online/livestream/watch/{id}`
            - Booca VOD: `https://booca.online/livestream/vod/{id}`
            - YouTube: `https://youtube.com/watch?v=...`
            - Twitch: `https://twitch.tv/channel_name`
            - Phone cam: `http://192.168.1.5:8080/video`
            - RTSP: `rtsp://camera_ip:554/stream`
            """)
        
        url_input = st.sidebar.text_input(
            "Enter Stream URL",
            value="http://192.168.1.5:8080/video",
            help="Enter direct stream URL or webpage URL (YouTube, Twitch, Booca, etc.)"
        )
        
        if url_input:
            source_path = resolve_stream_url(url_input)
            if source_path != url_input:
                st.sidebar.success(f"✅ Stream resolved!")
                # Store original URL for yt-dlp pipe fallback
                st.session_state['resolved_from_url'] = url_input
    
    # Session state for realtime processing
    if "analysis_running" not in st.session_state:
        st.session_state.analysis_running = False
        
    # Auto start if provided via query params and resolved successfully
    if injected_stream_id and source_path is not None and "auto_started" not in st.session_state:
        st.session_state.analysis_running = True
        st.session_state.auto_started = True
    
    def start_analysis():
        st.session_state.analysis_running = True
    
    def stop_analysis():
        st.session_state.analysis_running = False
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if not st.session_state.analysis_running:
            st.button("▶️ Start", on_click=start_analysis, type="primary", use_container_width="always", disabled=source_path is None)
        else:
            st.button("⏸️ Stop", on_click=stop_analysis, type="secondary", use_container_width="always")
    
    # Main content area
    if st.session_state.analysis_running and source_path is not None:
        # Broadcast pusher init
        pusher = None
        if enable_broadcast and _HAS_PUSHER:
            try:
                pusher = BroadcastPusher("http://localhost:8502", enabled=True)
            except Exception:
                pass

        # Header row: status + fullscreen
        h_col1, h_col2 = st.columns([5, 1])
        with h_col1:
            st_status = st.empty()
            st_status.info("🔄 Initializing models...")
        with h_col2:
            st.markdown('<a href="http://localhost:8502/mjpeg" target="_blank"><button style="background:#262730;color:#fff;border:none;border-radius:6px;padding:6px 16px;cursor:pointer;width:100%">⛶ Fullscreen</button></a>', unsafe_allow_html=True)

        # Video display — st.image() shows frames directly
        st_frame = st.empty()
        st_minimap = st.empty()

        # Sidebar controls
        rt_zoom = st.sidebar.slider("Zoom", 1.0, 3.0, 1.0, 0.25, key="rt_zoom_level")
        st_debug_area = st.empty()

        try:
            cap = None
            _is_booca = isinstance(source_path, str) and (
                "stream.booca.online" in source_path or
                "b-cdn.net" in source_path or
                st.session_state.get('booca_stream_info') is not None
            )
            stream_referer = st.session_state.get('stream_referer', 'https://watch.rkplayer.xyz/')
            original_url = st.session_state.get('original_player_url', '')

            # Booca streams: set correct headers before first attempt
            if _is_booca and isinstance(source_path, str) and ".m3u8" in source_path:
                _booca_ffmpeg_opts = (
                    "protocol_whitelist;file,http,https,tcp,tls,crypto"
                    "|headers;User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n"
                    "Referer: https://booca.online/\r\n"
                    "Origin: https://booca.online"
                    "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
                    "|rw_timeout;15000000"
                )
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _booca_ffmpeg_opts
                st_status.info("🔄 Connecting to Booca stream...")
                cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
                
                # Retry without reconnect params
                if not cap.isOpened():
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "protocol_whitelist;file,http,https,tcp,tls,crypto"
                        "|headers;User-Agent: Mozilla/5.0\r\n"
                        "Referer: https://booca.online/\r\n"
                        "Origin: https://booca.online"
                    )
                    cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)

                # Fix BunnyCDN Referer drop bug using our background HTTP proxy
                if not cap.isOpened():
                    proxy = get_hls_proxy()
                    proxied_url = proxy.get_proxy_url(source_path)
                    cap = cv2.VideoCapture(proxied_url, cv2.CAP_FFMPEG)
            else:
                if use_nvdec:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuda|video_codec;h264_cuvid"

                cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
                if not cap.isOpened() and isinstance(source_path, str) and ".m3u8" in source_path:
                    # Fix BunnyCDN Referer drop bug using our background HTTP proxy
                    proxy = get_hls_proxy()
                    proxied_url = proxy.get_proxy_url(source_path)
                    cap = cv2.VideoCapture(proxied_url, cv2.CAP_FFMPEG)
                if not cap.isOpened() and isinstance(source_path, str) and "fbcdn.net" in source_path:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "headers;User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n"
                        "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
                    )
                    cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(source_path)

            if isinstance(source_path, str) and (source_path.startswith("http") or source_path.startswith("rtsp")):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)

            if not cap.isOpened():
                st_error = st_debug_area.empty()
                if _is_booca:
                    _booca_info = st.session_state.get('booca_stream_info', {})
                    _stream_status = _booca_info.get('status', 'unknown')
                    if _stream_status not in ('live', 'vod_ready'):
                        st_error.error(f"❌ Cannot open — stream status is **{_stream_status}**. Only 🔴 LIVE or 📹 VOD (recorded) streams work in realtime.")
                    else:
                        st_error.error(f"❌ Cannot connect to Booca stream. URL: `{source_path[:120]}`. OpenCV failed to read frame.")
                else:
                    st_error.error(f"Cannot open: {source_path[:80]}")
            else:
                # Read initial frame — skip up to 30 corrupt frames (common with HLS mid-join)
                ret = False
                frame = None
                for _init_attempt in range(30):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        break
                
                if not ret or frame is None:
                    st_debug_area.error("Cannot read from source")
                else:
                    if resolution_scale < 1.0:
                        h, w = frame.shape[:2]
                        new_w, new_h = int(w * resolution_scale), int(h * resolution_scale)
                        frame = cv2.resize(frame, (new_w, new_h))

                    tracker = Tracker(model_path, class_ids, verbose=False, fp16=use_fp16, imgsz=imgsz)
                    camera_movement_estimator = CameraMovementEstimator(frame, class_ids, verbose=False)
                    team_assigner = TeamAssigner()
                    player_assigner = PlayerBallAssigner()
                    teams_assigned = False
                    frame_count = 0
                    _fps_start = time.time()
                    
                    # Initialize advanced analyzers for realtime
                    rt_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    rt_frame_h, rt_frame_w = frame.shape[:2]
                    rt_speed_estimator = SpeedEstimator(fps=rt_fps)
                    rt_tactical_analyzer = TacticalAnalyzer()
                    rt_event_detector = EventDetector(frame_width=rt_frame_w, frame_height=rt_frame_h)
                    rt_events_log = []

                    st_status.success("✅ Processing... Press Stop to end")
                    
                    # ================================================================
                    # 📊 LIVE ANALYTICS DASHBOARD — Premium UI
                    # ================================================================
                    rt_theme = get_theme_mode()
                    rt_is_dark = rt_theme == "dark"

                    # --- Design tokens ---
                    _card = "rgba(30,41,59,0.85)" if rt_is_dark else "rgba(255,255,255,0.92)"
                    _card_border = "rgba(51,65,85,0.5)" if rt_is_dark else "rgba(226,232,240,0.8)"
                    _glass = "backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);"
                    _tp = "#f1f5f9" if rt_is_dark else "#0f172a"
                    _ts = "#94a3b8" if rt_is_dark else "#64748b"
                    _t3 = "#64748b" if rt_is_dark else "#94a3b8"
                    _blue = "#3b82f6"
                    _blue_g = "linear-gradient(135deg,#3b82f6,#60a5fa)"
                    _red = "#ef4444"
                    _green = "#22c55e"
                    _amber = "#f59e0b"
                    _purple = "#a855f7"
                    _cyan = "#06b6d4"
                    _divider = "rgba(148,163,184,0.15)"

                    st.markdown("---")

                    # Dashboard header with live pulse
                    dash_header = f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px;"><div style="width:4px;height:28px;background:{_blue_g};border-radius:2px;"></div><span style="color:{_tp};font-size:18px;font-weight:700;letter-spacing:-0.3px;">Live Match Analytics</span><span style="background:{_green};color:#fff;font-size:10px;font-weight:600;padding:2px 8px;border-radius:10px;text-transform:uppercase;letter-spacing:0.5px;">● LIVE</span></div>'
                    st.markdown(dash_header, unsafe_allow_html=True)

                    # Row 1: 3 metric cards
                    rt_row1 = st.columns(3)
                    rt_possession_placeholder = rt_row1[0].empty()
                    rt_speed_placeholder = rt_row1[1].empty()
                    rt_space_placeholder = rt_row1[2].empty()

                    # Row 2: Events + AI Commentary
                    rt_row2 = st.columns([1, 1])
                    rt_events_placeholder = rt_row2[0].empty()
                    rt_commentary_placeholder = rt_row2[1].empty()

                    # Row 3: Player performance table
                    rt_player_table_placeholder = st.empty()

                    # Commentary generator
                    def _gen_commentary(t1p, t2p, top_pl, evts, pl_cnt, f_cnt, fps_v):
                        lines = []
                        es = f_cnt / fps_v if fps_v > 0 else 0
                        em, esec = int(es // 60), int(es % 60)
                        if t1p > 65:
                            lines.append(f"⚽ Team 1 đang kiểm soát bóng áp đảo với {t1p}% possession.")
                        elif t2p > 65:
                            lines.append(f"⚽ Team 2 đang thống trị quyền sở hữu bóng với {t2p}%.")
                        elif abs(t1p - t2p) < 10:
                            lines.append("⚖️ Hai đội đang cân bằng về khả năng kiểm soát bóng.")
                        if top_pl and len(top_pl) > 0:
                            t = top_pl[0]
                            if t["speed"] > 25:
                                lines.append(f"🔥 #{t['player_id']} đang sprint mạnh — {t['speed']} km/h!")
                            elif t["speed"] > 18:
                                lines.append(f"🏃 #{t['player_id']} dẫn đầu tốc độ với {t['speed']} km/h.")
                        goals = [e for e in evts if e.get("event") == "goal"]
                        passes = [e for e in evts if e.get("event") == "pass"]
                        if goals:
                            lines.append(f"🥅 BÀN THẮNG! Team {goals[-1].get('scoring_team','?')} ghi bàn!")
                        if len(passes) > 10:
                            lines.append(f"🔄 {len(passes)} đường chuyền — trận đấu sôi nổi!")
                        elif len(passes) > 0:
                            lines.append(f"📊 Đã phát hiện {len(passes)} đường chuyền.")
                        lines.append(f"⏱ {em}:{esec:02d} — Đang theo dõi {pl_cnt} cầu thủ trên sân.")
                        return lines

                    while cap.isOpened():
                        if not st.session_state.analysis_running:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            st_debug_area.warning("Stream ended")
                            break
                        # Skip corrupt/empty frames from H264 decode errors
                        if frame is None or frame.size == 0:
                            continue
                        frame_count += 1
                        if frame_count % frame_skip != 0:
                            continue
                        if resolution_scale < 1.0:
                            h, w = frame.shape[:2]
                            new_w, new_h = int(w * resolution_scale), int(h * resolution_scale)
                            frame = cv2.resize(frame, (new_w, new_h))
                        tracks = tracker.get_object_tracks_single_frame(frame)
                        tracker.add_position_to_tracks_single_frame(tracks)
                        camera_movement = camera_movement_estimator.get_camera_movement_single_frame(frame)
                        camera_movement_estimator.adjust_positions_to_tracks_single_frame(tracks, camera_movement)
                        players_in_frame = tracks.get("players", {})
                        if not teams_assigned and len(players_in_frame) > 0:
                            team_assigner.assign_team_colour(frame, players_in_frame, force=True)
                            teams_assigned = True
                        if teams_assigned:
                            for player_id, player_track in players_in_frame.items():
                                team = team_assigner.get_player_team(frame, player_track["bbox"], player_id)
                                tracks["players"][player_id]["team"] = team
                                tracks["players"][player_id]["team_colour"] = team_assigner.team_colours[team]
                        player_assigner.assign_ball_single_frame(tracks)
                        if player_assigner.ball_possession:
                            sanitized_possession = [-1 if x is None else x for x in player_assigner.ball_possession]
                            ball_possession_np = np.array(sanitized_possession)
                        else:
                            ball_possession_np = None
                        output_frame = tracker.draw_annotations_single_frame(frame, tracks, ball_possession_np)
                        output_frame = camera_movement_estimator.draw_camera_movement_single_frame(output_frame, camera_movement)
                        
                        # === Advanced realtime analytics ===
                        rt_top_players = rt_speed_estimator.update_speeds(tracks, frame_count)
                        rt_tactical_analyzer.update_heatmap(tracks)
                        
                        # Pass detection
                        rt_current_possessor = None
                        for pid, pdata in tracks.get("players", {}).items():
                            if pdata.get("has_ball", False):
                                rt_current_possessor = pid
                                break
                        rt_pass_evt = rt_tactical_analyzer.detect_passes(tracks, rt_current_possessor, frame_count)
                        if rt_pass_evt:
                            rt_events_log.append(rt_pass_evt)
                        
                        # Goal detection
                        rt_goal_evt = rt_event_detector.check_events(tracks, frame_count)
                        if rt_goal_evt:
                            rt_goal_evt["frame"] = frame_count
                            rt_events_log.append(rt_goal_evt)

                        # Push to broadcast server
                        if pusher is not None:
                            pusher.push(output_frame)

                        # Apply zoom before display
                        if rt_zoom > 1.0:
                            h, w = output_frame.shape[:2]
                            crop_w = int(w / rt_zoom)
                            crop_h = int(h / rt_zoom)
                            x1 = (w - crop_w) // 2
                            y1 = (h - crop_h) // 2
                            output_frame = output_frame[y1:y1+crop_h, x1:x1+crop_w]
                            output_frame = cv2.resize(output_frame, (w, h))

                        # Display frame in Streamlit
                        # Convert BGR (OpenCV) → RGB for Streamlit display
                        display_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                        st_frame.image(display_frame, channels="RGB")

                        try:
                            mm = render_minimap_rgb(tracks, width=320, height=180)
                            st_minimap.image(
                                mm,
                                channels="RGB",
                                caption="Minimap — bóng (vàng) & cầu thủ cầm bóng (viền theo đội)",
                            )
                        except Exception:
                            pass

                        # Update live stats every 30 frames
                        if frame_count % 30 == 0:
                            elapsed = time.time() - _fps_start
                            current_fps = frame_count / elapsed if elapsed > 0 else 0
                            
                            # --- Data prep ---
                            rt_bp = player_assigner.ball_possession
                            if rt_bp and len(rt_bp) > 0:
                                rt_bp_arr = np.array([-1 if x is None else x for x in rt_bp])
                                rt_t1 = int(np.sum(rt_bp_arr == 1))
                                rt_t2 = int(np.sum(rt_bp_arr == 2))
                                rt_total = rt_t1 + rt_t2
                                rt_t1_pct = round(rt_t1 / rt_total * 100) if rt_total > 0 else 50
                                rt_t2_pct = 100 - rt_t1_pct
                            else:
                                rt_t1_pct, rt_t2_pct = 50, 50

                            # ═══════════════════════════════════════════
                            # CARD 1: Possession
                            # ═══════════════════════════════════════════
                            poss_html = f'<div style="background:{_card};border:1px solid {_card_border};border-radius:14px;padding:18px 16px;{_glass}position:relative;overflow:hidden;">'
                            poss_html += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:{_blue_g};"></div>'
                            poss_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;"><span style="font-size:14px;">⚽</span><span style="color:{_tp};font-weight:700;font-size:13px;letter-spacing:-0.2px;">Ball Possession</span></div>'
                            poss_html += f'<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:10px;">'
                            poss_html += f'<div style="text-align:center;"><div style="color:{_blue};font-weight:800;font-size:28px;line-height:1;">{rt_t1_pct}%</div><div style="color:{_ts};font-size:10px;margin-top:2px;text-transform:uppercase;letter-spacing:0.8px;">Team 1</div></div>'
                            poss_html += f'<div style="color:{_t3};font-size:11px;font-weight:500;">VS</div>'
                            poss_html += f'<div style="text-align:center;"><div style="color:{_red};font-weight:800;font-size:28px;line-height:1;">{rt_t2_pct}%</div><div style="color:{_ts};font-size:10px;margin-top:2px;text-transform:uppercase;letter-spacing:0.8px;">Team 2</div></div></div>'
                            poss_html += f'<div style="background:rgba(148,163,184,0.15);border-radius:6px;height:10px;overflow:hidden;">'
                            poss_html += f'<div style="background:{_blue_g};height:100%;width:{rt_t1_pct}%;border-radius:6px;box-shadow:0 0 8px rgba(59,130,246,0.4);"></div></div></div>'
                            rt_possession_placeholder.markdown(poss_html, unsafe_allow_html=True)

                            # ═══════════════════════════════════════════
                            # CARD 2: Top Speeds
                            # ═══════════════════════════════════════════
                            speed_html = f'<div style="background:{_card};border:1px solid {_card_border};border-radius:14px;padding:18px 16px;{_glass}position:relative;overflow:hidden;">'
                            speed_html += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(135deg,{_amber},{_green});"></div>'
                            speed_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;"><span style="font-size:14px;">🏃</span><span style="color:{_tp};font-weight:700;font-size:13px;letter-spacing:-0.2px;">Top Speeds</span></div>'
                            if rt_top_players:
                                medals = ["🥇", "🥈", "🥉", ""]
                                for idx, tp in enumerate(rt_top_players[:4]):
                                    t_color = _blue if tp["team"] == 1 else _red
                                    bar_pct = min(tp["speed"] / 35 * 100, 100)
                                    medal = medals[min(idx, 3)]
                                    speed_html += f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid {_divider};">'
                                    speed_html += f'<span style="font-size:12px;width:16px;">{medal}</span>'
                                    speed_html += f'<span style="color:{t_color};font-size:11px;font-weight:600;width:30px;">#{tp["player_id"]}</span>'
                                    speed_html += f'<div style="flex:1;background:rgba(148,163,184,0.1);border-radius:3px;height:6px;overflow:hidden;"><div style="background:{t_color};height:100%;width:{bar_pct}%;border-radius:3px;"></div></div>'
                                    speed_html += f'<span style="color:{_tp};font-size:11px;font-weight:700;min-width:52px;text-align:right;">{tp["speed"]} km/h</span></div>'
                            else:
                                speed_html += f'<div style="color:{_ts};font-size:12px;text-align:center;padding:10px 0;">⏳ Measuring...</div>'
                            speed_html += '</div>'
                            rt_speed_placeholder.markdown(speed_html, unsafe_allow_html=True)

                            # ═══════════════════════════════════════════
                            # CARD 3: Space Control + Formations
                            # ═══════════════════════════════════════════
                            rt_tactics = rt_tactical_analyzer.analyze_tactics(tracks)
                            rt_sc = rt_tactics.get("space_control", {"team1": 50.0, "team2": 50.0})
                            rt_formations = rt_tactics.get("formations", {})
                            rt_f1 = rt_formations.get(1, 'N/A')
                            rt_f2 = rt_formations.get(2, 'N/A')

                            space_html = f'<div style="background:{_card};border:1px solid {_card_border};border-radius:14px;padding:18px 16px;{_glass}position:relative;overflow:hidden;">'
                            space_html += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(135deg,{_purple},{_cyan});"></div>'
                            space_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;"><span style="font-size:14px;">🗺️</span><span style="color:{_tp};font-weight:700;font-size:13px;letter-spacing:-0.2px;">Space Control</span></div>'
                            space_html += f'<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:10px;">'
                            space_html += f'<div style="text-align:center;"><div style="color:{_blue};font-weight:800;font-size:22px;line-height:1;">{rt_sc["team1"]}%</div></div>'
                            space_html += f'<div style="color:{_t3};font-size:11px;font-weight:500;">VS</div>'
                            space_html += f'<div style="text-align:center;"><div style="color:{_red};font-weight:800;font-size:22px;line-height:1;">{rt_sc["team2"]}%</div></div></div>'
                            space_html += f'<div style="background:rgba(148,163,184,0.15);border-radius:6px;height:10px;overflow:hidden;">'
                            space_html += f'<div style="background:{_blue_g};height:100%;width:{rt_sc["team1"]}%;border-radius:6px;"></div></div>'
                            space_html += f'<div style="display:flex;justify-content:space-between;margin-top:8px;">'
                            space_html += f'<div style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:6px;padding:3px 8px;"><span style="color:{_blue};font-size:10px;font-weight:600;">🔵 {rt_f1}</span></div>'
                            space_html += f'<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.2);border-radius:6px;padding:3px 8px;"><span style="color:{_red};font-size:10px;font-weight:600;">🔴 {rt_f2}</span></div></div></div>'
                            rt_space_placeholder.markdown(space_html, unsafe_allow_html=True)

                            # ═══════════════════════════════════════════
                            # CARD 4: Events Feed
                            # ═══════════════════════════════════════════
                            rt_t1_passes = sum(1 for e in rt_events_log if e.get("event") == "pass" and e.get("scoring_team", e.get("scoringTeam")) == 1)
                            rt_t2_passes = sum(1 for e in rt_events_log if e.get("event") == "pass" and e.get("scoring_team", e.get("scoringTeam")) == 2)
                            rt_t1_goals = sum(1 for e in rt_events_log if e.get("event") == "goal" and e.get("scoring_team") == 1)
                            rt_t2_goals = sum(1 for e in rt_events_log if e.get("event") == "goal" and e.get("scoring_team") == 2)

                            evt_html = f'<div style="background:{_card};border:1px solid {_card_border};border-radius:14px;padding:18px 16px;{_glass}position:relative;overflow:hidden;min-height:200px;">'
                            evt_html += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(135deg,{_green},{_amber});"></div>'
                            evt_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:14px;"><span style="font-size:14px;">📋</span><span style="color:{_tp};font-weight:700;font-size:13px;letter-spacing:-0.2px;">Match Events</span><span style="margin-left:auto;background:rgba(148,163,184,0.15);color:{_ts};font-size:10px;padding:2px 6px;border-radius:8px;">{len(rt_events_log)} total</span></div>'
                            evt_html += f'<div style="display:flex;gap:12px;margin-bottom:12px;">'
                            evt_html += f'<div style="flex:1;background:rgba(59,130,246,0.08);border-radius:8px;padding:8px;text-align:center;"><div style="color:{_blue};font-size:18px;font-weight:800;">{rt_t1_goals}</div><div style="color:{_ts};font-size:9px;text-transform:uppercase;">Goals T1</div></div>'
                            evt_html += f'<div style="flex:1;background:rgba(239,68,68,0.08);border-radius:8px;padding:8px;text-align:center;"><div style="color:{_red};font-size:18px;font-weight:800;">{rt_t2_goals}</div><div style="color:{_ts};font-size:9px;text-transform:uppercase;">Goals T2</div></div></div>'
                            evt_html += f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid {_divider};"><span style="color:{_ts};font-size:11px;">🔄 Passes</span><span style="color:{_tp};font-size:11px;font-weight:600;"><span style="color:{_blue}">{rt_t1_passes}</span> — <span style="color:{_red}">{rt_t2_passes}</span></span></div>'
                            evt_html += f'<div style="display:flex;justify-content:space-between;padding:4px 0;"><span style="color:{_ts};font-size:11px;">📊 On Field</span><span style="color:{_tp};font-size:11px;font-weight:600;">{len(players_in_frame)} players · {current_fps:.0f} FPS</span></div>'
                            for evt in rt_events_log[-3:]:
                                etype = evt.get("event", "")
                                if etype == "goal":
                                    evt_html += f'<div style="margin-top:6px;background:rgba(34,197,94,0.1);border-left:3px solid {_green};padding:4px 8px;border-radius:0 6px 6px 0;"><span style="color:{_green};font-size:11px;font-weight:600;">⚽ GOAL — Team {evt.get("scoring_team","?")}</span></div>'
                                elif etype == "pass":
                                    evt_html += f'<div style="margin-top:4px;padding-left:8px;"><span style="color:{_t3};font-size:10px;">🔄 #{evt.get("from_player","?")} → #{evt.get("to_player","?")}</span></div>'
                            evt_html += '</div>'
                            rt_events_placeholder.markdown(evt_html, unsafe_allow_html=True)

                            # ═══════════════════════════════════════════
                            # CARD 5: AI Automated Commentary
                            # ═══════════════════════════════════════════
                            rt_comm_lines = _gen_commentary(rt_t1_pct, rt_t2_pct, rt_top_players, rt_events_log, len(players_in_frame), frame_count, rt_fps)
                            comm_html = f'<div style="background:{_card};border:1px solid {_card_border};border-radius:14px;padding:18px 16px;{_glass}position:relative;overflow:hidden;min-height:200px;">'
                            comm_html += f'<div style="position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(135deg,{_cyan},#818cf8);"></div>'
                            comm_html += f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;"><span style="font-size:14px;">🤖</span><span style="color:{_tp};font-weight:700;font-size:13px;letter-spacing:-0.2px;">AI Commentary</span><span style="margin-left:auto;color:{_cyan};font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Auto</span></div>'
                            for cl in rt_comm_lines:
                                comm_html += f'<div style="color:{_ts};font-size:11px;line-height:1.6;padding:3px 0;border-bottom:1px solid {_divider};">{cl}</div>'
                            if not rt_comm_lines:
                                comm_html += f'<div style="color:{_t3};font-size:11px;text-align:center;padding:16px 0;">Đang phân tích trận đấu...</div>'
                            comm_html += '</div>'
                            rt_commentary_placeholder.markdown(comm_html, unsafe_allow_html=True)

                            # ═══════════════════════════════════════════
                            # Player Performance Table (every 90 frames)
                            # ═══════════════════════════════════════════
                            if frame_count % 90 == 0 and rt_speed_estimator.total_distances:
                                rt_sorted = sorted(
                                    rt_speed_estimator.total_distances.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:10]

                                rt_perf_data = []
                                for rk, (pid, dist) in enumerate(rt_sorted, 1):
                                    p_team = tracks.get("players", {}).get(pid, {}).get("team", "?")
                                    t_label = "🔵 T1" if p_team == 1 else "🔴 T2" if p_team == 2 else "⚪"
                                    rt_perf_data.append({
                                        "#": rk,
                                        "ID": int(pid),
                                        "Team": t_label,
                                        "Dist(m)": f"{dist * rt_speed_estimator.pixel_to_meter_scale:.0f}" if dist < 100 else f"{dist:.0f}",
                                        "Speed": f"{rt_speed_estimator.current_speeds.get(pid, 0):.1f}",
                                        "Sprints": rt_speed_estimator.sprint_counts.get(pid, 0),
                                    })

                                if rt_perf_data:
                                    import pandas as _pd_rt
                                    rt_perf_df = _pd_rt.DataFrame(rt_perf_data)
                                    rt_player_table_placeholder.dataframe(rt_perf_df, use_container_width=True, hide_index=True)



                    cap.release()
                    st_debug_area.info("⏹️ Stopped")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    elif st.session_state.analysis_running and source_path is None:
        st.error("❌ Please select a valid video source")
    else:
        # Instructions
        st.info("👈 Configure settings in the sidebar and click **Start** to begin analysis")
        
        st.markdown("""
        ### 🎯 Features
        - **Realtime Analysis**: Process live video streams with minimal latency
        - **Multiple Sources**: Webcam, video file, URL stream, or **Booca Stream**
        - **Booca Integration**: Analyze live streams & recorded VODs from Booca.online
        - **Player Tracking**: Individual player identification and tracking
        - **Team Assignment**: Automatic team color detection
        - **Ball Possession**: Real-time possession statistics
        - **Camera Movement**: Compensate for camera panning/zooming
        
        ### 📝 Quick Start
        1. Select **Analysis Mode** (Offline or Realtime)
        2. Choose **Video Source** type
        3. Configure visualization options
        4. Click **Start Analysis**
        
        ### 🟢 Booca Stream
        Paste a Booca URL to instantly analyze football streams:
        - 🔴 **Live:** `https://booca.online/livestream/watch/{id}`
        - 📹 **VOD:** `https://booca.online/livestream/vod/{id}`
        
        ### 📱 Using Phone as Camera
        Install **DroidCam** or **IP Webcam** app on your phone, connect to same WiFi, 
        then use URL Stream with format: `http://YOUR_PHONE_IP:8080/video`
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MatchVision** | Computer Vision Football Analysis")
