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

from utils import options, read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

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

# Mode Selection
st.sidebar.subheader("📊 Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Select Mode",
    ["Offline Processing", "Realtime Analysis"],
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
    min_value=1, max_value=5, value=2,
    help="Higher = faster but less smooth. 1 = process every frame, 3 = process every 3rd frame"
)
resolution_scale = st.sidebar.slider(
    "Resolution Scale",
    min_value=0.25, max_value=1.0, value=0.5, step=0.25,
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
    value=False,
    help="Enable half-precision inference. Requires NVIDIA GPU. ~2x throughput."
)
imgsz = st.sidebar.slider(
    "YOLO Input Size",
    min_value=320, max_value=1280, value=640, step=64,
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
        ["Demo Video", "Upload Video"]
    )
    
    video_data = None
    video_name = None
    
    if source_type == "Demo Video":
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
    
    # Process Button
    if video_data:
        if st.sidebar.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing video... This may take several minutes."):
                try:
                    # Process video
                    frames, fps, _, _ = read_video(video_data, verbose=False)
                    
                    progress_bar = st.progress(0, text="Initializing models...")
                    
                    # Track objects
                    tracker = Tracker(model_path, class_ids, verbose=False, fp16=use_fp16, imgsz=imgsz)
                    progress_bar.progress(10, text="Tracking objects...")
                    
                    tracks = tracker.get_object_tracks(frames)
                    tracker.add_position_to_tracks(tracks)
                    progress_bar.progress(30, text="Estimating camera movement...")
                    
                    # Camera movement
                    camera_movement_estimator = CameraMovementEstimator(frames[0], class_ids, verbose=False)
                    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames)
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
                    progress_bar.progress(85, text="Drawing annotations...")
                    
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
                    components.html(player_html, height=620, scrolling=False)
                    
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
        ["Webcam", "Video File", "URL Stream"]
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
        
        # Method 0: Extract m3u8 from embed player pages (rkplayer, cakhia, etc.)
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
    
    elif source_type == "URL Stream":
        st.sidebar.markdown("### 🌐 Stream URL")
        
        with st.sidebar.expander("ℹ️ Supported Platforms & Examples"):
            st.markdown("""
            **Supported:**
            - YouTube Live
            - Twitch
            - Facebook Live
            - Direct streams (HLS, RTSP, MP4)
            
            **Examples:**
            - YouTube: `https://youtube.com/watch?v=...`
            - Twitch: `https://twitch.tv/channel_name`
            - Phone cam: `http://192.168.1.5:8080/video`
            - RTSP: `rtsp://camera_ip:554/stream`
            """)
        
        url_input = st.sidebar.text_input(
            "Enter Stream URL",
            value="http://192.168.1.5:8080/video",
            help="Enter direct stream URL or webpage URL (YouTube, Twitch, etc.)"
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
    
    def start_analysis():
        st.session_state.analysis_running = True
    
    def stop_analysis():
        st.session_state.analysis_running = False
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if not st.session_state.analysis_running:
            st.button("▶️ Start", on_click=start_analysis, type="primary", use_container_width=True, disabled=source_path is None)
        else:
            st.button("⏸️ Stop", on_click=stop_analysis, type="secondary", use_container_width=True)
    
    # Main content area
    if st.session_state.analysis_running and source_path is not None:
        # Zoom controls for realtime mode
        zoom_col1, zoom_col2, zoom_col3 = st.columns([1, 2, 1])
        with zoom_col2:
            rt_zoom = st.slider(
                "🔍 Zoom Level",
                min_value=1.0, max_value=3.0, value=1.0, step=0.25,
                key="rt_zoom_level",
                help="Zoom into the video frame. 1x = normal, 3x = maximum zoom"
            )
        
        st_frame = st.empty()
        st_status = st.empty()
        st_debug = st.empty()  # Debug area

        # Initialize broadcast pusher (non-blocking HTTP POST)
        pusher = None
        if enable_broadcast and _HAS_PUSHER:
            try:
                pusher = BroadcastPusher("http://localhost:8502", enabled=True)
            except Exception as e:
                st.sidebar.warning(f"Broadcast init failed: {e}")

        # Broadcast status display
        if enable_broadcast:
            st_bc = st.sidebar.container()
            st_bc.info("📡 **Broadcast Active** — Open: http://localhost:8502/mjpeg")
            if broadcast_url:
                st_bc.caption(f"RTMP: {broadcast_url[:50]}...")
            else:
                st_bc.caption("MJPEG only (no RTMP)")

        st_status.info("🔄 Initializing models...")
        
        # Debug expander
        debug_expander = st.expander("🔧 Debug Info", expanded=True)
        
        try:
            # For HLS streams that require headers, use ffmpeg options
            cap = None

            # Enable NVDEC hardware decode for NVIDIA GPUs (set before any VideoCapture)
            # Note: m3u8 referer setting below will override this for HLS streams
            if use_nvdec:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuda|video_codec;h264_cuvid"

            # Get stored referer if available
            stream_referer = st.session_state.get('stream_referer', 'https://watch.rkplayer.xyz/')
            original_url = st.session_state.get('original_player_url', '')
            
            with debug_expander:
                st.write(f"**Source URL:** `{source_path[:100]}...`" if len(str(source_path)) > 100 else f"**Source URL:** `{source_path}`")
                st.write(f"**Is m3u8:** {'.m3u8' in str(source_path)}")
                st.write(f"**Referer:** `{stream_referer}`")
            
            if isinstance(source_path, str) and ".m3u8" in source_path:
                with debug_expander:
                    st.write("**Method 1:** Trying ffmpeg with headers...")
                
                # Try with ffmpeg and proper options for HLS - use stored referer
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"protocol_whitelist;file,http,https,tcp,tls,crypto|headers;User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36\r\nReferer: {stream_referer}\r\nOrigin: {stream_referer}"
                cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
                
                with debug_expander:
                    st.write(f"  → Result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
                
                if not cap.isOpened():
                    # Alternative: try with streamlink as middleman
                    with debug_expander:
                        st.write("**Method 2:** Trying streamlink...")
                    try:
                        import streamlink
                        streams = streamlink.streams(source_path)
                        with debug_expander:
                            st.write(f"  → Available streams: {list(streams.keys()) if streams else 'None'}")
                        if streams:
                            # Try different qualities
                            for quality in ["best", "worst", "720p", "480p"]:
                                if quality in streams:
                                    best_url = streams[quality].url
                                    with debug_expander:
                                        st.write(f"  → Trying quality: {quality}")
                                        st.write(f"  → Stream URL: `{best_url[:80]}...`")
                                    cap = cv2.VideoCapture(best_url, cv2.CAP_FFMPEG)
                                    if cap.isOpened():
                                        with debug_expander:
                                            st.write(f"  → ✅ Success with {quality}")
                                        break
                    except Exception as e:
                        with debug_expander:
                            st.write(f"  → ❌ Streamlink error: {str(e)}")
                
                if not cap or not cap.isOpened():
                    # Method 3: Direct without headers
                    with debug_expander:
                        st.write("**Method 3:** Trying direct connection...")
                    cap = cv2.VideoCapture(source_path)
                    with debug_expander:
                        st.write(f"  → Result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
            elif isinstance(source_path, str) and "fbcdn.net" in source_path:
                # Facebook CDN URLs require User-Agent headers
                with debug_expander:
                    st.write("**Method:** Facebook CDN with headers...")
                
                fb_ffmpeg_opts = (
                    "headers;User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\\r\\n"
                    "|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
                )
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = fb_ffmpeg_opts
                cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
                
                with debug_expander:
                    st.write(f"  → Result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
                
                # Fallback: Use yt-dlp to pipe video through ffmpeg
                if not cap.isOpened():
                    with debug_expander:
                        st.write("**Method 2:** yt-dlp pipe fallback...")
                    try:
                        import subprocess
                        # Get the original Facebook URL from session state
                        original_fb_url = st.session_state.get('resolved_from_url', '')
                        if not original_fb_url:
                            original_fb_url = source_path
                        
                        # Use yt-dlp to stream to stdout, pipe to ffmpeg/OpenCV
                        yt_dlp_cmd = [
                            sys.executable, "-m", "yt_dlp",
                            "-o", "-",  # Output to stdout
                            "--quiet",
                            original_fb_url
                        ]
                        
                        pipe = subprocess.Popen(
                            yt_dlp_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        
                        # Create temp file from yt-dlp output (first few MB for testing)
                        import tempfile
                        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        temp_path = temp_video.name
                        st.session_state['temp_fb_video'] = temp_path
                        
                        # Read first chunk to verify stream works
                        chunk = pipe.stdout.read(1024 * 1024)  # 1MB
                        if chunk:
                            temp_video.write(chunk)
                            # Continue reading in background - write all remaining data
                            remaining = pipe.stdout.read()
                            temp_video.write(remaining)
                            temp_video.close()
                            pipe.wait()
                            
                            cap = cv2.VideoCapture(temp_path, cv2.CAP_FFMPEG)
                            with debug_expander:
                                st.write(f"  → yt-dlp pipe result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
                        else:
                            temp_video.close()
                            pipe.kill()
                            with debug_expander:
                                st.write("  → ❌ yt-dlp pipe: no data received")
                    except Exception as e:
                        with debug_expander:
                            st.write(f"  → ❌ yt-dlp pipe failed: {str(e)[:80]}")
            else:
                # Normal capture for non-HLS, non-Facebook sources
                with debug_expander:
                    st.write("**Method:** Standard ffmpeg capture...")
                cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
                with debug_expander:
                    st.write(f"  → Result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
            
            # If failed, try default backend
            if cap is None or not cap.isOpened():
                with debug_expander:
                    st.write("**Fallback:** Trying default backend...")
                cap = cv2.VideoCapture(source_path)
                with debug_expander:
                    st.write(f"  → Result: {'✅ Opened' if cap.isOpened() else '❌ Failed'}")
            
            # Set parameters for network streams
            if isinstance(source_path, str) and (source_path.startswith("http") or source_path.startswith("rtsp")):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)
            
            if not cap.isOpened():
                st.error(f"❌ Cannot open video source")
                with st.expander("🔍 Troubleshooting Help"):
                    st.markdown(f"""
                    **Failed to open:** `{source_path[:100]}...`
                    
                    **Possible solutions:**
                    1. **For .m3u8 streams**: Install ffmpeg: `pip install ffmpeg-python`
                    2. **For protected streams**: May need authentication or cookies
                    3. **Try yt-dlp**: Install with `pip install yt-dlp` for better support
                    4. **Check URL**: Verify the stream is currently live/accessible
                    5. **Network**: Check firewall/proxy settings
                    
                    **Alternative methods:**
                    - Extract direct .m3u8/.mp4 URL using browser dev tools (Network tab)
                    - Use VLC to test the URL first
                    - Try a different stream quality/format
                    """)
            else:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Cannot read from video source")
                else:
                    # Apply resolution scaling to first frame (must be done before initializing models)
                    if resolution_scale < 1.0:
                        h, w = frame.shape[:2]
                        new_w, new_h = int(w * resolution_scale), int(h * resolution_scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                    
                    # Initialize models
                    tracker = Tracker(model_path, class_ids, verbose=False, fp16=use_fp16, imgsz=imgsz)
                    camera_movement_estimator = CameraMovementEstimator(frame, class_ids, verbose=False)
                    team_assigner = TeamAssigner()
                    player_assigner = PlayerBallAssigner()
                    
                    teams_assigned = False
                    frame_count = 0
                    
                    st_status.success("✅ Processing... Press Stop to end")
                    
                    while cap.isOpened():
                        if not st.session_state.analysis_running:
                            break
                        
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("⚠️ Stream ended")
                            break
                        
                        frame_count += 1
                        
                        # Frame skip for performance
                        if frame_count % frame_skip != 0:
                            continue
                        
                        # Resolution scaling for performance
                        if resolution_scale < 1.0:
                            h, w = frame.shape[:2]
                            new_w, new_h = int(w * resolution_scale), int(h * resolution_scale)
                            frame = cv2.resize(frame, (new_w, new_h))
                        
                        # Process frame
                        tracks = tracker.get_object_tracks_single_frame(frame)
                        tracker.add_position_to_tracks_single_frame(tracks)
                        
                        camera_movement = camera_movement_estimator.get_camera_movement_single_frame(frame)
                        camera_movement_estimator.adjust_positions_to_tracks_single_frame(tracks, camera_movement)
                        
                        # Team assignment
                        players_in_frame = tracks.get("players", {})
                        if not teams_assigned and len(players_in_frame) > 0:
                            team_assigner.assign_team_colour(frame, players_in_frame, force=True)
                            teams_assigned = True
                        
                        if teams_assigned:
                            for player_id, player_track in players_in_frame.items():
                                team = team_assigner.get_player_team(frame, player_track["bbox"], player_id)
                                tracks["players"][player_id]["team"] = team
                                tracks["players"][player_id]["team_colour"] = team_assigner.team_colours[team]
                        
                        # Ball assignment
                        player_assigner.assign_ball_single_frame(tracks)
                        
                        # Draw annotations
                        if player_assigner.ball_possession:
                            sanitized_possession = [-1 if x is None else x for x in player_assigner.ball_possession]
                            ball_possession_np = np.array(sanitized_possession)
                        else:
                            ball_possession_np = None
                        
                        output_frame = tracker.draw_annotations_single_frame(frame, tracks, ball_possession_np)
                        output_frame = camera_movement_estimator.draw_camera_movement_single_frame(output_frame, camera_movement)

                        # Convert BGR to RGB for display
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

                        # Apply realtime zoom (center crop + resize)
                        if rt_zoom > 1.0:
                            h, w = output_frame.shape[:2]
                            crop_w = int(w / rt_zoom)
                            crop_h = int(h / rt_zoom)
                            x1 = (w - crop_w) // 2
                            y1 = (h - crop_h) // 2
                            output_frame = output_frame[y1:y1+crop_h, x1:x1+crop_w]
                            output_frame = cv2.resize(output_frame, (w, h))

                        # Push annotated frame to broadcast server (MJPEG + RTMP)
                        if pusher is not None:
                            pusher.push(output_frame)

                        # Limit display width to prevent oversized video
                        max_display_w = 1280
                        h, w = output_frame.shape[:2]
                        if w > max_display_w:
                            new_h = int(h * max_display_w / w)
                            output_frame = cv2.resize(output_frame, (max_display_w, new_h))

                        st_frame.image(output_frame, channels="RGB")

                        # Update status every 30 frames
                        if frame_count % 30 == 0:
                            st_status.success(f"✅ Processing... Frame {frame_count}")
                    
                    cap.release()
                    st_status.info("⏹️ Analysis stopped")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
    
    elif st.session_state.analysis_running and source_path is None:
        st.error("❌ Please select a valid video source")
    else:
        # Instructions
        st.info("👈 Configure settings in the sidebar and click **Start** to begin analysis")
        
        st.markdown("""
        ### 🎯 Features
        - **Realtime Analysis**: Process live video streams with minimal latency
        - **Multiple Sources**: Webcam, video file, or URL stream
        - **Player Tracking**: Individual player identification and tracking
        - **Team Assignment**: Automatic team color detection
        - **Ball Possession**: Real-time possession statistics
        - **Camera Movement**: Compensate for camera panning/zooming
        
        ### 📝 Quick Start
        1. Select **Analysis Mode** (Offline or Realtime)
        2. Choose **Video Source** type
        3. Configure visualization options
        4. Click **Start Analysis**
        
        ### 📱 Using Phone as Camera
        Install **DroidCam** or **IP Webcam** app on your phone, connect to same WiFi, 
        then use URL Stream with format: `http://YOUR_PHONE_IP:8080/video`
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MatchVision** | Computer Vision Football Analysis")
