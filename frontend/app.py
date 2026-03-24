import streamlit as st
import cv2
import tempfile
import time
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath("."))

from utils import options, read_video, save_video
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

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
                    tracker = Tracker("models/best.pt", class_ids, verbose=False)
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
                    
                    # Display result
                    st.subheader("📹 Processed Video")
                    st.video(output_path)
                    
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
            
            ydl_opts = {
                'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'socket_timeout': 30,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and 'url' in info:
                    resolved_url = info['url']
                    st.sidebar.success("✅ Resolved via yt-dlp")
                    return resolved_url
                elif info and 'formats' in info and len(info['formats']) > 0:
                    # Get best format with height <= 720p
                    formats = [f for f in info['formats'] if f.get('url') and f.get('vcodec') != 'none']
                    if formats:
                        resolved_url = formats[-1]['url']
                        st.sidebar.success("✅ Resolved via yt-dlp")
                        return resolved_url
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
        st_frame = st.empty()
        st_status = st.empty()
        st_debug = st.empty()  # Debug area
        
        st_status.info("🔄 Initializing models...")
        
        # Debug expander
        debug_expander = st.expander("🔧 Debug Info", expanded=True)
        
        try:
            # For HLS streams that require headers, use ffmpeg options
            cap = None
            
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
            else:
                # Normal capture for non-HLS sources
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
                    tracker = Tracker("models/best.pt", class_ids, verbose=False)
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
                            team_assigner.assign_team_colour(frame, players_in_frame)
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
                        
                        # Convert BGR to RGB
                        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display
                        st_frame.image(output_frame, channels="RGB", use_column_width=True)
                        
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
