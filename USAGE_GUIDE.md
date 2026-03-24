# Football AI Analysis - Complete Usage Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Analysis Modes](#analysis-modes)
3. [Video Sources](#video-sources)
4. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Installation
```bash
cd e:\Football_computerVision\football-computer-vision
pip install -r requirements.txt
```

### Running the Application

**Option 1: Web Interface (Recommended)**
```bash
streamlit run frontend/app.py
```
Then open `http://localhost:8501` in your browser.

**Option 2: Command Line (Realtime)**
```bash
python realtime_main.py --source 0 --tracks players ball referees stats
```

**Option 3: Command Line (Offline)**
```bash
python main.py --video demos/demo1.mp4 --tracks players ball stats
```

---

## 📊 Analysis Modes

### 1. Offline Processing
Process entire video and save annotated output.

**Best for:**
- Full match analysis
- Creating highlight reels
- Detailed statistics generation
- Archival/sharing

**How to use:**
1. Open web interface
2. Select "Offline Processing" mode
3. Upload video or select demo
4. Configure visualization options
5. Click "Start Analysis"
6. Download processed video from `output/output.mp4`

### 2. Realtime Analysis
Process video stream with live visualization.

**Best for:**
- Live match monitoring
- Quick testing
- Low-latency analysis
- Camera calibration

**How to use:**
1. Open web interface
2. Select "Realtime Analysis" mode
3. Choose video source (Webcam/File/URL)
4. Click "Start" to begin
5. Click "Stop" to end

---

## 📹 Video Sources

### 🎥 Webcam / Laptop Camera

**Web Interface:**
1. Select "Realtime Analysis" mode
2. Choose "Webcam" as source
3. Select camera from dropdown
4. Click "Start"

**Command Line:**
```bash
# Default webcam
python realtime_main.py --source 0

# Second camera
python realtime_main.py --source 1
```

---

### 📱 Phone Camera

You can use your phone as a wireless camera!

#### Setup Instructions:

**For Android:**
1. Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) or [DroidCam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam)
2. Open the app and start the server
3. Note the IP address shown (e.g., `192.168.1.5:8080`)
4. Connect phone and computer to **same WiFi network**

**For iOS:**
1. Install [iVCam](https://apps.apple.com/app/ivcam/id1164464478) or [EpocCam](https://apps.apple.com/app/epoccam/id435355256)
2. Follow app instructions
3. Note the connection URL

#### Using Phone Camera:

**Web Interface:**
1. Select "Realtime Analysis" mode
2. Choose "URL Stream" as source
3. Enter URL: `http://YOUR_PHONE_IP:8080/video`
   - Example: `http://192.168.1.5:8080/video`
4. Click "Start"

**Command Line:**
```bash
python realtime_main.py --source "http://192.168.1.5:8080/video"
```

---

### 📁 Video File Upload

**Supported Formats:**
- MP4 (recommended)
- AVI
- MOV
- MKV
- FLV

**Web Interface:**
1. Select analysis mode (Offline or Realtime)
2. Choose "Video File" / "Upload Video"
3. Drag & drop file or click to browse
4. Preview video
5. Click "Start Analysis"

**Command Line (Offline):**
```bash
python main.py --video path/to/video.mp4 --tracks players ball stats
```

**Command Line (Realtime):**
```bash
python realtime_main.py --source path/to/video.mp4
```

**Tips:**
- Maximum recommended file size: 500MB for smooth browser performance
- Higher resolution = longer processing time (offline mode)
- For large files, consider using command line

---

### 🌐 URL Stream / Livestream

Stream from online platforms or network cameras.

#### Supported Platforms:
- ✅ YouTube Live
- ✅ Twitch
- ✅ Facebook Live
- ✅ Direct HLS streams (.m3u8)
- ✅ RTSP cameras
- ✅ Direct MP4 URLs

#### Setup Requirements:
```bash
pip install streamlink
```

#### Platform-Specific Examples:

**YouTube Live:**
```
URL: https://www.youtube.com/watch?v=LIVE_VIDEO_ID
```

**Twitch:**
```
URL: https://www.twitch.tv/channel_name
```

**Facebook Live:**
```
URL: https://www.facebook.com/username/videos/VIDEO_ID
```

**Direct HLS Stream:**
```
URL: https://example.com/stream/playlist.m3u8
```

**RTSP Camera:**
```
URL: rtsp://username:password@camera_ip:554/stream
```

**IP Camera:**
```
URL: http://camera_ip/video
```

#### Using URL Stream:

**Web Interface:**
1. Select "Realtime Analysis" mode
2. Choose "URL Stream" as source
3. Paste the URL
4. Click "Start" (app will auto-resolve the stream)

**Command Line:**
```bash
# YouTube Live
python realtime_main.py --source "https://youtube.com/watch?v=VIDEO_ID"

# Twitch
python realtime_main.py --source "https://twitch.tv/channel"

# RTSP Camera
python realtime_main.py --source "rtsp://192.168.1.100:554/stream"

# Direct HLS
python realtime_main.py --source "https://example.com/stream.m3u8"
```

---

## 🎨 Visualization Options

Configure what to display in the output:

| Option | Description |
|--------|-------------|
| **Players** | Highlight all field players with bounding boxes |
| **Goalkeepers** | Highlight goalkeepers separately |
| **Ball** | Track and highlight the ball |
| **Referees** | Highlight match officials |
| **Possession Stats** | Display ball possession percentage |

**Web Interface:**
- Use checkboxes in sidebar under "Visualization Options"

**Command Line:**
```bash
# Track everything
python realtime_main.py --source 0 --tracks players ball referees stats

# Track only players and ball
python realtime_main.py --source 0 --tracks players ball

# Minimal tracking
python realtime_main.py --source 0 --tracks ball
```

---

## 🛠️ Troubleshooting

### Webcam Not Detected

**Problem:** No cameras appear in dropdown

**Solutions:**
1. Check camera permissions in Windows Settings
2. Close other apps using the camera (Zoom, Teams, etc.)
3. Try a different browser (Chrome recommended)
4. Restart the Streamlit app

### Phone Camera Connection Failed

**Problem:** Stream URL doesn't work

**Solutions:**
1. Verify phone and computer on **same WiFi network**
2. Check IP address in phone app (it may change)
3. Disable phone firewall/VPN temporarily
4. Try format: `http://IP:8080/video` (add `/video` suffix)
5. Test URL in browser first - should show video feed

### URL Stream Resolution Failed

**Problem:** "Could not resolve stream" error

**Solutions:**
1. Install streamlink: `pip install streamlink`
2. Verify URL is accessible in your browser
3. Check if stream is actually live (not offline)
4. Try using direct stream URL instead of webpage URL
5. For YouTube, use format: `https://youtube.com/watch?v=VIDEO_ID`

### Uploaded File Won't Process

**Problem:** Upload fails or processing errors

**Solutions:**
1. Check file format is supported (MP4, AVI, MOV, MKV, FLV)
2. Try converting to MP4 with tools like FFmpeg
3. Reduce file size if over 500MB
4. Check file isn't corrupted (play in VLC first)

### Processing is Very Slow

**Problem:** Analysis takes too long

**Solutions:**
1. Use smaller resolution videos (720p recommended)
2. Use fewer tracks (disable goalkeepers/referees)
3. For realtime: use faster computer or reduce FPS
4. For offline: be patient - high quality takes time
5. Consider GPU acceleration (CUDA) if available

### Model Loading Error

**Problem:** "Model not found" or similar errors

**Solutions:**
1. Verify `models/best.pt` exists in project directory
2. Check file permissions
3. Re-download model if corrupted
4. Update ultralytics: `pip install --upgrade ultralytics`

### Streamlit App Won't Start

**Problem:** `streamlit run app.py` fails

**Solutions:**
1. Ensure you're in correct directory
2. Check all dependencies: `pip install -r requirements.txt`
3. Try: `python -m streamlit run frontend/app.py`
4. Check port 8501 isn't already in use
5. Kill existing streamlit processes

### Poor Detection Quality

**Problem:** Players/ball not detected accurately

**Solutions:**
1. Ensure good video quality (720p+ recommended)
2. Avoid videos with multiple camera angles/cuts
3. Use videos with clear view of the field
4. Minimize camera movement/shaking
5. Ensure good lighting conditions

---

## 💡 Tips & Best Practices

### For Best Results:
- ✅ Use videos with **single camera angle** (no cuts)
- ✅ **720p or 1080p** resolution recommended
- ✅ **Well-lit** outdoor matches work best
- ✅ **Stable camera** (minimal shaking)
- ✅ **Clear view** of the pitch

### Performance Optimization:
- 🚀 Close other programs during processing
- 🚀 Use SSD storage for faster I/O
- 🚀 Enable GPU acceleration if available
- 🚀 Process shorter clips for faster results
- 🚀 Use realtime mode for quick testing

### Networking Tips (Phone/IP Cameras):
- 📡 Use **5GHz WiFi** for better performance
- 📡 Keep phone/camera close to router
- 📡 Reduce other network activity during streaming
- 📡 Use **static IP** for cameras if possible

---

## 📞 Support

For issues or questions:
1. Check this guide first
2. Review console logs for error details
3. Open an issue on GitHub
4. Include: error message, video source type, system info

---

**MatchVision** | Automated Football Analysis using Computer Vision 🎯⚽
