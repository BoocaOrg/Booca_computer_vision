# Broadcast Guide

## Quick Start

### 1. Start Broadcast Server
```sh
# MJPEG only (browser stream):
python scripts/broadcast_server.py

# With YouTube Live:
python scripts/broadcast_server.py --rtmp rtmp://a.rtmp.youtube.com/live2/YOUR_KEY

# With Twitch:
python scripts/broadcast_server.py --rtmp rtmp://live.twitch.tv/app/YOUR_STREAM_KEY
```

### 2. Access Stream
- Open **http://localhost:8502** in your browser
- MJPEG stream: **http://localhost:8502/mjpeg** (for embedding in OBS or other tools)

### 3. Use with MatchVision
- Enable "Broadcast" checkbox in the Streamlit sidebar
- Processed frames will be streamed to the broadcast server
- The broadcast server forwards to:
  - MJPEG: Any browser or OBS
  - RTMP: YouTube Live, Twitch, or any RTMP-compatible platform

## OBS Integration
1. Add Media Source in OBS
2. Set URL to: `http://localhost:8502/mjpeg`
3. Input format: MJPEG
4. Resolution: Match your stream

## YouTube Live Setup
1. Go to YouTube Studio > Create > Go Live
2. Copy the RTMP URL and Stream Key
3. Run: `python scripts/broadcast_server.py --rtmp "rtmp://a.rtmp.youtube.com/live2/YOUR_KEY"`
4. Start analysis in MatchVision
