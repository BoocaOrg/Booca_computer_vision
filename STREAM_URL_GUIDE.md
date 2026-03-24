# Hướng Dẫn Phân Tích Stream Từ Các Trang Web

## 🎯 Tổng Quan

App hỗ trợ phân tích video từ nhiều nguồn khác nhau, bao gồm các trang web live streaming. Có **3 phương pháp** để lấy stream URL:

---

## Phương Pháp 1: Tự Động (Khuyến Nghị) ✨

App sẽ tự động thử extract URL stream từ trang web.

### Cách Sử Dụng:

1. Mở app tại `http://localhost:8502`
2. Chọn **"Realtime Analysis"**
3. Chọn **"URL Stream"**
4. Paste URL trang web trực tiếp
5. Click **"Start"**

App sẽ thử các công cụ theo thứ tự:
- **Streamlink** (hỗ trợ YouTube, Twitch, Facebook Live, v.v.)
- **yt-dlp** (fallback nếu streamlink thất bại)
- **Direct OpenCV** (nếu cả hai thất bại)

### Ví Dụ:

```
✅ YouTube Live:
https://youtube.com/watch?v=LIVE_VIDEO_ID

✅ Twitch:
https://twitch.tv/channel_name

✅ Facebook Live:
https://www.facebook.com/username/videos/VIDEO_ID
```

---

## Phương Pháp 2: Lấy URL Thủ Công 🛠️

Nếu phương pháp tự động không hoạt động, bạn có thể lấy URL stream trực tiếp.

### Bước 1: Mở Developer Tools

- **Chrome/Edge**: Nhấn `F12` hoặc `Ctrl+Shift+I`
- **Firefox**: Nhấn `F12`

### Bước 2: Vào Tab Network

1. Click vào tab **"Network"**
2. **Refresh** trang web (F5)
3. Chọn filter **"Media"** hoặc **"All"**

### Bước 3: Tìm Stream URL

Tìm các request với extension:
- `.m3u8` (HLS playlist) - **Quan trọng nhất**
- `.ts` (video segments)
- `.mp4` (direct video)
- `.flv` (Flash video)

### Bước 4: Copy URL

1. Click chuột phải vào request
2. **Copy** → **Copy URL** hoặc **Copy as cURL**
3. Paste vào app

### Ví Dụ URL Tìm Được:

```
✅ HLS Stream:
https://example.com/live/stream/chunklist.m3u8

✅ Direct MP4:
https://cdn.example.com/video/stream.mp4

✅ RTSP:
rtsp://192.168.1.100:554/live/stream
```

---

## Phương Pháp 3: Sử Dụng yt-dlp CLI 💻

Nếu URL quá phức tạp, dùng yt-dlp để extract trước.

### Cài Đặt:

```bash
pip install yt-dlp
```

### Lấy URL:

```bash
# Lấy URL trực tiếp
yt-dlp -g URL_TRANG_WEB

# Lấy URL chất lượng tốt nhất
yt-dlp -f best -g URL_TRANG_WEB

# Lấy URL với format MP4
yt-dlp -f "best[ext=mp4]" -g URL_TRANG_WEB
```

### Ví Dụ:

```bash
# YouTube Live
yt-dlp -g "https://youtube.com/watch?v=VIDEO_ID"

# Kết quả:
# https://manifest.googlevideo.com/api/manifest/hls_...m3u8

# Copy URL này vào app
```

---

## 🔧 Xử Lý Lỗi Thường Gặp

### Lỗi: "Cannot open video source"

**Nguyên nhân:**
- URL không hợp lệ hoặc stream đã offline
- Stream cần authentication
- Stream có geo-restriction
- OpenCV không hỗ trợ codec

**Giải pháp:**

#### 1. Kiểm tra URL trong VLC:
```
1. Mở VLC Media Player
2. Media → Open Network Stream
3. Paste URL
4. Play
```
Nếu VLC không play được → URL có vấn đề

#### 2. Cài đặt FFmpeg:
```bash
# Windows (Chocolatey)
choco install ffmpeg

# hoặc download từ: https://ffmpeg.org/download.html
```

#### 3. Thử yt-dlp:
```bash
pip install yt-dlp
# Sau đó refresh app, yt-dlp sẽ tự động được dùng
```

#### 4. Lấy URL trực tiếp từ Developer Tools (xem Phương pháp 2)

---

### Lỗi: "Stream ended" ngay sau khi bắt đầu

**Nguyên nhân:**
- Stream chỉ phát một lần rồi dừng
- URL là video on-demand (VOD) thay vì live stream
- Timeout quá ngắn

**Giải pháp:**
- Kiểm tra stream có đang live không
- Nếu là VOD, dùng chế độ "Offline Processing" thay vì "Realtime"

---

### Lỗi: Streamlink/yt-dlp không cài được

**Giải pháp:**

```bash
# Upgrade pip trước
python -m pip install --upgrade pip

# Cài streamlink
pip install streamlink

# Cài yt-dlp
pip install yt-dlp

# Kiểm tra
streamlink --version
yt-dlp --version
```

---

## 📋 Các Trang Web Được Test

### ✅ Hoạt động tốt:
- YouTube Live (streamlink + yt-dlp)
- Twitch (streamlink)
- Direct .m3u8 URLs
- RTSP cameras
- IP Webcam apps

### ⚠️ Cần extract URL thủ công:
- Một số trang streaming Việt Nam
- Các trang có DRM protection
- Streams cần authentication

### ❌ Không hỗ trợ:
- Streams với DRM (Netflix, Disney+, etc.)
- Streams cần login trước
- Flash-only streams

---

## 💡 Mẹo Tối Ưu

### 1. Kiểm tra Stream trước:
```bash
# Test với yt-dlp
yt-dlp --list-formats URL

# Test với streamlink
streamlink URL
```

### 2. Chọn chất lượng phù hợp:
- **720p hoặc 480p**: Tốt nhất cho phân tích realtime
- **1080p**: Chỉ nếu máy đủ mạnh
- **Tránh 4K**: Quá nặng cho realtime processing

### 3. Kết nối ổn định:
- Dùng dây LAN thay vì WiFi nếu được
- Tắt các ứng dụng tốn băng thông khác
- Kiểm tra ping đến server stream

### 4. Xử lý stream không ổn định:
```python
# Nếu stream hay bị gián đoạn, có thể sửa code:
# Trong app.py, thêm retry logic:

max_retries = 3
retry_count = 0

while retry_count < max_retries:
    cap = cv2.VideoCapture(source_path)
    if cap.isOpened():
        break
    retry_count += 1
    time.sleep(2)
```

---

## 📞 Ví Dụ Cụ Thể

### Ví Dụ 1: YouTube Live Stream

```bash
# URL gốc:
https://youtube.com/watch?v=ABC123

# App sẽ tự động:
1. Thử streamlink → thành công → lấy .m3u8 URL
2. OpenCV mở stream
3. Bắt đầu phân tích
```

### Ví Dụ 2: Trang Web Không Rõ

```bash
# URL gốc:
https://unknown-site.com/live

# Bước 1: Thử trực tiếp trong app
- Paste URL → Click Start
- Nếu lỗi → Tiếp tục bước 2

# Bước 2: Dùng yt-dlp
yt-dlp -g "https://unknown-site.com/live"
# Nếu thành công → Copy URL kết quả → Paste vào app

# Bước 3: Developer Tools (nếu bước 2 thất bại)
1. F12 → Network → Media
2. Refresh trang
3. Tìm .m3u8 hoặc .ts
4. Copy URL → Paste vào app
```

### Ví Dụ 3: IP Camera

```bash
# RTSP camera:
rtsp://admin:password@192.168.1.100:554/stream

# HTTP stream:
http://192.168.1.100:8080/video

# Paste trực tiếp vào app, không cần resolve
```

---

## 📊 Bảng Tóm Tắt

| Loại Stream | Công Cụ Tốt Nhất | Độ Khó | Ghi Chú |
|-------------|-------------------|--------|---------|
| YouTube Live | Streamlink/yt-dlp | ⭐ Dễ | Tự động 100% |
| Twitch | Streamlink | ⭐ Dễ | Tự động 100% |
| Facebook Live | yt-dlp | ⭐⭐ Trung bình | Đôi khi cần thủ công |
| RTSP Camera | Direct | ⭐ Dễ | Paste trực tiếp |
| IP Webcam | Direct | ⭐ Dễ | Paste trực tiếp |
| Trang web VN | Developer Tools | ⭐⭐⭐ Khó | Cần extract thủ công |
| HLS (.m3u8) | Direct | ⭐ Dễ | Paste trực tiếp |

---

## 🚀 Quick Reference

```bash
# Cài đặt đầy đủ
pip install streamlink yt-dlp ffmpeg-python

# Test stream
streamlink URL                          # Xem có support không
yt-dlp -g URL                          # Lấy direct URL
ffprobe URL                            # Kiểm tra stream info

# Trong app
1. Realtime Analysis
2. URL Stream
3. Paste URL
4. Start

# Nếu lỗi
1. Check VLC
2. Try yt-dlp extract
3. Use Developer Tools
4. Check troubleshooting trong app
```

---

**Cập nhật:** 2026-01-13 | App version: 2.0 với yt-dlp support

Nếu vẫn gặp vấn đề, mở error details trong app để xem log chi tiết!
