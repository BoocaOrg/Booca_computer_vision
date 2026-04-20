import cv2
import threading
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        target_url = self.path[1:]
        print(f"Proxying GET: {target_url}")
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
        except Exception as e:
            print(f"Proxy error: {e}")
            self.send_error(500)

class SimpleProxy:
    def __init__(self, port=8899):
        self.port = port
        self.server = HTTPServer(('127.0.0.1', port), ProxyHTTPRequestHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(1)
        
    def get_proxy_url(self, target_url):
        return f"http://127.0.0.1:{self.port}/{target_url}"

proxy = SimpleProxy()
url = "https://vz-ce21b40a-bf7.b-cdn.net/12b29631-e23a-4f6d-8697-52a7bf904c86/playlist.m3u8"
proxied_url = proxy.get_proxy_url(url)
print(f"Opening via proxy: {proxied_url}")

cap = cv2.VideoCapture(proxied_url)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"SUCCESS: Read frame {frame.shape}")
    else:
        print("FAILED: Opened but could not read frame")
else:
    print("FAILED: Could not open stream")
