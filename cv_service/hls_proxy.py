import socket
import threading
import time
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote
from urllib.parse import urlparse
import collections
import os


def _now() -> float:
    return time.time()


def _is_m3u8(url: str) -> bool:
    return url.endswith(".m3u8") or ".m3u8?" in url


def _is_ts(url: str) -> bool:
    # Common HLS segment extensions
    return url.endswith(".ts") or ".ts?" in url or url.endswith(".m4s") or ".m4s?" in url

class ProxyHTTPRequestHandler(BaseHTTPRequestHandler):
    # Aggregate request counts per host to diagnose outbound "DDoS-like" patterns.
    _counts = collections.Counter()
    _lock = threading.Lock()
    _last_report = 0.0

    # Very small in-memory cache to reduce repeated manifest fetches.
    # This helps when players/reconnects trigger bursts of identical GETs.
    _cache = {}  # url -> (expires_at, status_code, headers_dict, body_bytes)
    _cache_bytes = 0
    _cache_max_bytes = int(os.getenv("HLS_PROXY_CACHE_MAX_BYTES", str(2 * 1024 * 1024)))  # 2MB default
    _cache_ttl_m3u8 = float(os.getenv("HLS_PROXY_CACHE_TTL_M3U8_SEC", "1.0"))  # very short
    _cache_ttl_ts = float(os.getenv("HLS_PROXY_CACHE_TTL_TS_SEC", "10.0"))     # short, segments are immutable

    def log_message(self, format, *args):
        # Silence default per-request logging (keeps logs readable under load)
        return

    @classmethod
    def _cache_get(cls, url: str):
        item = cls._cache.get(url)
        if not item:
            return None
        expires_at, status_code, headers, body = item
        if _now() >= expires_at:
            cls._cache_pop(url)
            return None
        return status_code, headers, body

    @classmethod
    def _cache_pop(cls, url: str):
        item = cls._cache.pop(url, None)
        if item:
            try:
                cls._cache_bytes -= len(item[3])
            except Exception:
                pass

    @classmethod
    def _cache_put(cls, url: str, status_code: int, headers: dict, body: bytes):
        if cls._cache_max_bytes <= 0:
            return
        if body is None:
            return
        size = len(body)
        # Refuse caching huge objects
        if size <= 0 or size > cls._cache_max_bytes:
            return
        # Simple eviction: drop everything if we'd exceed max (keeps it robust)
        if cls._cache_bytes + size > cls._cache_max_bytes:
            cls._cache.clear()
            cls._cache_bytes = 0
        ttl = cls._cache_ttl_m3u8 if _is_m3u8(url) else (cls._cache_ttl_ts if _is_ts(url) else 0.0)
        if ttl <= 0:
            return
        cls._cache[url] = (_now() + ttl, status_code, headers, body)
        cls._cache_bytes += size

    def do_GET(self):
        target_url = unquote(self.path[1:]) # remove leading slash
        try:
            host = urlparse(target_url).netloc or "unknown"
        except Exception:
            host = "unknown"

        # Count + periodic report (every 30s)
        now = time.time()
        with ProxyHTTPRequestHandler._lock:
            ProxyHTTPRequestHandler._counts[host] += 1
            if now - ProxyHTTPRequestHandler._last_report >= 30:
                ProxyHTTPRequestHandler._last_report = now
                top = ProxyHTTPRequestHandler._counts.most_common(5)
                if top:
                    print("[Proxy] Outbound request counts (last ~30s window, cumulative until restart):")
                    for h, c in top:
                        print(f"  - {h}: {c}")

        # Serve from cache (manifest/segments) when possible
        with ProxyHTTPRequestHandler._lock:
            cached = ProxyHTTPRequestHandler._cache_get(target_url)
        if cached is not None:
            status_code, cached_headers, body = cached
            self.send_response(status_code)
            for k, v in (cached_headers or {}).items():
                if k.lower() not in ['transfer-encoding', 'content-encoding']:
                    self.send_header(k, v)
            self.end_headers()
            if body:
                self.wfile.write(body)
            return

        headers = {
            "Referer": "https://booca.online/",
            "User-Agent": "Mozilla/5.0",
        }
        try:
            # Don't stream for cacheable resources; streaming defeats caching and increases overhead.
            cacheable = _is_m3u8(target_url) or _is_ts(target_url)
            resp = requests.get(target_url, stream=not cacheable, headers=headers, timeout=10)
            self.send_response(resp.status_code)
            resp_headers = {}
            for k, v in resp.headers.items():
                if k.lower() not in ['transfer-encoding', 'content-encoding']:
                    resp_headers[k] = v
                    self.send_header(k, v)
            self.end_headers()
            if cacheable:
                body = resp.content
                if body:
                    self.wfile.write(body)
                with ProxyHTTPRequestHandler._lock:
                    ProxyHTTPRequestHandler._cache_put(target_url, resp.status_code, resp_headers, body)
            else:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        self.wfile.write(chunk)
        except Exception as e:
            print(f"[Proxy] Error fetching {target_url}: {e}")
            self.send_error(500)

class SimpleProxy:
    def __init__(self):
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
            return target_url
        return f"http://127.0.0.1:{self.port}/{target_url}"

_global_proxy = None

def get_hls_proxy():
    global _global_proxy
    if _global_proxy is None:
        _global_proxy = SimpleProxy()
    return _global_proxy
