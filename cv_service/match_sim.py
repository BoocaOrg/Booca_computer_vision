"""
Simple "match simulation" renderer (stylized pitch) similar to broadcast graphics:
- Draw full pitch with markings
- Draw ball and a short trail (history)
- Highlight possessor (player with has_ball)

This is purely a UI visualization: it uses the same `tracks` dict shape as minimap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.bbox_utils import get_center_of_bbox, get_foot_position


def _foot_or_adjusted(pdata: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pa = pdata.get("position_adjusted")
    if pa is not None and len(pa) >= 2:
        return float(pa[0]), float(pa[1])
    bbox = pdata.get("bbox")
    if bbox:
        fx, fy = get_foot_position(bbox)
        return float(fx), float(fy)
    return None


def _collect_bounds(tracks: Dict[str, Any], ball_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    players = tracks.get("players") or {}
    for _pid, pdata in players.items():
        p = _foot_or_adjusted(pdata)
        if p:
            xs.append(p[0])
            ys.append(p[1])

    def _push_ball(b: Dict[str, Any]):
        if 1 in b and isinstance(b[1], dict) and "bbox" in b[1]:
            cx, cy = get_center_of_bbox(b[1]["bbox"])
            xs.append(float(cx))
            ys.append(float(cy))

    ball = tracks.get("ball") or {}
    if isinstance(ball, dict):
        _push_ball(ball)
    if ball_history:
        for h in ball_history:
            hb = (h or {}).get("ball") or {}
            if isinstance(hb, dict):
                _push_ball(hb)

    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max(max_x - min_x, 1.0)
    ry = max(max_y - min_y, 1.0)
    pad = 0.1
    return (min_x - rx * pad, max_x + rx * pad, min_y - ry * pad, max_y + ry * pad)


def _world_to_pixel(x: float, y: float, min_x: float, max_x: float, min_y: float, max_y: float, w: int, h: int) -> Tuple[int, int]:
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)
    px = int((x - min_x) / rx * (w - 1))
    py = int((y - min_y) / ry * (h - 1))
    return int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))


def _draw_pitch(img: np.ndarray, w: int, h: int) -> None:
    # Slightly nicer pitch: gradient stripes
    base1 = np.array([32, 92, 44], dtype=np.uint8)   # BGR
    base2 = np.array([36, 105, 50], dtype=np.uint8)
    stripe_h = max(8, h // 10)
    for y in range(0, h, stripe_h):
        img[y : y + stripe_h] = base1 if (y // stripe_h) % 2 == 0 else base2

    line = (230, 245, 255)
    t = 1
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), line, t)
    cv2.line(img, (w // 2, 2), (w // 2, h - 3), line, t)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 8, line, t)

    # Penalty boxes (approx)
    box_w = int(w * 0.16)
    box_h = int(h * 0.55)
    y1 = (h - box_h) // 2
    y2 = y1 + box_h
    cv2.rectangle(img, (2, y1), (2 + box_w, y2), line, t)
    cv2.rectangle(img, (w - 3 - box_w, y1), (w - 3, y2), line, t)


def render_match_sim_bgr(
    tracks: Dict[str, Any],
    *,
    ball_history: Optional[List[Dict[str, Any]]] = None,
    width: int = 420,
    height: int = 260,
) -> np.ndarray:
    w, h = int(width), int(height)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _draw_pitch(img, w, h)

    min_x, max_x, min_y, max_y = _collect_bounds(tracks, ball_history=ball_history)

    players = tracks.get("players") or {}
    possessor_id: Optional[int] = None
    for pid, pdata in players.items():
        if pdata.get("has_ball"):
            possessor_id = int(pid)
            break

    # Draw ball trail
    if ball_history:
        pts = []
        for hitem in ball_history:
            b = (hitem or {}).get("ball") or {}
            if 1 in b and isinstance(b[1], dict) and "bbox" in b[1]:
                bx, by = get_center_of_bbox(b[1]["bbox"])
                pts.append(_world_to_pixel(float(bx), float(by), min_x, max_x, min_y, max_y, w, h))
        for i in range(1, len(pts)):
            alpha = i / max(1, (len(pts) - 1))
            col = (0, int(120 + 100 * alpha), int(200 + 55 * alpha))  # cyan-ish trail
            cv2.line(img, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

    # Draw current ball
    ball = tracks.get("ball") or {}
    if 1 in ball and isinstance(ball[1], dict) and "bbox" in ball[1]:
        bx, by = get_center_of_bbox(ball[1]["bbox"])
        px, py = _world_to_pixel(float(bx), float(by), min_x, max_x, min_y, max_y, w, h)
        cv2.circle(img, (px, py), 7, (0, 255, 255), -1)  # yellow
        cv2.circle(img, (px, py), 10, (0, 0, 0), 1)

    # Highlight possessor
    if possessor_id is not None and possessor_id in players:
        pdata = players[possessor_id]
        p = _foot_or_adjusted(pdata)
        if p:
            px, py = _world_to_pixel(p[0], p[1], min_x, max_x, min_y, max_y, w, h)
            team = int(pdata.get("team") or 0)
            if team == 1:
                col = (0, 0, 255)
            elif team == 2:
                col = (255, 0, 0)
            else:
                col = (255, 255, 255)
            cv2.circle(img, (px, py), 12, col, 2)
            cv2.circle(img, (px, py), 3, col, -1)

    # Small title
    cv2.putText(img, "Simulation", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_match_sim_rgb(
    tracks: Dict[str, Any],
    *,
    ball_history: Optional[List[Dict[str, Any]]] = None,
    width: int = 420,
    height: int = 260,
) -> np.ndarray:
    bgr = render_match_sim_bgr(tracks, ball_history=ball_history, width=width, height=height)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

