"""
Top-down minimap: pitch schematic + ball + player holding the ball.

Uses the same coordinate space as `position_adjusted` (after camera compensation),
falling back to foot position from bbox when needed.
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


def _collect_bounds(tracks: Dict[str, Any]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    players = tracks.get("players") or {}
    for _pid, pdata in players.items():
        p = _foot_or_adjusted(pdata)
        if p:
            xs.append(p[0])
            ys.append(p[1])
    ball = tracks.get("ball") or {}
    if 1 in ball and "bbox" in ball[1]:
        cx, cy = get_center_of_bbox(ball[1]["bbox"])
        xs.append(float(cx))
        ys.append(float(cy))
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rx = max(max_x - min_x, 1.0)
    ry = max(max_y - min_y, 1.0)
    pad = 0.05
    return (
        min_x - rx * pad,
        max_x + rx * pad,
        min_y - ry * pad,
        max_y + ry * pad,
    )


def _world_to_pixel(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    w: int,
    h: int,
) -> Tuple[int, int]:
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)
    px = int((x - min_x) / rx * (w - 1))
    py = int((y - min_y) / ry * (h - 1))
    return int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))


def _draw_pitch_lines(img: np.ndarray, w: int, h: int) -> None:
    """Simple white lines on green field (BGR)."""
    col = (220, 240, 255)
    t = 1
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, t)
    cv2.line(img, (w // 2, 0), (w // 2, h - 1), col, t)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 8, col, t)


def render_minimap_bgr(tracks: Dict[str, Any], width: int = 320, height: int = 180) -> np.ndarray:
    """
    Return BGR uint8 image (height x width x 3).
    """
    w, h = int(width), int(height)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Pitch green (BGR)
    img[:] = (34, 90, 46)

    min_x, max_x, min_y, max_y = _collect_bounds(tracks)
    _draw_pitch_lines(img, w, h)

    players = tracks.get("players") or {}
    possessor_id: Optional[int] = None
    for pid, pdata in players.items():
        if pdata.get("has_ball"):
            possessor_id = int(pid)
            break

    ball = tracks.get("ball") or {}
    if 1 in ball and "bbox" in ball[1]:
        bx, by = get_center_of_bbox(ball[1]["bbox"])
        px, py = _world_to_pixel(float(bx), float(by), min_x, max_x, min_y, max_y, w, h)
        cv2.circle(img, (px, py), 6, (0, 255, 255), -1)  # yellow (BGR)
        cv2.circle(img, (px, py), 8, (0, 0, 0), 1)

    if possessor_id is not None and possessor_id in players:
        pdata = players[possessor_id]
        p = _foot_or_adjusted(pdata)
        if p:
            px, py = _world_to_pixel(p[0], p[1], min_x, max_x, min_y, max_y, w, h)
            team = int(pdata.get("team") or 0)
            if team == 1:
                color = (0, 0, 255)  # red (BGR)
            elif team == 2:
                color = (255, 0, 0)  # blue (BGR)
            else:
                color = (200, 200, 200)
            cv2.circle(img, (px, py), 12, color, 2)
            cv2.putText(
                img,
                f"#{possessor_id}",
                (px + 10, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        img,
        "Minimap",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return img


def render_minimap_rgb(tracks: Dict[str, Any], width: int = 320, height: int = 180) -> np.ndarray:
    bgr = render_minimap_bgr(tracks, width=width, height=height)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
