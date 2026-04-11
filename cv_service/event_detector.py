"""
Event Detector — detects goals, cards, and other events from CV data.
Phase 1: Basic goal detection via ball position in goal zones.
"""
import numpy as np
from typing import Optional, Dict, List


class EventDetector:
    def __init__(self, frame_width: int, frame_height: int):
        self.fw = frame_width
        self.fh = frame_height

        # Goal zones: ~8% width on each side, ~40% height centered vertically
        self.goal_zone_left = (0, int(self.fh * 0.3), int(self.fw * 0.08), int(self.fh * 0.7))
        self.goal_zone_right = (int(self.fw * 0.92), int(self.fh * 0.3), self.fw, int(self.fh * 0.7))

        # Cooldown to avoid duplicate detections (~5s at 30fps)
        self.ball_in_goal_cooldown = 0
        self.COOLDOWN_FRAMES = 150

    def check_events(self, tracks: Dict, frame_count: int) -> Optional[Dict]:
        """
        Check all event types for the current frame.
        Returns event dict if detected, None otherwise.
        """
        # Decrement cooldown
        if self.ball_in_goal_cooldown > 0:
            self.ball_in_goal_cooldown -= 1

        # Check goal
        ball_dict = tracks.get("ball", {})
        if ball_dict and 1 in ball_dict:
            ball_bbox = ball_dict[1].get("bbox")
            if ball_bbox:
                goal = self._check_goal(ball_bbox)
                if goal:
                    return goal

        return None

    def _check_goal(self, ball_bbox: List[float]) -> Optional[Dict]:
        """Check if ball is in a goal zone."""
        if self.ball_in_goal_cooldown > 0:
            return None

        bx1, by1, bx2, by2 = ball_bbox
        ball_cx = (bx1 + bx2) / 2
        ball_cy = (by1 + by2) / 2

        in_left = self._in_zone(ball_cx, ball_cy, self.goal_zone_left)
        in_right = self._in_zone(ball_cx, ball_cy, self.goal_zone_right)

        if in_left or in_right:
            self.ball_in_goal_cooldown = self.COOLDOWN_FRAMES
            # Ball in left goal → team 2 scored; ball in right goal → team 1 scored
            scoring_team = 2 if in_left else 1
            return {
                "event": "goal",
                "scoring_team": scoring_team,
                "side": "left" if in_left else "right",
            }

        return None

    def _in_zone(self, cx: float, cy: float, zone: tuple) -> bool:
        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2
