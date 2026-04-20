"""
Event Detector — detects goals, corner kicks, and offside from CV data.
Supports: goal, corner_kick, offside detection.
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

        # Corner zones: near the 4 corners of the pitch
        self.corner_zones = [
            (0, 0, int(self.fw * 0.06), int(self.fh * 0.15)),                        # top-left
            (int(self.fw * 0.94), 0, self.fw, int(self.fh * 0.15)),                   # top-right
            (0, int(self.fh * 0.85), int(self.fw * 0.06), self.fh),                   # bottom-left
            (int(self.fw * 0.94), int(self.fh * 0.85), self.fw, self.fh),             # bottom-right
        ]

        # Cooldowns to avoid duplicate detections
        self.ball_in_goal_cooldown = 0
        self.COOLDOWN_FRAMES = 150          # ~5s at 30fps

        self.corner_cooldown = 0
        self.CORNER_COOLDOWN_FRAMES = 120   # ~4s at 30fps

        self.offside_cooldown = 0
        self.OFFSIDE_COOLDOWN_FRAMES = 90   # ~3s at 30fps

    def check_events(self, tracks: Dict, frame_count: int) -> Optional[Dict]:
        """
        Check all event types for the current frame.
        Priority: goal > corner > offside.
        Returns event dict if detected, None otherwise.
        """
        # Decrement cooldowns
        if self.ball_in_goal_cooldown > 0:
            self.ball_in_goal_cooldown -= 1
        if self.corner_cooldown > 0:
            self.corner_cooldown -= 1
        if self.offside_cooldown > 0:
            self.offside_cooldown -= 1

        # Extract ball position
        ball_dict = tracks.get("ball", {})
        ball_bbox = None
        if ball_dict and 1 in ball_dict:
            ball_bbox = ball_dict[1].get("bbox")

        # Check goal (highest priority)
        if ball_bbox:
            goal = self._check_goal(ball_bbox)
            if goal:
                return goal

        # Check corner kick
        if ball_bbox:
            corner = self._check_corner(ball_bbox)
            if corner:
                return corner

        # Check offside (requires player tracking data)
        offside = self._check_offside(tracks)
        if offside:
            return offside

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

    def _check_corner(self, ball_bbox: List[float]) -> Optional[Dict]:
        """Check if ball enters a corner zone (potential corner kick)."""
        if self.corner_cooldown > 0:
            return None

        bx1, by1, bx2, by2 = ball_bbox
        ball_cx = (bx1 + bx2) / 2
        ball_cy = (by1 + by2) / 2

        corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
        for i, zone in enumerate(self.corner_zones):
            if self._in_zone(ball_cx, ball_cy, zone):
                self.corner_cooldown = self.CORNER_COOLDOWN_FRAMES
                # Ball near left goal corners → team 1 gets corner (attacking right)
                # Ball near right goal corners → team 2 gets corner (attacking left)
                is_left_side = i in (0, 2)
                awarding_team = 1 if is_left_side else 2
                return {
                    "event": "corner_kick",
                    "scoring_team": awarding_team,
                    "side": "left" if is_left_side else "right",
                    "corner_position": corner_names[i],
                }

        return None

    def _check_offside(self, tracks: Dict) -> Optional[Dict]:
        """
        Heuristic offside detection:
        A player is potentially offside if they are beyond the second-to-last
        defender of the opposing team AND beyond the ball position,
        in the attacking half of the pitch.
        """
        if self.offside_cooldown > 0:
            return None

        players = tracks.get("players", {})
        ball_dict = tracks.get("ball", {})

        if not ball_dict or 1 not in ball_dict:
            return None

        ball_bbox = ball_dict[1].get("bbox")
        if not ball_bbox:
            return None

        ball_cx = (ball_bbox[0] + ball_bbox[2]) / 2

        # Gather X positions by team
        team1_xs = []
        team2_xs = []
        for pid, pdata in players.items():
            if "position_adjusted" not in pdata:
                continue
            x = pdata["position_adjusted"][0]
            team = pdata.get("team")
            if team == 1:
                team1_xs.append(x)
            elif team == 2:
                team2_xs.append(x)

        # Need at least 2 players per team to make an offside judgment
        if len(team1_xs) < 2 or len(team2_xs) < 2:
            return None

        # Team 1 attacks right: check if any team1 player is beyond
        # the second-to-last team2 defender (sorted rightmost first)
        team2_sorted_desc = sorted(team2_xs, reverse=True)
        second_last_t2 = team2_sorted_desc[1]

        for x in team1_xs:
            if x > second_last_t2 and x > ball_cx and x > self.fw * 0.55:
                self.offside_cooldown = self.OFFSIDE_COOLDOWN_FRAMES
                return {
                    "event": "offside",
                    "scoring_team": 1,
                    "side": "right",
                }

        # Team 2 attacks left: check if any team2 player is beyond
        # the second-to-last team1 defender (sorted leftmost first)
        team1_sorted_asc = sorted(team1_xs)
        second_last_t1 = team1_sorted_asc[1]

        for x in team2_xs:
            if x < second_last_t1 and x < ball_cx and x < self.fw * 0.45:
                self.offside_cooldown = self.OFFSIDE_COOLDOWN_FRAMES
                return {
                    "event": "offside",
                    "scoring_team": 2,
                    "side": "left",
                }

        return None

    def _in_zone(self, cx: float, cy: float, zone: tuple) -> bool:
        x1, y1, x2, y2 = zone
        return x1 <= cx <= x2 and y1 <= cy <= y2
