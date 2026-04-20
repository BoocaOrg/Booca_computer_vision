"""
Pitch Calibration Module (#3)

Auto-detect pitch white lines using Hough Transform to compute
an accurate pixel_to_meter_scale for speed estimation.

Standard football pitch: 105m x 68m
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class PitchCalibrator:
    """Auto-calibrate pixel-to-meter scale by detecting pitch lines."""

    PITCH_LENGTH_M = 105.0
    PITCH_WIDTH_M = 68.0
    DEFAULT_SCALE = 0.05  # fallback if calibration fails

    def __init__(self):
        self._last_scale: float = self.DEFAULT_SCALE
        self._calibrated: bool = False
        self._frame_count_at_calibration: int = 0

    @property
    def scale(self) -> float:
        return self._last_scale

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate_from_frame(self, frame: np.ndarray) -> float:
        """
        Attempt to detect pitch lines and compute pixel_to_meter_scale.

        Strategy:
        1. Convert to grayscale, threshold for white lines
        2. Detect line segments using HoughLinesP
        3. Find the two longest roughly-horizontal parallel lines (touchlines)
        4. Their pixel distance ≈ pitch width (68m) → compute scale

        Returns the computed scale, or DEFAULT_SCALE if detection fails.
        """
        try:
            h, w = frame.shape[:2]

            # 1. Preprocess: focus on white pixels (pitch lines)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE for better contrast on varying lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Threshold for white-ish pixels (pitch lines are white)
            _, binary = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Edge detection
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)

            # 2. Detect line segments
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=w * 0.15,  # at least 15% of frame width
                maxLineGap=30,
            )

            if lines is None or len(lines) < 2:
                print("[Calibrator] Not enough lines detected, using default scale")
                return self.DEFAULT_SCALE

            # 3. Classify lines by angle: horizontal vs vertical
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

                if angle < 25 or angle > 155:  # roughly horizontal
                    horizontal_lines.append((x1, y1, x2, y2, length, (y1 + y2) / 2))
                elif 65 < angle < 115:  # roughly vertical
                    vertical_lines.append((x1, y1, x2, y2, length, (x1 + x2) / 2))

            scale = self._try_horizontal_calibration(horizontal_lines, h, w)
            if scale is None:
                scale = self._try_vertical_calibration(vertical_lines, h, w)

            if scale is not None:
                # Sanity check: scale should be reasonable (0.01 ~ 0.2)
                if 0.01 <= scale <= 0.2:
                    self._last_scale = scale
                    self._calibrated = True
                    print(f"[Calibrator] ✅ Calibrated: px_to_m = {scale:.5f}")
                    return scale
                else:
                    print(f"[Calibrator] Scale {scale:.5f} out of range, using default")

            print("[Calibrator] Could not determine scale, using default")
            return self.DEFAULT_SCALE

        except Exception as e:
            print(f"[Calibrator] Error during calibration: {e}")
            return self.DEFAULT_SCALE

    def _try_horizontal_calibration(
        self, lines: list, frame_h: int, frame_w: int
    ) -> Optional[float]:
        """Find two horizontal lines (touchlines) and compute scale from their distance."""
        if len(lines) < 2:
            return None

        # Sort by length (longest first)
        lines_sorted = sorted(lines, key=lambda l: l[4], reverse=True)

        # Take top 10 longest lines
        candidates = lines_sorted[:10]

        # Group by Y position (cluster lines near same Y)
        groups = self._cluster_lines_by_position(candidates, axis='y', threshold=frame_h * 0.05)

        if len(groups) < 2:
            return None

        # Find the two groups with the largest Y separation (touchlines)
        best_pair = None
        max_dist = 0

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                y1 = np.mean([l[5] for l in groups[i]])
                y2 = np.mean([l[5] for l in groups[j]])
                dist = abs(y2 - y1)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (y1, y2)

        if best_pair is None or max_dist < frame_h * 0.2:
            return None

        # The pixel distance between touchlines ≈ 68 meters (pitch width)
        pixel_distance = max_dist
        scale = self.PITCH_WIDTH_M / pixel_distance
        return scale

    def _try_vertical_calibration(
        self, lines: list, frame_h: int, frame_w: int
    ) -> Optional[float]:
        """Fallback: find two vertical lines (goal lines or sidelines) and compute scale."""
        if len(lines) < 2:
            return None

        lines_sorted = sorted(lines, key=lambda l: l[4], reverse=True)
        candidates = lines_sorted[:10]

        groups = self._cluster_lines_by_position(candidates, axis='x', threshold=frame_w * 0.05)

        if len(groups) < 2:
            return None

        best_pair = None
        max_dist = 0

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                x1 = np.mean([l[5] for l in groups[i]])
                x2 = np.mean([l[5] for l in groups[j]])
                dist = abs(x2 - x1)
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (x1, x2)

        if best_pair is None or max_dist < frame_w * 0.2:
            return None

        # Vertical lines at max distance ≈ pitch length (105m)
        pixel_distance = max_dist
        scale = self.PITCH_LENGTH_M / pixel_distance
        return scale

    def _cluster_lines_by_position(
        self, lines: list, axis: str, threshold: float
    ) -> list:
        """Group lines that are close together on the specified axis."""
        if not lines:
            return []

        pos_idx = 5  # position is at index 5 (avg y or avg x)
        sorted_lines = sorted(lines, key=lambda l: l[pos_idx])

        groups = [[sorted_lines[0]]]
        for line in sorted_lines[1:]:
            if abs(line[pos_idx] - groups[-1][-1][pos_idx]) < threshold:
                groups[-1].append(line)
            else:
                groups.append([line])

        return groups

    def should_recalibrate(self, current_frame: int, interval: int = 3000) -> bool:
        """Check if enough frames have passed since last calibration."""
        if not self._calibrated:
            return True
        return (current_frame - self._frame_count_at_calibration) >= interval

    def recalibrate_if_needed(
        self, frame: np.ndarray, current_frame: int, interval: int = 3000
    ) -> float:
        """Recalibrate periodically. Returns current scale."""
        if self.should_recalibrate(current_frame, interval):
            self._frame_count_at_calibration = current_frame
            return self.calibrate_from_frame(frame)
        return self._last_scale
