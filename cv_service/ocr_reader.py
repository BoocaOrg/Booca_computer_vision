"""
OCR Reader — Jersey number detection using EasyOCR.

Designed for low-resolution cameras (typical amateur football fields).
Only attempts OCR when player bounding box is large enough (>150px height).
Uses a voting system to confirm jersey numbers across multiple frames.
"""
import collections
import numpy as np

# Lazy import EasyOCR to avoid loading model if not needed
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print("[OCR] EasyOCR model loaded successfully")
        except ImportError:
            print("[OCR] WARNING: easyocr not installed. OCR disabled.")
            _reader = "disabled"
    return _reader


class JerseyOCRReader:
    def __init__(self, min_bbox_height=150, confidence_threshold=0.5):
        """
        Args:
            min_bbox_height: Minimum bbox height (pixels) to attempt OCR.
                             Below this, jersey numbers are too small to read.
            confidence_threshold: Minimum OCR confidence to accept a reading.
        """
        self.min_bbox_height = min_bbox_height
        self.confidence_threshold = confidence_threshold

        # Vote counter: { player_track_id: Counter({number: count}) }
        self.player_votes = collections.defaultdict(collections.Counter)

        # Confirmed mappings: { player_track_id: jersey_number }
        self.confirmed_numbers = {}

        # Min votes needed to confirm a number
        self.min_votes = 3

        # Cooldown: don't OCR same player every frame
        self.last_ocr_frame = {}
        self.ocr_cooldown = 30  # frames between OCR attempts per player

    def process_frame(self, frame, tracks, frame_count):
        """
        Attempt to read jersey numbers from player bounding boxes.
        Only processes players with large enough bounding boxes.

        Returns: dict of { player_id: jersey_number } for confirmed players.
        """
        reader = _get_reader()
        if reader == "disabled":
            return self.confirmed_numbers

        players = tracks.get("players", {})

        for pid, pdata in players.items():
            # Skip if already confirmed
            if pid in self.confirmed_numbers:
                continue

            # Skip if OCR'd recently
            last = self.last_ocr_frame.get(pid, -999)
            if frame_count - last < self.ocr_cooldown:
                continue

            bbox = pdata.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in bbox]
            bbox_height = y2 - y1
            bbox_width = x2 - x1

            # Only attempt OCR on large enough bboxes
            if bbox_height < self.min_bbox_height:
                continue

            self.last_ocr_frame[pid] = frame_count

            # Crop the upper-middle portion of the bbox (jersey area)
            # Jersey number is typically on the upper 40-70% of the body
            crop_y1 = y1 + int(bbox_height * 0.2)
            crop_y2 = y1 + int(bbox_height * 0.6)
            crop_x1 = x1 + int(bbox_width * 0.15)
            crop_x2 = x2 - int(bbox_width * 0.15)

            # Bounds check
            h, w = frame.shape[:2]
            crop_y1 = max(0, crop_y1)
            crop_y2 = min(h, crop_y2)
            crop_x1 = max(0, crop_x1)
            crop_x2 = min(w, crop_x2)

            if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
                continue

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # OCR
            try:
                results = reader.readtext(
                    crop,
                    allowlist="0123456789",
                    detail=1,
                    paragraph=False,
                )
            except Exception as e:
                continue

            for (bbox_coords, text, conf) in results:
                if conf < self.confidence_threshold:
                    continue

                # Clean text — only keep digits
                digits = "".join(c for c in text if c.isdigit())
                if not digits:
                    continue

                number = int(digits)
                if 1 <= number <= 99:  # Valid jersey number range
                    self.player_votes[pid][number] += 1

                    # Check if any number has enough votes
                    most_common = self.player_votes[pid].most_common(1)
                    if most_common and most_common[0][1] >= self.min_votes:
                        self.confirmed_numbers[pid] = most_common[0][0]
                        print(
                            f"[OCR] Confirmed: Player track#{pid} → Jersey #{most_common[0][0]} "
                            f"({most_common[0][1]} votes)"
                        )
                    break  # Only use first valid reading per crop

        return self.confirmed_numbers

    def get_player_label(self, pid, team=None):
        """Get display label for a player."""
        number = self.confirmed_numbers.get(pid)
        team_label = f"Đội {'A' if team == 1 else 'B'}" if team in [1, 2] else ""

        if number is not None:
            return f"#{number} ({team_label})" if team_label else f"#{number}"
        else:
            return f"Player {pid} ({team_label})" if team_label else f"Player {pid}"
