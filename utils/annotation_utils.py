from typing import List, Tuple
import numpy as np
import cv2
from . import get_center_of_bbox, get_bbox_dimensions

# data.yaml class IDs
# ball: 0, goalkeeper: 1, player: 2, referee: 3
# additional for drawing stats: 4
options = {"ball": 0, "goalkeepers": 1, "players": 2, "referees": 3, "stats": 4}

def ellipse(frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int], tracker_id: int=None):
    # xyxy bboxes --> y2 at last index
    y2 = int(bbox[3])   # ellipse should be below the player

    x_center, _ = get_center_of_bbox(bbox)
    width, _ = get_bbox_dimensions(bbox)

    cv2.ellipse(frame, 
                center=(x_center, y2), 
                axes=(width,                # length of major axis
                      int(0.35*width)),     # length of minor axis
                angle=0,                    # rotation of the ellipse
                startAngle=-45,             # start and end: upper part of ellipse not drawn
                endAngle=235,    
                color=colour, 
                thickness=2)
    
    if tracker_id:
        rect_width = 40
        rect_height = 20
        rect_x1 = x_center - rect_width // 2
        rect_x2 = x_center + rect_width // 2
        rect_y1 = y2 - rect_height // 2 + 15       # padding
        rect_y2 = y2 + rect_height // 2 + 15       # padding

        cv2.rectangle(frame, 
                    pt1=(rect_x1, rect_y1),   # top left corner
                    pt2=(rect_x2, rect_y2),   # bottom right corner
                    color=colour,
                    thickness=cv2.FILLED)

        # size of the text for centering
        text = str(tracker_id)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        text_x = rect_x1 + (rect_width - text_width) // 2
        text_y = rect_y1 + (rect_height + text_height) // 2

        cv2.putText(frame, 
                    text=text, 
                    org=(text_x, text_y),  # bottom left corner of the text
                    fontFace=font_face, 
                    fontScale=font_scale, 
                    color=(0, 0, 0), 
                    thickness=thickness)

    return frame
    
def triangle(frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int]):
    y = int(bbox[1])    # y1 --> top of the ball
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],                 # bottom corner
        [x - 8, y - 15],       # top left corner
        [x + 8, y - 15],       # top right corner
    ])   

    cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=colour, thickness=cv2.FILLED)   
    cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=(0, 0, 0), thickness=2)   # border  

    return frame 

# Running counters for O(1) ball possession calculation (reset per video)
_team1_frames = 0
_team2_frames = 0

def ball_possession_box(frame_num: int, frame: np.ndarray, ball_possession: np.ndarray) -> np.ndarray:
    global _team1_frames, _team2_frames

    # Detect new video: ball_possession length shrinks (replay) or resets
    # We detect a reset by checking if frame_num is 0 or if ball_possession is very short
    if frame_num == 0:
        _team1_frames = 0
        _team2_frames = 0

    overlay = frame.copy()

    cv2.rectangle(overlay, pt1=(1350, 850), pt2=(1900, 970), color=(255, 255, 255), thickness=cv2.FILLED)
    alpha = 0.4
    cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1-alpha, gamma=0, dst=frame)

    current_team = ball_possession[frame_num]
    if current_team == 1:
        _team1_frames += 1
    elif current_team == 2:
        _team2_frames += 1

    total = _team1_frames + _team2_frames

    if total > 0:
        team_1_possession = int(round(_team1_frames / total, 2) * 100)
        team_2_possession = int(round(_team2_frames / total, 2) * 100)
    else:
        team_1_possession = 0
        team_2_possession = 0

    cv2.putText(frame, text=f"Team 1: {team_1_possession}%", org=(1400, 900), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
    cv2.putText(frame, text=f"Team 2: {team_2_possession}%", org=(1400, 950), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

    return frame