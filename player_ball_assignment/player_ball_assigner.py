from typing import List, Dict
import numpy as np
from utils import get_center_of_bbox, get_distance

class PlayerBallAssigner():
    def __init__(self) -> None:
        self.max_player_ball_distance = 70  # pixels
        self.ball_possession = None
        self.player_has_ball = None

    def assign_ball_to_player(self, player_tracks: Dict, ball_bbox: List[float]) -> int:
        ball_position = get_center_of_bbox(ball_bbox)

        min_distance = float("inf")     # used to find player closest to the ball
        assigned_player = -1
        
        for player_id, player in player_tracks.items():
            player_bbox = player["bbox"]

            distance_left_foot = get_distance((player_bbox[0], player_bbox[3]), ball_position)   # assumption that foot is in the corner of the bbox
            distance_right_foot = get_distance((player_bbox[2], player_bbox[3]), ball_position)

            distance = min(distance_left_foot, distance_right_foot)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

        return assigned_player
    
    def get_player_and_possession(self, tracks: Dict[str, List[Dict]]) -> None:
        ball_possession = []

        for frame_num, player in enumerate(tracks["players"]):
            ball_frame = tracks["ball"][frame_num]
            # Handle missing ball detection (KeyError guard)
            if not ball_frame or 1 not in ball_frame:
                if len(ball_possession) > 0:
                    ball_possession.append(ball_possession[-1])
                else:
                    ball_possession.append(0)
                continue

            ball_bbox = ball_frame[1]["bbox"]
            assigned_player = self.assign_ball_to_player(player, ball_bbox)

            if assigned_player != -1:
                # add new key has_ball
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team = tracks["players"][frame_num][assigned_player].get("team", 0)
                ball_possession.append(team)
            elif len(ball_possession) > 0:
                ball_possession.append(ball_possession[-1])
            else:
                ball_possession.append(0)

        # ball possession counter in ball_possession_box() requires numpy array
        ball_possession = np.array(ball_possession) if ball_possession else np.array([])

        self.ball_possession = ball_possession
    
    def assign_ball_single_frame(self, tracks: Dict[str, Dict]) -> None:
        """
        Assign ball to player for a single frame in realtime processing.
        Updates ball_possession list incrementally.
        """
        # Initialize if first frame
        if self.ball_possession is None:
            self.ball_possession = []
        
        # Reset player with ball for new frame
        self.player_has_ball = None

        # Get ball and player data for this frame
        ball_dict = tracks.get("ball", {})
        player_dict = tracks.get("players", {})
        
        # If no ball detected, keep previous possession
        if not ball_dict or 1 not in ball_dict:
            if len(self.ball_possession) > 0:
                self.ball_possession.append(self.ball_possession[-1])
            else:
                self.ball_possession.append(0)
            return
        
        ball_bbox = ball_dict[1]["bbox"]
        assigned_player = self.assign_ball_to_player(player_dict, ball_bbox)
        
        self.player_has_ball = assigned_player if assigned_player != -1 else None

        if assigned_player != -1 and assigned_player in player_dict:
            # Mark player as having the ball
            tracks["players"][assigned_player]["has_ball"] = True
            
            # Add team to possession if team assigned
            if "team" in player_dict[assigned_player]:
                team = player_dict[assigned_player]["team"]
                self.ball_possession.append(team)
            elif len(self.ball_possession) > 0:
                self.ball_possession.append(self.ball_possession[-1])
            else:
                self.ball_possession.append(0)
        elif len(self.ball_possession) > 0:
            # No player assigned, keep previous possession
            self.ball_possession.append(self.ball_possession[-1])
        else:
            self.ball_possession.append(0)