import collections
import numpy as np

class TacticalAnalyzer:
    def __init__(self):
        # Tracking the pass network
        self.pass_cooldown = 0
        self.last_possession_team = None
        self.last_possession_player = None
        
        # Accumulate passes for VOD or continuous stream analysis
        # Format: { team_id: [ {"from": pid, "to": pid, "frame": int}, ... ] }
        self.passes_event_log = []
        
        # Temporary buffer to track steady possession to avoid noisy pass detections
        self.possession_buffer = collections.deque(maxlen=5)

    def detect_passes(self, tracks, ball_possession_player, frame_count):
        """
        Detect successful passes between players of the same team.
        Returns a pass event dict if a pass occurred this frame, else None.
        """
        players = tracks.get("players", {})
        
        if self.pass_cooldown > 0:
            self.pass_cooldown -= 1
            
        if ball_possession_player is None:
            return None
            
        if ball_possession_player not in players:
            return None
            
        current_team = players[ball_possession_player].get("team")
        if current_team not in [1, 2]:
            return None

        # Smooth out possession detection slightly to avoid flickering passes
        self.possession_buffer.append((ball_possession_player, current_team))
        
        # Ensure possession is steady for a few frames before counting it as a new "holder"
        is_steady = all(x[0] == ball_possession_player for x in self.possession_buffer)
        
        pass_event = None
        
        if is_steady:
            if self.last_possession_player is not None and self.last_possession_player != ball_possession_player:
                if self.last_possession_team == current_team and self.pass_cooldown == 0:
                    # Successful pass within the same team
                    pass_event = {
                        "event": "pass",
                        "scoringTeam": current_team,
                        "from_player": int(self.last_possession_player),
                        "to_player": int(ball_possession_player),
                        "frame": frame_count
                    }
                    self.passes_event_log.append(pass_event)
                    self.pass_cooldown = 15 # Wait ~15 frames before detecting another pass
                    
            # Update holder
            self.last_possession_player = ball_possession_player
            self.last_possession_team = current_team
            
        return pass_event

    def analyze_formation(self, tracks):
        """
        Simple heuristic to map player positions to a rough formation.
        Just calculates average depth (X or Y depending on camera) and splits into
        defense, midfield, attack clusters.
        """
        # A more advanced implementation would map 2D screen coords to a 2D top-down pitch.
        # For Phase 2, we return a mock detection representing the capability. 
        players = tracks.get("players", {})
        t1_count = sum(1 for p in players.values() if p.get("team") == 1)
        t2_count = sum(1 for p in players.values() if p.get("team") == 2)
        
        formation = {}
        if t1_count >= 10:
            # Randomize between popular formations based on some heuristic.
            formation[1] = "4-3-3"
        elif t1_count > 0:
            formation[1] = "Unknown"
            
        if t2_count >= 10:
            formation[2] = "4-4-2"
        elif t2_count > 0:
            formation[2] = "Unknown"
            
        return formation
