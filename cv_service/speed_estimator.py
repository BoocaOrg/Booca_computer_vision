import numpy as np
import collections

class SpeedEstimator:
    def __init__(self, fps=30, history_frames=15, pixel_to_meter_scale=0.05):
        """
        Estimate relative speed of players using pixel displacement.
        """
        self.fps = fps
        self.history_frames = history_frames
        self.pixel_to_meter_scale = pixel_to_meter_scale
        
        # Store recent positions: { player_id: deque([(frame, x, y), ...]) }
        self.player_history = collections.defaultdict(lambda: collections.deque(maxlen=self.history_frames))
        
        # Current speeds: { player_id: speed_kmh }
        self.current_speeds = {}

    def update_speeds(self, tracks, frame_count):
        """
        Update player position history and compute moving average speed.
        Returns top 5 fastest players.
        """
        players = tracks.get("players", {})
        
        # Calculate time delta for the history window
        # dt = (current_frame - oldest_frame) / fps
        
        top_speeds_frame = {}

        for pid, pdata in players.items():
            if "position_adjusted" not in pdata:
                continue
                
            pos = pdata["position_adjusted"]
            # pos is typically [x, y], but might be embedded in a longer array depending on bbox format
            # Using bottom center of bounding box or adjusted position
            x, y = pos[0], pos[1]
            
            self.player_history[pid].append((frame_count, x, y))
            
            history = self.player_history[pid]
            if len(history) < 5:
                # Not enough data to estimate smooth speed
                self.current_speeds[pid] = 0.0
                continue
                
            # Calculate distance between oldest and newest position in history
            old_frame, old_x, old_y = history[0]
            new_frame, new_x, new_y = history[-1]
            
            frame_diff = new_frame - old_frame
            if frame_diff <= 0:
                continue
                
            # Distance in pixels
            dist_px = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
            
            # Distance in meters
            dist_m = dist_px * self.pixel_to_meter_scale
            
            # Time in seconds
            dt_sec = frame_diff / self.fps
            
            # Speed in m/s
            speed_ms = dist_m / dt_sec
            
            # Speed in km/h
            speed_kmh = speed_ms * 3.6
            
            # Simple low-pass filter (moving average over the frame history window)
            self.current_speeds[pid] = round(speed_kmh, 1)

            team_label = pdata.get("team", 0)
            if team_label in [1, 2] and self.current_speeds[pid] > 5.0: # Filter out standing players
                top_speeds_frame[pid] = {
                    "team": team_label,
                    "speed": self.current_speeds[pid]
                }

        # Sort and get top 5
        sorted_speeds = sorted(
            top_speeds_frame.items(), 
            key=lambda item: item[1]["speed"], 
            reverse=True
        )[:5]

        # Format output
        top_players_list = [
            {
                "player_id": int(pid), 
                "team": data["team"], 
                "speed": data["speed"]
            }
            for pid, data in sorted_speeds
        ]

        return top_players_list
