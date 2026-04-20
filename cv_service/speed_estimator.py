import numpy as np
import collections

class SpeedEstimator:
    def __init__(self, fps=30, history_frames=15, pixel_to_meter_scale=0.05):
        """
        Estimate relative speed, total distance, and sprints of players.
        """
        self.fps = fps
        self.history_frames = history_frames
        self.pixel_to_meter_scale = pixel_to_meter_scale
        
        # Store recent positions: { player_id: deque([(frame, x, y), ...]) }
        self.player_history = collections.defaultdict(lambda: collections.deque(maxlen=self.history_frames))
        
        # Current logic state
        self.current_speeds = {}
        self.total_distances = collections.defaultdict(float)
        self.sprint_counts = collections.defaultdict(int)
        # Store highest ever speed to create a stable leaderboard
        self.top_speeds_ever = collections.defaultdict(float)
        self.player_teams = {}
        # Track sprint state
        self.is_sprinting = collections.defaultdict(bool)
        
        # Thresholds
        self.SPRINT_THRESHOLD_KMH = 24.0

    def update_scale(self, new_scale: float):
        """Dynamic recalibration of pixel-to-meter scale (#3)."""
        if 0.005 <= new_scale <= 0.5:
            self.pixel_to_meter_scale = new_scale

    def update_speeds(self, tracks, frame_count):
        """
        Update player position history, compute speeds, total distance, and sprints.
        Returns top 5 fastest players and accumulated stats based on ALL-TIME max speeds.
        """
        players = tracks.get("players", {})
        
        for pid, pdata in players.items():
            if "position_adjusted" not in pdata:
                continue
                
            pos = pdata["position_adjusted"]
            x, y = pos[0], pos[1]
            
            # Update team mapping for leaderboard memory
            team_label = pdata.get("team", 0)
            if team_label in [1, 2]:
                self.player_teams[pid] = team_label

            # Distance logic for total distance
            if len(self.player_history[pid]) > 0:
                last_frame, last_x, last_y = self.player_history[pid][-1]
                # Increment total distance for each frame jump
                dist_px = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                self.total_distances[pid] += dist_px * self.pixel_to_meter_scale
                
            self.player_history[pid].append((frame_count, x, y))
            
            history = self.player_history[pid]
            if len(history) < 5:
                self.current_speeds[pid] = 0.0
                continue
                
            # Calculate speed over history window
            old_frame, old_x, old_y = history[0]
            new_frame, new_x, new_y = history[-1]
            
            frame_diff = new_frame - old_frame
            if frame_diff <= 0:
                continue
                
            # Distance in pixels
            dist_px = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
            
            # Distance in meters
            dist_m = dist_px * self.pixel_to_meter_scale
            
            # Speed in m/s
            dt_sec = frame_diff / self.fps
            speed_ms = dist_m / dt_sec
            
            # Speed in km/h
            speed_kmh = speed_ms * 3.6
            self.current_speeds[pid] = round(speed_kmh, 1)

            # Update highest speed ever (Ignore unrealistic speeds > 38.0 km/h as tracking glitches)
            if 5.0 < self.current_speeds[pid] <= 38.0:
                if self.current_speeds[pid] > self.top_speeds_ever[pid]:
                    self.top_speeds_ever[pid] = self.current_speeds[pid]

            # Sprint counting logic
            if speed_kmh >= self.SPRINT_THRESHOLD_KMH:
                if not self.is_sprinting[pid]:
                    self.sprint_counts[pid] += 1
                    self.is_sprinting[pid] = True
            else:
                # If speed drops below e.g., 20 kmh, reset sprint state
                if speed_kmh < (self.SPRINT_THRESHOLD_KMH - 4.0):
                    self.is_sprinting[pid] = False

        # Sort ALL top speeds ever recorded, pick top 5
        sorted_speeds = sorted(
            [(pid, speed) for pid, speed in self.top_speeds_ever.items() if pid in self.player_teams], 
            key=lambda item: item[1], 
            reverse=True
        )[:5]

        top_players_list = [
            {
                "player_id": int(pid), 
                "team": self.player_teams[pid], 
                "speed": speed,
                "total_distance": round(self.total_distances[pid], 2),
                "sprint_count": self.sprint_counts[pid]
            }
            for pid, speed in sorted_speeds
        ]

        return top_players_list

    def get_player_stats(self, player_id):
        """Return the distance and sprint count for a specific player."""
        return {
            "total_distance_m": round(self.total_distances.get(player_id, 0), 2),
            "sprint_count": self.sprint_counts.get(player_id, 0)
        }
