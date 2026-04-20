import collections
import numpy as np

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class TacticalAnalyzer:
    def __init__(self):
        # Tracking the pass network
        self.pass_cooldown = 0
        self.last_possession_team = None
        self.last_possession_player = None

        # Accumulate passes for VOD or continuous stream analysis
        self.passes_event_log = []
        self.possession_buffer = collections.deque(maxlen=5)

        # Heatmap tracking
        # For simplicity, we track positions in a 10x10 normalized grid
        self.heatmap_grid_size = 10
        self.player_heatmaps = collections.defaultdict(lambda: np.zeros((self.heatmap_grid_size, self.heatmap_grid_size)))

        # We need a reference for pitch dimensions. Assuming normalized bounding boxes or using frame size.
        # But for absolute positions we might just use the min max of observed positions.
        self.observed_min_x = float('inf')
        self.observed_max_x = float('-inf')
        self.observed_min_y = float('inf')
        self.observed_max_y = float('-inf')

    def detect_passes(self, tracks, ball_possession_player, frame_count):
        """
        Detect successful passes between players of the same team.
        """
        players = tracks.get("players", {})

        if self.pass_cooldown > 0:
            self.pass_cooldown -= 1

        if ball_possession_player is None or ball_possession_player not in players:
            return None

        current_team = players[ball_possession_player].get("team")
        if current_team not in [1, 2]:
            return None

        self.possession_buffer.append((ball_possession_player, current_team))
        is_steady = all(x[0] == ball_possession_player for x in self.possession_buffer)

        pass_event = None

        if is_steady:
            if self.last_possession_player is not None and self.last_possession_player != ball_possession_player:
                if self.last_possession_team == current_team and self.pass_cooldown == 0:
                    # snake_case matches EventDetector + Node cvAnalysis.service
                    pass_event = {
                        "event": "pass",
                        "scoring_team": current_team,
                        "from_player": int(self.last_possession_player),
                        "to_player": int(ball_possession_player),
                        "frame": frame_count,
                    }
                    self.passes_event_log.append(pass_event)
                    self.pass_cooldown = 15

            self.last_possession_player = ball_possession_player
            self.last_possession_team = current_team

        return pass_event

    def update_heatmap(self, tracks):
        """
        Update the heatmap grid for each player.
        """
        players = tracks.get("players", {})

        # Update boundaries
        for pid, pdata in players.items():
            if "position_adjusted" in pdata:
                x, y = pdata["position_adjusted"][0], pdata["position_adjusted"][1]
                self.observed_min_x = min(self.observed_min_x, x)
                self.observed_max_x = max(self.observed_max_x, x)
                self.observed_min_y = min(self.observed_min_y, y)
                self.observed_max_y = max(self.observed_max_y, y)

        # Avoid division by zero
        range_x = max(1.0, self.observed_max_x - self.observed_min_x)
        range_y = max(1.0, self.observed_max_y - self.observed_min_y)

        for pid, pdata in players.items():
            if "position_adjusted" in pdata:
                x, y = pdata["position_adjusted"][0], pdata["position_adjusted"][1]

                # Normalize position to 0-1
                norm_x = (x - self.observed_min_x) / range_x
                norm_y = (y - self.observed_min_y) / range_y

                grid_x = min(self.heatmap_grid_size - 1, int(norm_x * self.heatmap_grid_size))
                grid_y = min(self.heatmap_grid_size - 1, int(norm_y * self.heatmap_grid_size))

                self.player_heatmaps[pid][grid_y, grid_x] += 1

    def get_player_heatmap(self, player_id):
        """Return the heatmap array as list of lists for a specific player."""
        if player_id in self.player_heatmaps:
            return self.player_heatmaps[player_id].tolist()
        return []

    def _detect_formation(self, players, team_id):
        """
        Detect formation based on real player X-positions (along the pitch).
        Removes the goalkeeper, then clusters outfield players into 3 layers
        (defenders, midfielders, forwards) using normalized position thresholds.
        Returns a string like '4-3-3' or '4-4-2'.
        """
        team_positions = []
        for pid, pdata in players.items():
            if pdata.get("team") == team_id and "position_adjusted" in pdata:
                team_positions.append(pdata["position_adjusted"][0])

        count = len(team_positions)
        if count < 7:
            return "Building/Unknown"

        sorted_x = sorted(team_positions)

        # Remove goalkeeper (extreme position closest to own goal)
        if team_id == 1:
            sorted_x = sorted_x[1:]   # leftmost is GK for left-side team
        else:
            sorted_x = sorted_x[:-1]  # rightmost is GK for right-side team

        if len(sorted_x) < 3:
            return "Building/Unknown"

        # Normalize outfield positions to 0–1 range
        min_x = sorted_x[0]
        max_x = sorted_x[-1]
        range_x = max(max_x - min_x, 1)
        normalized = [(x - min_x) / range_x for x in sorted_x]

        # Classify into 3 layers
        defenders = sum(1 for n in normalized if n < 0.35)
        midfielders = sum(1 for n in normalized if 0.35 <= n < 0.70)
        forwards = sum(1 for n in normalized if n >= 0.70)

        # Fallback: if any line is empty, split evenly into thirds
        if defenders == 0 or forwards == 0:
            n = len(sorted_x)
            third = n // 3
            remainder = n - 2 * third
            defenders = third
            midfielders = remainder
            forwards = third

        return f"{defenders}-{midfielders}-{forwards}"

    def _compute_space_control(self, t1_pts, t2_pts):
        """
        Compute space control using convex hull area (scipy) for accuracy,
        with bounding box area as fallback.
        """
        def area_of(pts):
            if len(pts) < 3:
                return 0.0
            if _HAS_SCIPY:
                try:
                    pts_array = np.array(pts)
                    hull = ConvexHull(pts_array)
                    return hull.volume  # in 2D, volume = area
                except Exception:
                    pass
            # Fallback: bounding box area
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        a1 = area_of(t1_pts)
        a2 = area_of(t2_pts)

        space = {"team1": 50.0, "team2": 50.0}
        if a1 + a2 > 0:
            space["team1"] = round((a1 / (a1 + a2)) * 100, 1)
            space["team2"] = round((a2 / (a1 + a2)) * 100, 1)
        return space

    def analyze_tactics(self, tracks):
        """
        Analyze formation and rough space control.
        """
        players = tracks.get("players", {})
        t1_count = sum(1 for p in players.values() if p.get("team") == 1)
        t2_count = sum(1 for p in players.values() if p.get("team") == 2)

        # Real formation detection based on player positions
        formation = {}
        if t1_count >= 7:
            formation[1] = self._detect_formation(players, 1)
        elif t1_count > 0:
            formation[1] = "Building/Unknown"

        if t2_count >= 7:
            formation[2] = self._detect_formation(players, 2)
        elif t2_count > 0:
            formation[2] = "Building/Unknown"

        # Space control using convex hull area
        t1_pts = [p["position_adjusted"] for p in players.values() if p.get("team") == 1 and "position_adjusted" in p]
        t2_pts = [p["position_adjusted"] for p in players.values() if p.get("team") == 2 and "position_adjusted" in p]

        space_control = self._compute_space_control(t1_pts, t2_pts)

        # Build combined team heatmaps (10x10)
        t1_heatmap = np.zeros((self.heatmap_grid_size, self.heatmap_grid_size))
        t2_heatmap = np.zeros((self.heatmap_grid_size, self.heatmap_grid_size))

        for pid, hmap in self.player_heatmaps.items():
            # Figure out which team the player is on (approximate by checking last track)
            if pid in players:
                team = players[pid].get("team")
                if team == 1:
                    t1_heatmap += hmap
                elif team == 2:
                    t2_heatmap += hmap

        # Optional: Normalize heatmaps (values 0.0 to 1.0)
        max_t1 = np.max(t1_heatmap) if np.max(t1_heatmap) > 0 else 1
        max_t2 = np.max(t2_heatmap) if np.max(t2_heatmap) > 0 else 1

        norm_t1 = (t1_heatmap / max_t1).tolist()
        norm_t2 = (t2_heatmap / max_t2).tolist()

        return {
            "formations": formation,
            "space_control": space_control,
            "total_passes": len(self.passes_event_log),
            "heatmaps": {
                "team1": norm_t1,
                "team2": norm_t2
            }
        }
