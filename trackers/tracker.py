import logging
import os
from typing import List, Dict
import time
from datetime import datetime
import numpy as np
import pandas as pd
import ultralytics
import supervision as sv
from utils import ellipse, triangle, ball_possession_box, get_device, get_center_of_bbox, get_foot_position, options

os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/tracking.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("tracker")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class Tracker:
    """
    Byte tracker. 
    Tracking persons by close bounding box in next frame combined with movement and visual features like shirt colour.
    Assigning bounding boxes unique IDs.
    Predicting and then tracking with supervision instead of YOLO tracking due to overwriting goalkeepers.
    """
    def __init__(self, model_path: str, classes: List[int], verbose: bool=True, fp16: bool=False, imgsz: int=640) -> None:
        base_path = model_path.replace(".pt", "")
        ext = os.path.splitext(model_path)[1]

        # Priority: .engine (TensorRT) > .onnx > .pt (PyTorch)
        if ext == ".engine" or os.path.exists(base_path + ".engine"):
            model_file = base_path + ".engine" if ext != ".engine" else model_path
            self.model = ultralytics.YOLO(model_file)
            self.model_type = "tensorrt"
            self.fp16 = False
            if verbose:
                print(f"[Tracker] Loaded TensorRT engine: {model_file}")
        elif ext == ".onnx" or os.path.exists(base_path + ".onnx"):
            model_file = base_path + ".onnx" if ext != ".onnx" else model_path
            self.model = ultralytics.YOLO(model_file)
            self.model_type = "onnx"
            self.fp16 = False
            if verbose:
                print(f"[Tracker] Loaded ONNX model: {model_file}")
        else:
            self.model = ultralytics.YOLO(model_path)
            self.model_type = "pytorch"
            self.fp16 = False  # Fixed: fp16 causes expected m1 and m2 to have the same dtype error
            if self.fp16:
                self.model.half()
            if verbose:
                print(f"[Tracker] Loaded PyTorch model: {model_path} (fp16={self.fp16})")

        self.classes = classes
        self.tracker = sv.ByteTrack()
        self.verbose = verbose
        self.imgsz = imgsz
        self.interpolation_tracker = None   # used for ball annotation: don't draw ball in a large interpolation window

        # Detect if this is a COCO model (yolov8n.pt etc.) vs custom-trained model
        # COCO models use "person"/"sports ball"; custom model uses "player"/"goalkeeper"/"ball"/"referee"
        model_names = self.model.names if hasattr(self.model, 'names') else {}
        model_class_values = set(model_names.values()) if model_names else set()
        self._is_coco_model = "person" in model_class_values and "player" not in model_class_values
        if self._is_coco_model and verbose:
            print("[Tracker] COCO model detected — mapping person→player, sports ball→ball")

    def _resolve_class_names(self, cls_names_switched: dict) -> dict:
        """
        Resolve class name → class ID mapping, handling both custom and COCO models.
        For COCO models, maps: person→player, sports ball→ball.
        Returns dict with keys: 'player', 'goalkeeper', 'ball', 'referee' (some may be missing).
        """
        if not self._is_coco_model:
            return cls_names_switched

        resolved = {}
        # person → player (and goalkeeper)
        if "person" in cls_names_switched:
            resolved["player"] = cls_names_switched["person"]
            resolved["goalkeeper"] = cls_names_switched["person"]  # same class for COCO
        # sports ball → ball
        if "sports ball" in cls_names_switched:
            resolved["ball"] = cls_names_switched["sports ball"]
        # COCO has no "referee" class — leave it out so lookups return None gracefully
        return resolved

    def interpolate_ball_positions(self, ball_tracks: List[Dict]) -> List[Dict]:
        """
        If the ball is not detected in every frame, take the frames where it is detected and interpolate
        ball position in the frames between by drawing a line and simulate the position evenly along the line.
        """
        # tracker_id 1 for ball, {} if nothing found, at tracker_id 1, get bbox, [] if nothing found
        # {1: {"bbox": [....], ...
        ball_positions = [track.get(1, {}).get("bbox", []) for track in ball_tracks]

        self.interpolation_tracker = [1 if not bbox else 0 for bbox in ball_positions]   # if no datapoint: empty list --> interpolation
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        df_ball_positions = df_ball_positions.interpolate() # interpolate NaN values, estimation based on previous data
        df_ball_positions = df_ball_positions.bfill()       # --> edge case beginning: no previous data --> bfill (replace NaN with next following value)

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()] # transform back

        return ball_positions

    def detect_frames(self, frames: List[np.ndarray], batch_size: int=20) -> list:
        """
        List of frame predictions processed in batches to avoid memory issues.
        """
        detections = []

        
        start_time = time.time()

        if self.verbose:
            logger.info(f"[Device: {get_device()}] Starting object detection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i in range(0, len(frames), batch_size):
            frame_time = time.time()
            
            detections_batch = self.model.predict(source=frames[i:i+batch_size], conf=0.15, verbose=self.verbose, device=get_device(), imgsz=self.imgsz, half=self.fp16)
            detections += detections_batch

            if self.verbose:
                logger.info(f"Processed frames {i} to {min(i+batch_size-1, len(frames))} in {time.time() - frame_time:.2f} seconds.")
        
        if self.verbose:
            logger.info(f"Detected objects in {len(frames)} frames in {time.time() - start_time:.2f} seconds.")
  
        return detections

    def get_object_tracks(self, frames: List[np.ndarray]) -> Dict[str, List[Dict]]:        
        detections = self.detect_frames(frames)

        # key: tracker_id, value: bbox, index: frame
        tracks = {
            "players": [],      # {tracker_id: {"bbox": [....]}}, tracker_id: {"bbox": [....]}}, tracker_id: {"bbox": [....]}}, tracker_id: {"bbox": [....]}}}  (same for referees and ball)
            "referees": [],     
            "ball": []
        }

        start_time = time.time()

        if self.verbose:
            logger.info(f"[Device: {get_device()}] Starting object tracking at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_switched = {v: k for k, v in cls_names.items()}       # swap keys and values, e.g. ball: 1 --> 1: ball for easier access
            resolved = self._resolve_class_names(cls_names_switched)

            # convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)      # xyxy bboxes
            
            # convert goalkeeper to player
            # goalkeepers might get predicted as players in some frames and that could cause tracking issues
            player_cls_id = resolved.get("player")
            goalkeeper_cls_id = resolved.get("goalkeeper")
            for index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper" and player_cls_id is not None:
                    detection_supervision.class_id[index] = player_cls_id
            
            # track objects
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({}) 
            tracks["referees"].append({})
            tracks["ball"].append({})

            referee_cls_id = resolved.get("referee")
            ball_cls_id = resolved.get("ball")

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                tracker_id = frame_detection[4]

                if player_cls_id is not None and class_id == player_cls_id:
                    tracks["players"][frame_num][tracker_id] = {"bbox": bbox}

                if referee_cls_id is not None and class_id == referee_cls_id:
                    tracks["referees"][frame_num][tracker_id] = {"bbox": bbox}

            # no tracker for the ball as there is only one
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if ball_cls_id is not None and class_id == ball_cls_id and frame_detection[2] >= 0.3:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}   # ID 1 as there is only one ball

        if self.verbose:
            logger.info(f"Tracked objects in {len(frames)} frames in {time.time() - start_time:.2f} seconds.")

            separator = f"{'-'*10} [End of tracking] at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {'-'*10}"
            logger.info(separator)

        return tracks
    
    def add_position_to_tracks(self, tracks: Dict[str, List[Dict]]) -> None:
        for object, object_tracks in tracks.items():
            for frame_num, track_dict in enumerate(object_tracks):
                for tracker_id, track in track_dict.items():
                    bbox = track["bbox"]

                    if object == "ball":
                        position = get_center_of_bbox(bbox)
                    else:   # player
                        position = get_foot_position(bbox)

                    # add new key position
                    tracks[object][frame_num][tracker_id]["position"] = position
    
    def draw_annotations(self, frames: List[np.ndarray], tracks: Dict[str, List[Dict]], ball_possession: np.ndarray) -> List[np.ndarray]:   # TODO extra folder for custom drawings and then import?
        output_frames = []  # frames after changing the annotations
        num_interpolated = 0

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()       # don't change original

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            if options["players"] in self.classes:
                for tracker_id, player in player_dict.items():
                    colour = player.get("team_colour", (255, 255, 255))         # get team colour if it exists, else white 
                    frame = ellipse(frame, player["bbox"], colour, tracker_id)

                    if player.get("has_ball", False):
                        frame = triangle(frame, player["bbox"], (0, 0, 255))    # red triangle

            if options["referees"] in self.classes: 
                for _, referee in referee_dict.items():
                    frame = ellipse(frame, referee["bbox"], (0, 255, 255))      # yellow ellipse

            if options["ball"] in self.classes:
                if self.interpolation_tracker[frame_num] == 1:
                    num_interpolated += 1
                else:
                    num_interpolated = 0 
                
                for tracker_id, ball in ball_dict.items():
                    # only draw detected ball or if not too many consecutive interpolated trackings 
                    if num_interpolated <= 25 or self.interpolation_tracker[frame_num] == 0:
                        frame = triangle(frame, ball["bbox"], (0, 255, 0))          # green triangle
            
            if options["stats"] in self.classes: 
                frame = ball_possession_box(frame_num, frame, ball_possession)

            output_frames.append(frame)

        return output_frames
    
    # ============================================================================
    # REALTIME PROCESSING METHODS (Single Frame)
    # ============================================================================
    
    def get_object_tracks_single_frame(self, frame: np.ndarray) -> Dict[str, Dict]:
        """
        Process a single frame and return tracking information.
        Returns dict with format: {"players": {id: {"bbox": [...]}}, "referees": {...}, "ball": {...}}
        """
        # Detect objects in single frame
        detection = self.model.predict(source=frame, conf=0.15, verbose=False, device=get_device(), imgsz=self.imgsz, half=self.fp16)[0]
        
        cls_names = detection.names
        cls_names_switched = {v: k for k, v in cls_names.items()}
        resolved = self._resolve_class_names(cls_names_switched)
        
        # Convert to supervision format
        detection_supervision = sv.Detections.from_ultralytics(detection)
        
        # Convert goalkeeper to player
        player_cls_id = resolved.get("player")
        for index, class_id in enumerate(detection_supervision.class_id):
            if cls_names[class_id] == "goalkeeper" and player_cls_id is not None:
                detection_supervision.class_id[index] = player_cls_id
        
        # Track objects
        detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
        
        # Build tracks dict
        tracks = {
            "players": {},
            "referees": {},
            "ball": {}
        }
        
        referee_cls_id = resolved.get("referee")
        ball_cls_id = resolved.get("ball")
        
        for frame_detection in detections_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]
            
            if player_cls_id is not None and cls_id == player_cls_id:
                tracks["players"][track_id] = {"bbox": bbox}
            elif referee_cls_id is not None and cls_id == referee_cls_id:
                tracks["referees"][track_id] = {"bbox": bbox}
        
        # Ball tracking (no tracking ID, just detection)
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            conf = frame_detection[2]

            if ball_cls_id is not None and cls_id == ball_cls_id and conf >= 0.3:
                tracks["ball"][1] = {"bbox": bbox}
                break  # Only one ball
        
        return tracks
    
    def add_position_to_tracks_single_frame(self, tracks: Dict[str, Dict]) -> None:
        """Add position (center of bbox) to tracks for single frame."""
        for object_name, object_tracks in tracks.items():
            for track_id, track_info in object_tracks.items():
                bbox = track_info["bbox"]
                if object_name == "ball":
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object_name][track_id]["position"] = position
    
    def draw_annotations_single_frame(self, frame: np.ndarray, tracks: Dict[str, Dict], 
                                      ball_possession: np.ndarray = None) -> np.ndarray:
        """
        Draw annotations on a single frame.
        """
        frame = frame.copy()
        
        # Draw players
        for player_id, player_track in tracks.get("players", {}).items():
            colour = player_track.get("team_colour", (0, 255, 0))
            frame = ellipse(frame, player_track["bbox"], colour, player_id)
            
            # Check ball possession
            if player_track.get("has_ball", False):
                frame = triangle(frame, player_track["bbox"], (0, 0, 255))
        
        # Draw referees
        for referee_id, referee_track in tracks.get("referees", {}).items():
            frame = ellipse(frame, referee_track["bbox"], (0, 255, 255), referee_id)
        
        # Draw ball
        for ball_id, ball_track in tracks.get("ball", {}).items():
            frame = triangle(frame, ball_track["bbox"], (0, 255, 0))
        
        return frame