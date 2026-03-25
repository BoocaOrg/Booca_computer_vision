from typing import List, Optional, Union
import os
import cv2
import argparse
import numpy as np
import warnings
from utils import options
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

try:
    from scripts.push_frame import BroadcastPusher
    _HAS_PUSHER = True
except Exception:
    _HAS_PUSHER = False

def process_realtime(source: Union[str, int], classes: List[int], verbose: bool=True, frame_skip: int=1, fp16: bool=False, imgsz: int=640, nvdec: bool=False, model_path: str="models/best.pt", broadcast: bool=False) -> None:
    """
    Process video in real-time from webcam, video file, or stream URL.
    
    Args:
        source: Video source - int for webcam index, str for file path or URL
        classes: List of class IDs to track
        verbose: Enable verbose output
    """
    # Enable NVDEC hardware decode for NVIDIA GPUs
    if nvdec:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "hwaccel;cuda|video_codec;h264_cuvid"

    # Initialize Video Capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Read first frame to initialize components
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    # Initialize Components
    tracker = Tracker(model_path, classes, verbose=False, fp16=fp16, imgsz=imgsz)
    camera_movement_estimator = CameraMovementEstimator(frame, classes, verbose=False)
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    
    # Calibration state
    teams_assigned = False

    # Initialize broadcast pusher
    pusher = None
    if broadcast and _HAS_PUSHER:
        try:
            pusher = BroadcastPusher("http://localhost:8502", enabled=True)
            print("[Broadcast] Pusher enabled — stream to http://localhost:8502/mjpeg")
        except Exception as e:
            print(f"[Broadcast] Init failed: {e}")

    print("Starting real-time processing... Press 'q' to quit.")

    frame_count = 0
    last_tracks = None

    while True:
        frame_count += 1

        # Frame skip: only run detection every N frames
        if frame_skip > 1 and frame_count % frame_skip != 1:
            # Use last known tracks for visualization of skipped frames
            tracks = last_tracks if last_tracks is not None else {"players": {}, "referees": {}, "ball": {}}

            # Draw annotations for skipped frame (basic, no processing)
            output_frame = tracker.draw_annotations_single_frame(frame, tracks, None)
            cv2.imshow("Football AI Analysis - Press 'q' to quit", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()
            if not ret:
                print("Stream ended or video finished.")
                break
            continue

        # Track objects
        tracks = tracker.get_object_tracks_single_frame(frame)
        last_tracks = tracks
        tracker.add_position_to_tracks_single_frame(tracks)

        # Camera Movement
        camera_movement = camera_movement_estimator.get_camera_movement_single_frame(frame)
        camera_movement_estimator.adjust_positions_to_tracks_single_frame(tracks, camera_movement)

        # Team Assignment
        players_in_frame = tracks.get("players", {})
        
        if not teams_assigned and len(players_in_frame) > 0:
            # Calibrate teams on first frame with players
            team_assigner.assign_team_colour(frame, players_in_frame, force=True)
            teams_assigned = True
            if verbose:
                print("Teams calibrated.")
        
        if teams_assigned:
            for player_id, player_track in players_in_frame.items():
                team = team_assigner.get_player_team(frame, player_track["bbox"], player_id)
                tracks["players"][player_id]["team"] = team
                tracks["players"][player_id]["team_colour"] = team_assigner.team_colours[team]

        # Ball Assignment
        player_assigner.assign_ball_single_frame(tracks)
        
        # Draw Annotations
        if player_assigner.ball_possession:
            # Replace None with -1 for numpy array
            sanitized_possession = [-1 if x is None else x for x in player_assigner.ball_possession]
            ball_possession_np = np.array(sanitized_possession)
        else:
            ball_possession_np = None
            
        output_frame = tracker.draw_annotations_single_frame(frame, tracks, ball_possession_np)
        output_frame = camera_movement_estimator.draw_camera_movement_single_frame(output_frame, camera_movement)

        # Push annotated frame to broadcast server
        if pusher is not None:
            pusher.push(output_frame)

        # Display
        cv2.imshow("Football AI Analysis - Press 'q' to quit", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read next frame
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or video finished.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

def _classes(classes: List[str]) -> List[int]:
    """Convert class names to class IDs."""
    class_ids = [value for key, value in options.items() if key in classes]
    invalid_classes = [cls for cls in classes if cls not in options.keys()]
    
    if len(invalid_classes) == len(options):
        raise argparse.ArgumentTypeError("Classes are invalid.")
    if invalid_classes:
        warnings.warn(f"Invalid classes: {', '.join(invalid_classes)}. Valid options are: {', '.join(options.keys())}. Continuing with subset of valid classes.")
    
    return class_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Football Match Analysis using Computer Vision.")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Video source: '0' for default webcam, '1' for second camera, path to video file, or stream URL (http://..., rtsp://...)"
    )
    parser.add_argument(
        "--tracks", 
        nargs="+", 
        type=str, 
        default=["players", "ball", "referees", "stats"], 
        help="Select objects to track and visualize: players, goalkeepers, referees, ball, stats"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose model output and logging"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame for YOLO (1=all frames, 2=every other frame, etc.)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 half precision (CUDA only)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO input image size (416=fast, 640=balanced, 1280=accurate)"
    )
    parser.add_argument(
        "--nvdec",
        action="store_true",
        help="Enable NVDEC hardware decode for RTSP streams (NVIDIA GPU only)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best.pt",
        help="Path to model file (.pt, .engine, .onnx)"
    )
    parser.add_argument(
        "--engine",
        action="store_true",
        help="Use TensorRT engine if available (models/best.engine)"
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use ONNX model if available (models/best.onnx)"
    )
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Enable broadcast streaming to broadcast_server.py (run: python scripts/broadcast_server.py)"
    )

    args = parser.parse_args()
    
    # Parse source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    classes = _classes(args.tracks)
    
    # Determine model path
    if args.model != "models/best.pt":
        model_path = args.model
    elif args.engine and os.path.exists("models/best.engine"):
        model_path = "models/best.engine"
    elif args.onnx and os.path.exists("models/best.onnx"):
        model_path = "models/best.onnx"
    else:
        model_path = "models/best.pt"

    print(f"Starting analysis with source: {source}")
    print(f"Tracking: {', '.join(args.tracks)}")
    if args.frame_skip > 1:
        print(f"Frame skip: {args.frame_skip} (processing every {args.frame_skip}th frame)")
    print(f"YOLO imgsz: {args.imgsz}, fp16: {args.fp16}, nvdec: {args.nvdec}, model: {model_path}, broadcast: {args.broadcast}")

    process_realtime(source, classes, args.verbose, args.frame_skip, args.fp16, args.imgsz, args.nvdec, model_path=model_path, broadcast=args.broadcast)
