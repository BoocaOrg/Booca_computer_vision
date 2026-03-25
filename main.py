from typing import Union, List
import os
import argparse
import warnings
from utils import read_video, save_video, options
from trackers import Tracker
from team_assignment import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
from camera_movement import CameraMovementEstimator

def _expand_tracks(tracks: dict, sampled_indices: list, total_frames: int) -> dict:
    """Expand sampled tracks back to full frame count by holding last known values."""
    expanded = {"players": [], "referees": [], "ball": []}
    sample_ptr = 0
    for i in range(total_frames):
        if sample_ptr < len(sampled_indices) and i == sampled_indices[sample_ptr]:
            expanded["players"].append(tracks["players"][sample_ptr])
            expanded["referees"].append(tracks["referees"][sample_ptr])
            expanded["ball"].append(tracks["ball"][sample_ptr])
            sample_ptr += 1
        else:
            # Hold last known state for skipped frames
            expanded["players"].append(expanded["players"][-1] if expanded["players"] else {})
            expanded["referees"].append(expanded["referees"][-1] if expanded["referees"] else {})
            expanded["ball"].append(expanded["ball"][-1] if expanded["ball"] else {})
    return expanded


def process_video(data: Union[str, bytes], classes: List[int], verbose: bool=True, frame_skip: int=1, fp16: bool=False, imgsz: int=640, model_path: str="models/best.pt") -> None:
    frames, fps, _, _ = read_video(data, verbose)
    total_frames = len(frames)

    tracker = Tracker(model_path, classes, verbose, fp16=fp16, imgsz=imgsz)

    if frame_skip > 1:
        sampled_indices = list(range(0, total_frames, frame_skip))
        sampled_frames = frames[::frame_skip]
        if verbose:
            print(f"Frame skip: processing {len(sampled_frames)}/{total_frames} frames")
        tracks = tracker.get_object_tracks(sampled_frames)
        tracker.add_position_to_tracks(tracks)
        tracks = _expand_tracks(tracks, sampled_indices, total_frames)
    else:
        tracks = tracker.get_object_tracks(frames)
        tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(frames[0], classes, verbose)
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(frames)
    camera_movement_estimator.adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    team_assigner.get_teams(frames, tracks)

    player_assigner = PlayerBallAssigner()
    player_assigner.get_player_and_possession(tracks)

    output = tracker.draw_annotations(frames, tracks, player_assigner.ball_possession)
    output = camera_movement_estimator.draw_camera_movement(output, camera_movement_per_frame)

    save_video(output, "output/output.mp4", fps, verbose)

def _video(path: str) -> None:
    if not path.lower().endswith(".mp4"):
        raise argparse.ArgumentTypeError(f"File '{path}' is not an MP4 file.")
    
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File '{path}' does not exist.") 
    
def _classes(classes: List[str]) -> List[int]:
    class_ids = [value for key, value in options.items() if key in classes]
    
    invalid_classes = [cls for cls in classes if cls not in options.keys()]
    
    # all classes invalid, raise error
    if len(invalid_classes) == len(options):
        raise argparse.ArgumentTypeError("Classes are invalid.")

    # continue with the subset of valid classes
    if invalid_classes:
        warnings.warn(f"Invalid classes: {', '.join(invalid_classes)}. Valid options are: {', '.join(options.keys())}. Continuing with subset of valid classes.")
    
    return class_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MatchVision football analytics.")

    parser.add_argument("--video", type=str, help="Video path of the video (must be .mp4)")
    parser.add_argument("--tracks", nargs="+", type=str, help="Select the objects to visualise: players, goalkeepers, referees, ball")
    parser.add_argument("--verbose", action="store_true", help="Model output and logging")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (1=all frames)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 half precision (CUDA only)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size (416=fast, 640=balanced, 1280=accurate)")
    parser.add_argument("--engine", action="store_true", help="Use TensorRT engine if available (models/best.engine)")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model if available (models/best.onnx)")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to model file (.pt, .engine, .onnx)")

    args = parser.parse_args()
    
    if args.video and args.tracks:
        _video(args.video)
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

        process_video(args.video, classes, args.verbose, args.frame_skip, args.fp16, args.imgsz, model_path=model_path)