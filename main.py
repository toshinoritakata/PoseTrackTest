"""Entry point. Adjust Config fields or pass CLI args as needed."""

import argparse
from pipeline import Pipeline, Config


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="Camera index (0) or path to video file or YouTube URL")
    p.add_argument("--yolo-weights", default="yolo11s.pt")
    p.add_argument("--pose-onnx", default=None,
                   help="RTMPose ONNX URL or local path (default: RTMPose-m)")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--output", default=None, help="Path to save output video")
    p.add_argument("--smooth-beta", type=float, default=0.01,
                   help="One-Euro beta: larger = more responsive on fast motion")

    # Face ID
    p.add_argument("--face", action="store_true",
                   help="Enable face-based target tracking")
    p.add_argument("--target-photo", default=None,
                   help="Path to target person's photo for face registration")
    p.add_argument("--face-thresh", type=float, default=0.6,
                   help="Cosine similarity threshold for face matching (default: 0.6)")

    args = p.parse_args()

    return Config(
        source=args.source,
        yolo_weights=args.yolo_weights,
        pose_onnx=args.pose_onnx,
        show=not args.no_show,
        output_path=args.output,
        smooth_beta=args.smooth_beta,
        face_enabled=args.face,
        face_target_photo=args.target_photo,
        face_similarity_thresh=args.face_thresh,
    )


if __name__ == "__main__":
    cfg = parse_args()
    Pipeline(cfg).run()
