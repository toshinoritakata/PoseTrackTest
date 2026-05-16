"""Entry point. Adjust Config fields or pass CLI args as needed."""

import argparse
from pipeline import Pipeline, Config


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0,
                   help="Camera index (0) or path to video file")
    p.add_argument("--yolo-weights", default="yolo11s.pt")
    p.add_argument("--pose-onnx", default=None,
                   help="RTMPose ONNX URL or local path (default: RTMPose-m)")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--output", default=None, help="Path to save output video")
    p.add_argument("--smooth-beta", type=float, default=0.01,
                   help="One-Euro beta: larger = more responsive on fast motion")
    args = p.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string path

    kwargs = dict(
        source=source,
        yolo_weights=args.yolo_weights,
        show=not args.no_show,
        output_path=args.output,
        smooth_beta=args.smooth_beta,
    )
    if args.pose_onnx is not None:
        kwargs["pose_onnx"] = args.pose_onnx
    return Config(**kwargs)


if __name__ == "__main__":
    cfg = parse_args()
    Pipeline(cfg).run()
