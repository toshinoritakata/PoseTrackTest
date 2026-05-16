import cv2
import numpy as np

# COCO-17 skeleton pairs (0-indexed)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12),                          # torso
    (11, 12),                                  # hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]

_PALETTE = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
    (255, 153, 255), (153, 204, 255), (255, 102, 255), (255, 51, 255),
    (102, 178, 255), (51, 153, 255), (255, 153, 153), (255, 102, 102),
    (255, 51, 51), (153, 255, 153), (102, 255, 102), (51, 255, 51),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def draw_results(
    frame: np.ndarray,
    tracks: list[dict],
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Render bboxes, IDs, and skeletons onto frame (in-place).

    Args:
        tracks: list of dicts with keys:
            - track_id  : int
            - bbox      : (x1, y1, x2, y2)
            - keypoints : np.ndarray (17, 2) or (17, 3)
    """
    out = frame.copy()

    for t in tracks:
        tid = t["track_id"]
        color = _track_color(tid)
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID:{tid}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        kps = t.get("keypoints")
        if kps is None:
            continue

        has_conf = kps.shape[1] >= 3

        # Keypoints
        for i, kp in enumerate(kps):
            if has_conf and kp[2] < conf_threshold:
                continue
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(out, (cx, cy), 4, color, -1)

        # Skeleton
        for a, b in SKELETON:
            if has_conf and (kps[a, 2] < conf_threshold or kps[b, 2] < conf_threshold):
                continue
            p1 = (int(kps[a, 0]), int(kps[a, 1]))
            p2 = (int(kps[b, 0]), int(kps[b, 1]))
            cv2.line(out, p1, p2, color, 2)

    return out


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame
