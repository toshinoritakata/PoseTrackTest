"""
Multi-person tracking pipeline.

  YOLOv11-s  →  BoT-SORT  →  RTMPose-m  →  Temporal Smoothing
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from smoother import SmootherRegistry
from visualizer import draw_results, draw_fps

_RTMPOSE_M_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
)
_N_KEYPOINTS = 17


@dataclass
class Config:
    # --- Detection ---
    yolo_weights: str = "yolo11s.pt"
    yolo_device: str = "mps"         # "mps" | "cpu" | "cuda"
    yolo_conf: float = 0.4
    yolo_iou: float = 0.45
    yolo_imgsz: int = 416            # 640→416: 検出速度 +20〜30%

    # --- Tracking ---
    tracker_with_reid: bool = True
    # ReID モデル名またはローカル .pt/.onnx パス
    # 精度重視: osnet_x1_0_msmt17.pt、軽量: osnet_x0_25_msmt17.pt
    reid_model: str = "osnet_x1_0_msmt17.pt"
    reid_device: str = "mps"

    # --- Pose ---
    pose_onnx: str | None = None     # None → _RTMPOSE_M_URL
    pose_input_size: tuple = (192, 256)   # (W, H) for RTMPose-m
    pose_backend: str = "onnxruntime"
    pose_device: str = "mps"        # "mps" → CoreML EP（CPU比 2〜3×高速）
    pose_stride: int = 2            # RTMPose を N フレームに1回だけ実行

    # --- Smoothing ---
    smooth_freq: float = 30.0
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 0.01       # 大きくするほど速い動きへの追従が向上

    # --- I/O ---
    source: int | str = 0
    show: bool = True
    output_path: str | None = None

    def __post_init__(self) -> None:
        if self.pose_onnx is None:
            self.pose_onnx = _RTMPOSE_M_URL


def _resolve_source(source: int | str) -> int | str:
    """Coerce string camera index to int; resolve YouTube URL to direct stream."""
    if not isinstance(source, str):
        return source
    try:
        return int(source)
    except ValueError:
        pass
    if "youtube.com" not in source and "youtu.be" not in source:
        return source

    import yt_dlp  # lazy import — only needed for YouTube sources

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source, download=False)
        url = info.get("url") or info["requested_formats"][0]["url"]
    return url


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._detector = self._build_detector()
        self._tracker = self._build_tracker()
        self._pose = self._build_pose()
        self._smoother = SmootherRegistry(
            freq=cfg.smooth_freq,
            min_cutoff=cfg.smooth_min_cutoff,
            beta=cfg.smooth_beta,
        )
        self._frame_idx: int = 0
        self._pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _build_detector(self):
        from ultralytics import YOLO
        return YOLO(self.cfg.yolo_weights)

    def _build_tracker(self):
        from boxmot.trackers.botsort.botsort import BotSort
        reid_model = None
        if self.cfg.tracker_with_reid:
            from boxmot.reid.core.reid import ReID
            from boxmot.utils import WEIGHTS
            reid = ReID(path=WEIGHTS / self.cfg.reid_model, device=self.cfg.reid_device)
            reid_model = reid.model
        return BotSort(reid_model=reid_model, with_reid=self.cfg.tracker_with_reid)

    def _build_pose(self):
        from rtmlib import RTMPose
        return RTMPose(
            self.cfg.pose_onnx,
            model_input_size=self.cfg.pose_input_size,
            backend=self.cfg.pose_backend,
            device=self.cfg.pose_device,
        )

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Return detections as float32 array of shape (N, 6): [x1,y1,x2,y2,conf,cls]."""
        results = self._detector.predict(
            frame,
            classes=[0],
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            imgsz=self.cfg.yolo_imgsz,
            device=self.cfg.yolo_device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return boxes.data.cpu().numpy().astype(np.float32)

    def _track(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Return tracks array of shape (M, 8): [x1,y1,x2,y2,track_id,conf,cls,det_ind].

        検出なしでも update() を呼ぶことで Kalman フィルタが内部状態を更新し、
        消えたトラックを正しくエージアウトできる。
        """
        tracks = self._tracker.update(dets, frame)
        if tracks is None or len(tracks) == 0:
            return np.empty((0, 8), dtype=np.float32)
        return np.asarray(tracks, dtype=np.float32)

    def _estimate_pose(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        track_ids: list[int],
        active_ids: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run RTMPose on tracked bboxes with stride-based skipping.

        pose_stride フレームに1回だけ推論を実行し、それ以外のフレームでは
        前回のキーポイントをキャッシュから返す。スムーザーが補間するため
        視覚的な劣化はほぼ出ない。

        Returns:
            keypoints : (M, _N_KEYPOINTS, 2)
            scores    : (M, _N_KEYPOINTS)
        """
        if self._frame_idx % self.cfg.pose_stride == 0:
            keypoints, scores = self._pose(frame, bboxes=bboxes.tolist())
            for i, tid in enumerate(track_ids):
                self._pose_cache[tid] = (keypoints[i].copy(), scores[i].copy())
        else:
            keypoints = np.zeros((len(track_ids), _N_KEYPOINTS, 2), dtype=np.float32)
            scores    = np.zeros((len(track_ids), _N_KEYPOINTS),    dtype=np.float32)
            for i, tid in enumerate(track_ids):
                if tid in self._pose_cache:
                    keypoints[i], scores[i] = self._pose_cache[tid]

        for tid in set(self._pose_cache) - active_ids:
            del self._pose_cache[tid]

        return keypoints, scores

    def _smooth(
        self,
        track_ids: list[int],
        keypoints: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Apply per-track One-Euro smoothing and return (M, _N_KEYPOINTS, 3) kps+conf."""
        kps_with_conf = np.concatenate([keypoints, scores[:, :, None]], axis=-1)
        smoothed = np.empty_like(kps_with_conf)
        for i, tid in enumerate(track_ids):
            smoothed[i] = self._smoother.update(tid, kps_with_conf[i])
        return smoothed

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Run full pipeline on one frame.

        Returns:
            List of dicts, one per tracked person:
            {
                "track_id"  : int,
                "bbox"      : (x1, y1, x2, y2),
                "keypoints" : np.ndarray (_N_KEYPOINTS, 3),  # x, y, conf
            }
        """
        self._frame_idx += 1

        dets   = self._detect(frame)
        tracks = self._track(dets, frame)

        if len(tracks) == 0:
            self._pose_cache.clear()
            self._smoother.cleanup(set())
            return []

        track_ids  = tracks[:, 4].astype(int).tolist()
        bboxes     = tracks[:, :4]
        active_ids = set(track_ids)

        keypoints, scores = self._estimate_pose(frame, bboxes, track_ids, active_ids)
        smoothed = self._smooth(track_ids, keypoints, scores)
        self._smoother.cleanup(active_ids)

        return [
            {"track_id": tid, "bbox": bboxes[i], "keypoints": smoothed[i]}
            for i, tid in enumerate(track_ids)
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(_resolve_source(self.cfg.source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.cfg.source}")

        writer = None
        if self.cfg.output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.cfg.output_path, fourcc, fps, (w, h))

        t_prev = time.perf_counter()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.process_frame(frame)

                t_now = time.perf_counter()
                fps_live = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                vis = draw_results(frame, results)
                vis = draw_fps(vis, fps_live)

                if self.cfg.show:
                    cv2.imshow("Multi-person Tracking", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if writer is not None:
                    writer.write(vis)

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
