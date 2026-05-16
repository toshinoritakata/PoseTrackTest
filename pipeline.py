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
    # N フレーム連続マッチしたトラックを「安定」と判定し ReID をスキップ
    reid_stability_thresh: int = 5
    # 安定トラックの前回 bbox との IoU がこれ以上なら embedding を再利用
    reid_stable_iou_thresh: float = 0.5

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

    # --- Face ID ---
    face_enabled: bool = False
    # facenet-pytorch デバイス（MPS は現時点で不安定なため cpu 推奨）
    face_device: str = "cpu"
    # コサイン類似度の閾値（0〜1、高いほど厳格）
    face_similarity_thresh: float = 0.6
    # ターゲット登録用の写真パス。None の場合は起動後にクリックで登録
    face_target_photo: str | None = None
    # 顔認識をNフレームに1回実行（ターゲット確定後の定期確認）
    face_stride: int = 15

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


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between (N,4) and (M,4) xyxy arrays. Returns (N,M)."""
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (np.maximum(inter_x2 - inter_x1, 0)
             * np.maximum(inter_y2 - inter_y1, 0))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._reid = None                              # populated by _build_tracker
        self._detector = self._build_detector()
        self._tracker = self._build_tracker()          # may set self._reid
        self._pose = self._build_pose()
        self._smoother = SmootherRegistry(
            freq=cfg.smooth_freq,
            min_cutoff=cfg.smooth_min_cutoff,
            beta=cfg.smooth_beta,
        )
        self._frame_idx: int = 0
        self._pose_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # ReID 安定スキップ用ステート
        self._track_hits: dict[int, int] = {}          # track_id → 連続マッチ回数
        self._emb_cache: dict[int, np.ndarray] = {}   # track_id → 最後の embedding
        self._bbox_cache: dict[int, np.ndarray] = {}  # track_id → 最後の bbox
        # 顔認識ステート
        self._face = self._build_face()                # FaceIdentifier or None
        self._target_id: int | None = None             # 現在のターゲット track_id
        self._registering: bool = (                    # クリック登録待ち状態
            cfg.face_enabled and cfg.face_target_photo is None
        )
        self._pending_frame: np.ndarray | None = None  # マウスコールバック用
        self._pending_results: list[dict] | None = None

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
            self._reid = ReID(path=WEIGHTS / self.cfg.reid_model, device=self.cfg.reid_device)
            reid_model = self._reid.model
        return BotSort(reid_model=reid_model, with_reid=self.cfg.tracker_with_reid)

    def _build_face(self):
        if not self.cfg.face_enabled:
            return None
        print("[FaceID] Loading MTCNN + InceptionResnetV1 (VGGFace2) …")
        from face_identifier import FaceIdentifier
        fi = FaceIdentifier(
            similarity_thresh=self.cfg.face_similarity_thresh,
            device=self.cfg.face_device,
        )
        if self.cfg.face_target_photo:
            if not fi.register_from_photo(self.cfg.face_target_photo):
                raise RuntimeError(
                    f"No face detected in photo: {self.cfg.face_target_photo}"
                )
            print(f"[FaceID] Target registered from photo: {self.cfg.face_target_photo}")
        else:
            print("[FaceID] Ready — click on a person in the video window to register target")
        return fi

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

    def _compute_embeddings_selective(
        self, dets: np.ndarray, frame: np.ndarray
    ) -> np.ndarray:
        """安定トラックにマッチする検出は embedding キャッシュを再利用。

        安定トラック（hits >= reid_stability_thresh）の前回 bbox と IoU を比較し、
        閾値以上なら新規推論をスキップする。新規/不安定な検出のみ get_features() を呼ぶ。

        ゼロベクトル埋めは BoT-SORT の外観マッチングを壊すため使わない。
        キャッシュにマッチしない検出はすべて新規推論する（安定スキップで
        大半の検出は再利用されるため実際の推論数は限定的）。

        Returns:
            embs: (N, D) float32
        """
        n = len(dets)
        bboxes = dets[:, :4]

        # 安定トラック: hits が閾値以上かつキャッシュあり
        stable = {
            tid: (self._bbox_cache[tid], self._emb_cache[tid])
            for tid, hits in self._track_hits.items()
            if hits >= self.cfg.reid_stability_thresh
            and tid in self._bbox_cache
            and tid in self._emb_cache
        }

        # IoU マッチングで各検出を安定トラックに対応付け
        reuse: dict[int, np.ndarray] = {}   # det_idx → reuse embedding
        if stable:
            stable_tids = list(stable)
            stable_bboxes = np.array([stable[tid][0] for tid in stable_tids])
            iou = _bbox_iou(bboxes, stable_bboxes)   # (N, S)
            best_val = iou.max(axis=1)
            best_idx = iou.argmax(axis=1)
            for i in range(n):
                if best_val[i] >= self.cfg.reid_stable_iou_thresh:
                    reuse[i] = stable[stable_tids[best_idx[i]]][1]

        # キャッシュ再利用できない検出をすべて新規推論する
        # ゼロベクトルでの代替は ID switch の原因になるため行わない
        fresh_idx = [i for i in range(n) if i not in reuse]

        fresh_embs = (
            self._reid.model.get_features(bboxes[fresh_idx], frame)
            if fresh_idx else np.empty((0, 1), dtype=np.float32)
        )

        emb_dim = (
            fresh_embs.shape[1] if fresh_idx
            else next(iter(reuse.values())).shape[0] if reuse
            else 512  # OSNet デフォルト次元
        )
        embs = np.zeros((n, emb_dim), dtype=np.float32)
        for j, i in enumerate(fresh_idx):
            embs[i] = fresh_embs[j]
        for i, emb in reuse.items():
            embs[i] = emb
        return embs

    def _track(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Return tracks array of shape (M, 8): [x1,y1,x2,y2,track_id,conf,cls,det_ind].

        検出なしでも update() を呼ぶことで Kalman フィルタが内部状態を更新し、
        消えたトラックを正しくエージアウトできる。
        """
        embs = None
        if self._reid is not None and len(dets) > 0:
            embs = self._compute_embeddings_selective(dets, frame)

        tracks = self._tracker.update(dets, frame, embs)
        result = (
            np.asarray(tracks, dtype=np.float32)
            if tracks is not None and len(tracks) > 0
            else np.empty((0, 8), dtype=np.float32)
        )
        self._update_reid_state(result, dets, embs)
        return result

    def _update_reid_state(
        self,
        tracks: np.ndarray,
        dets: np.ndarray,
        embs: np.ndarray | None,
    ) -> None:
        """安定度カウンタ・embedding・bbox キャッシュを更新。"""
        active_ids = set(tracks[:, 4].astype(int)) if len(tracks) > 0 else set()

        # 安定度カウンタ更新（登場: +1 / 消滅: 削除）
        for tid in active_ids:
            self._track_hits[tid] = self._track_hits.get(tid, 0) + 1
        for tid in set(self._track_hits) - active_ids:
            del self._track_hits[tid]
            self._emb_cache.pop(tid, None)
            self._bbox_cache.pop(tid, None)

        if len(tracks) == 0 or embs is None:
            return

        # bbox・embedding キャッシュ更新
        # track bbox と det bbox の IoU で対応する embedding を特定
        det_bboxes = dets[:, :4]
        for track in tracks:
            tid = int(track[4])
            self._bbox_cache[tid] = track[:4].copy()
            iou = _bbox_iou(track[:4][None, :], det_bboxes)[0]  # (N,)
            best = int(iou.argmax())
            if iou[best] >= 0.3:
                self._emb_cache[tid] = embs[best].copy()

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

        # ターゲットトラックが消えた場合はリセット
        if self._target_id is not None and self._target_id not in active_ids:
            self._target_id = None

        results = [
            {
                "track_id":  tid,
                "bbox":      bboxes[i],
                "keypoints": smoothed[i],
                "is_target": tid == self._target_id,
            }
            for i, tid in enumerate(track_ids)
        ]

        # 顔認識によるターゲット同定
        if self._face is not None and self._face.registered and not self._registering:
            run_face = (
                self._target_id is None                                   # まだ未発見
                or self._frame_idx % self.cfg.face_stride == 0            # 定期確認
            )
            if run_face:
                found = self._face.find_target(
                    frame, results, preferred_id=self._target_id
                )
                if found is not None and found != self._target_id:
                    self._target_id = found
                    for r in results:
                        r["is_target"] = r["track_id"] == self._target_id

        return results

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """クリックされた bbox の人物顔をターゲットとして登録する。"""
        if event != cv2.EVENT_LBUTTONDOWN or not self._registering:
            return
        if self._pending_frame is None or self._pending_results is None:
            return
        fh, fw = self._pending_frame.shape[:2]
        for t in self._pending_results:
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            if x1 <= x <= x2 and y1 <= y <= y2:
                cx1 = max(0, x1); cy1 = max(0, y1)
                cx2 = min(fw, x2); cy2 = min(fh, y2)
                crop = self._pending_frame[cy1:cy2, cx1:cx2]
                if self._face.register_from_crop(crop):
                    self._registering = False
                    self._target_id = t["track_id"]
                    print(f"[FaceID] Target registered (ID {self._target_id})")
                else:
                    print("[FaceID] No face detected — try clicking again")
                break

    def _draw_face_overlay(self, vis: np.ndarray) -> np.ndarray:
        """顔認識の状態をフレームに重ねて表示する。"""
        if self._registering:
            msg, color = "Click on target person", (0, 255, 255)
        elif self._target_id is not None:
            msg, color = f"Target: ID {self._target_id}", (0, 255, 0)
        else:
            msg, color = "Searching for target...", (0, 165, 255)
        cv2.putText(vis, msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return vis

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

        win = "Multi-person Tracking"
        if self.cfg.show:
            cv2.namedWindow(win)
            if self._face is not None:
                cv2.setMouseCallback(win, self._on_mouse)

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
                if self._face is not None:
                    vis = self._draw_face_overlay(vis)

                # マウスコールバック用に最新フレームを保持
                self._pending_frame = frame
                self._pending_results = results

                if self.cfg.show:
                    cv2.imshow(win, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if writer is not None:
                    writer.write(vis)

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
