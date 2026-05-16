"""Face detection and recognition using MTCNN + InceptionResnetV1 (facenet-pytorch).

MTCNN   : 顔検出 + アライメント（160×160 クロップ）
ResNet  : VGGFace2 学習済み 512 次元埋め込み
類似度  : コサイン類似度（L2 正規化済みベクトルの内積）
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image


class FaceIdentifier:
    """ターゲット顔を登録し、トラック群の中から同一人物を識別する。"""

    # これ未満のピクセルサイズの顔は認識精度が低いためスキップ
    _MIN_FACE_PX = 40

    def __init__(
        self,
        similarity_thresh: float = 0.6,
        device: str = "cpu",
    ) -> None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self._device = torch.device(device)
        self._mtcnn = MTCNN(
            keep_all=False,                # 最も確信度の高い顔1つだけ返す
            min_face_size=self._MIN_FACE_PX,
            device=self._device,
        )
        self._resnet = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self._device)
        )
        self._target_emb: np.ndarray | None = None
        self._thresh = similarity_thresh

    @property
    def registered(self) -> bool:
        return self._target_emb is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """BGR 画像から顔を検出し L2 正規化済み 512 次元埋め込みを返す。
        顔が検出されない場合は None。
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face = self._mtcnn(Image.fromarray(img_rgb))   # (3,160,160) or None
        if face is None:
            return None
        with torch.no_grad():
            emb = self._resnet(face.unsqueeze(0).to(self._device))
        emb = emb.cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-6)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_from_photo(self, path: str) -> bool:
        """画像ファイルからターゲット顔を登録する。成功したら True。"""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        emb = self._embed(img)
        if emb is None:
            return False
        self._target_emb = emb
        return True

    def register_from_crop(self, crop: np.ndarray) -> bool:
        """人物 bbox クロップからターゲット顔を登録する。成功したら True。"""
        if crop.size == 0:
            return False
        emb = self._embed(crop)
        if emb is None:
            return False
        self._target_emb = emb
        return True

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    def find_target(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        preferred_id: int | None = None,
    ) -> int | None:
        """登録済みターゲット顔に最も類似するトラックの track_id を返す。

        preferred_id が指定された場合はそのトラックを先に検査し、
        閾値を超えれば即座に返す（安定ターゲットの高速パス）。

        類似度が閾値を超えるトラックが存在しない場合は None。
        """
        if not self.registered or not tracks:
            return None

        # 安定ターゲットの高速パス: 同じ track_id を先に確認
        if preferred_id is not None:
            for t in tracks:
                if t["track_id"] != preferred_id:
                    continue
                x1, y1, x2, y2 = (int(v) for v in t["bbox"])
                emb = self._embed(frame[y1:y2, x1:x2])
                if emb is not None:
                    sim = float(np.dot(emb, self._target_emb))
                    if sim >= self._thresh:
                        return preferred_id
                break  # preferred が閾値未満 → 全探索へフォールスルー

        best_tid: int | None = None
        best_sim: float = self._thresh

        for t in tracks:
            if t["track_id"] == preferred_id:
                continue   # 上で既に検査済み
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            emb = self._embed(frame[y1:y2, x1:x2])
            if emb is None:
                continue
            sim = float(np.dot(emb, self._target_emb))
            if sim > best_sim:
                best_sim = sim
                best_tid = t["track_id"]

        return best_tid
