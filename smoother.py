import math
import numpy as np


class _OneEuroFilter:
    """Single-axis One-Euro Filter."""

    def __init__(self, freq: float, min_cutoff: float, beta: float, d_cutoff: float):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_hat: float | None = None
        self._dx_hat: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

    def __call__(self, x: float) -> float:
        if self._x_hat is None:
            self._x_hat = x
            return x

        dx = (x - self._x_hat) * self.freq
        alpha_d = self._alpha(self.d_cutoff, self.freq)
        self._dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_hat

        cutoff = self.min_cutoff + self.beta * abs(self._dx_hat)
        alpha = self._alpha(cutoff, self.freq)
        self._x_hat = alpha * x + (1.0 - alpha) * self._x_hat
        return self._x_hat


class KeypointSmoother:
    """One-Euro Filter set for one person's keypoints.

    Maintains independent (x, y) filters per keypoint so each axis is
    smoothed separately, which matters when motion is axis-asymmetric.
    """

    def __init__(
        self,
        n_keypoints: int = 17,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        # beta: speed coefficient — larger → less lag on fast motion
        beta: float = 0.01,
        d_cutoff: float = 1.0,
    ):
        self._filters: list[tuple[_OneEuroFilter, _OneEuroFilter]] = [
            (
                _OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
                _OneEuroFilter(freq, min_cutoff, beta, d_cutoff),
            )
            for _ in range(n_keypoints)
        ]

    def update(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply filters to keypoints.

        Args:
            keypoints: shape (n_keypoints, 2) or (n_keypoints, 3) where
                       col 0=x, 1=y, 2=conf (conf is passed through unchanged)
        Returns:
            Smoothed keypoints, same shape as input.
        """
        out = keypoints.copy().astype(float)
        for i, (fx, fy) in enumerate(self._filters):
            out[i, 0] = fx(keypoints[i, 0])
            out[i, 1] = fy(keypoints[i, 1])
        return out


class SmootherRegistry:
    """Manages one KeypointSmoother per active track ID."""

    def __init__(self, **smoother_kwargs):
        self._kwargs = smoother_kwargs
        self._registry: dict[int, KeypointSmoother] = {}

    def update(self, track_id: int, keypoints: np.ndarray) -> np.ndarray:
        if track_id not in self._registry:
            self._registry[track_id] = KeypointSmoother(**self._kwargs)
        return self._registry[track_id].update(keypoints)

    def cleanup(self, active_ids: set[int]) -> None:
        for stale_id in set(self._registry) - active_ids:
            del self._registry[stale_id]

    def __len__(self) -> int:
        return len(self._registry)
