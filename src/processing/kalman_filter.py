# processing/kalman_filter.py
# Kalman filter for smoothing 2D landmark trajectories (x, y)
# One KalmanTracker instance per landmark (wrist, thumb, index)

import numpy as np
from typing import Optional

from config import KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE, FPS


class KalmanTracker:
    """
    2D Kalman filter for tracking a single landmark point.

    State vector: [x, y, vx, vy]  (position + velocity)
    Measurement:  [x, y]           (pixel coordinates from MediaPipe)

    When MediaPipe returns None (hand not detected), call update(None)
    → the filter predicts forward using last known velocity.
    This gives smooth interpolation across dropped frames.

    Usage:
        tracker = KalmanTracker()
        for frame in video:
            detection = detector.process(frame)
            smooth_xy = tracker.update(detection.wrist)   # np.ndarray [x, y]
    """

    def __init__(self, fps: float = FPS):
        dt = 1.0 / fps   # time step between frames

        # ── State transition matrix F (constant velocity model) ──
        # x'  = x  + vx*dt
        # y'  = y  + vy*dt
        # vx' = vx
        # vy' = vy
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        # ── Measurement matrix H (we observe x, y only) ──
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # ── Process noise covariance Q ──
        # How much we trust the motion model.
        # Higher → filter reacts faster but is noisier.
        q = KALMAN_PROCESS_NOISE
        self.Q = np.eye(4, dtype=np.float32) * q

        # ── Measurement noise covariance R ──
        # How much we trust MediaPipe detections.
        # Higher → smoother but more lag.
        r = KALMAN_MEASUREMENT_NOISE
        self.R = np.eye(2, dtype=np.float32) * r

        # ── Initial state & covariance ──
        self.x = np.zeros((4, 1), dtype=np.float32)   # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 1.0     # uncertainty (start high)

        self._initialized = False

    # ──────────────────────────────────────────────────────────────────────────

    def update(self, measurement: Optional[np.ndarray]) -> np.ndarray:
        """
        One Kalman cycle: predict → (optionally) correct.

        Args:
            measurement: np.ndarray shape (2,) with [x, y] in pixels,
                         or None if MediaPipe had no detection this frame.

        Returns:
            np.ndarray shape (2,) — smoothed [x, y] estimate.
        """
        if measurement is not None and not self._initialized:
            # Seed the filter with the first real detection
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self._initialized = True

        if not self._initialized:
            # No detection yet at all — return zeros
            return np.zeros(2, dtype=np.float32)

        # ── Predict ──
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ── Correct (only if we have a measurement) ──
        if measurement is not None:
            z = measurement.reshape(2, 1).astype(np.float32)

            # Innovation (residual)
            y = z - self.H @ self.x

            # Innovation covariance
            S = self.H @ self.P @ self.H.T + self.R

            # Kalman gain
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # Update state and covariance
            self.x = self.x + K @ y
            self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        return self.x[:2].flatten()   # return [x, y]

    def reset(self):
        """Reset tracker state — call between patients / videos."""
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self._initialized = False


# ── Convenience wrapper for all three landmarks ───────────────────────────────

class HandKalman:
    """
    Manages three KalmanTrackers (wrist, thumb_tip, index_tip) together.

    Usage:
        hk = HandKalman()
        smooth = hk.update(detection)   # returns dict with smoothed coords
        hk.reset()                       # between videos
    """

    def __init__(self, fps: float = FPS):
        self.trackers = {
            "wrist":     KalmanTracker(fps),
            "thumb_tip": KalmanTracker(fps),
            "index_tip": KalmanTracker(fps),
        }

    def update(self, detection) -> dict:
        """
        Args:
            detection: HandDetection from hand_detector.py

        Returns:
            dict {
                "wrist":     np.ndarray [x, y],
                "thumb_tip": np.ndarray [x, y],
                "index_tip": np.ndarray [x, y],
            }
            All values are always present (Kalman predicts even on None input).
        """
        return {
            "wrist":     self.trackers["wrist"].update(detection.wrist),
            "thumb_tip": self.trackers["thumb_tip"].update(detection.thumb_tip),
            "index_tip": self.trackers["index_tip"].update(detection.index_tip),
        }

    def reset(self):
        for t in self.trackers.values():
            t.reset()