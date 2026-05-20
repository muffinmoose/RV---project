# detection/hand_detector.py
# MediaPipe-based hand detector for 9HPT kinematic analysis

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import (
    MP_MAX_HANDS,
    MP_DETECTION_CONFIDENCE,
    MP_TRACKING_CONFIDENCE,
    LANDMARKS_OF_INTEREST,
    LANDMARK_COLOR,
    LANDMARK_RADIUS,
    SKELETON_COLOR,
)


@dataclass
class HandDetection:
    """
    Result of detecting a hand in one frame.
    All coordinates are in pixels (float), relative to the original frame size.
    None means MediaPipe did not detect a hand this frame →  will interpolate.
    """
    wrist:     Optional[np.ndarray]   # shape (2,) → [x, y]
    thumb_tip: Optional[np.ndarray]
    index_tip: Optional[np.ndarray]
    all_landmarks: Optional[list]     # full 21-point list, for drawing


class HandDetector:
    """
    Wraps MediaPipe Hands for single-hand, top-down 9HPT video.

    Usage:
        detector = HandDetector()
        for frame in video:
            detection = detector.process(frame)
            if detection.wrist is not None:
                x, y = detection.wrist
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,          # video mode → uses tracking
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )

    def process(self, frame: np.ndarray) -> HandDetection:
        """
        Run MediaPipe on one BGR frame.
        Returns HandDetection with pixel coords, or all-None if no hand found.
        """
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # small perf gain
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return HandDetection(
                wrist=None,
                thumb_tip=None,
                index_tip=None,
                all_landmarks=None,
            )

        # Take the first (and ideally only) detected hand
        hand_lm = results.multi_hand_landmarks[0]
        lm = hand_lm.landmark   # list of 21 NormalizedLandmark (x,y,z in [0,1])

        def to_px(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        return HandDetection(
            wrist=     to_px(LANDMARKS_OF_INTEREST["wrist"]),
            thumb_tip= to_px(LANDMARKS_OF_INTEREST["thumb_tip"]),
            index_tip= to_px(LANDMARKS_OF_INTEREST["index_tip"]),
            all_landmarks=hand_lm,
        )

    def draw(self, frame: np.ndarray, detection: HandDetection) -> np.ndarray:
        """
        Draw skeleton + landmark dots on frame (in-place copy).
        Call only when landmarks toggle is ON.
        """
        if detection.all_landmarks is None:
            return frame

        out = frame.copy()

        # Full MediaPipe skeleton (connections)
        self._mp_draw.draw_landmarks(
            out,
            detection.all_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self._mp_draw.DrawingSpec(
                color=LANDMARK_COLOR,
                thickness=1,
                circle_radius=LANDMARK_RADIUS,
            ),
            connection_drawing_spec=self._mp_draw.DrawingSpec(
                color=SKELETON_COLOR,
                thickness=1,
            ),
        )

        # Extra highlight on our three key points
        h, w = frame.shape[:2]
        for name, coords in [
            ("W",  detection.wrist),
            ("TH", detection.thumb_tip),
            ("IX", detection.index_tip),
        ]:
            if coords is not None:
                x, y = int(coords[0]), int(coords[1])
                cv2.circle(out, (x, y), LANDMARK_RADIUS + 3, (0, 200, 255), 2)
                cv2.putText(out, name, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        return out

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()