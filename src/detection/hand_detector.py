# detection/hand_detector.py
# MediaPipe-based hand detector for 9HPT kinematic analysis
#
# Strategy:
#   1. Detect up to MP_MAX_HANDS hands
#   2. Wait for baseline_ready signal (HoleTracker must set baseline first)
#   3. Activation: index_tip OR thumb_tip enters STORAGE_BBOX_PX
#   4. Once activated → lock onto that hand by HANDEDNESS LABEL (not index)
#      This prevents losing the hand when MediaPipe swaps hand indices between frames

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
    BOARD_HOLES_BBOX_PX,
    STORAGE_BBOX_PX,
)


@dataclass
class HandDetection:
    """
    Result of detecting the active hand in one frame.
    All None if active hand not yet found or lost this frame.
    """
    wrist:         Optional[np.ndarray]
    thumb_tip:     Optional[np.ndarray]
    index_tip:     Optional[np.ndarray]
    all_landmarks: Optional[object]


class HandDetector:
    """
    Stateful hand detector.
    Waits for HoleTracker baseline before activating.
    Locks onto first hand entering storage bbox.
    Tracks by handedness label (LEFT/RIGHT) — robust against MediaPipe index swaps.
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils

        self.hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MP_MAX_HANDS,
            min_detection_confidence=MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        )

        self._active_hand_idx   = None   # fallback index if no handedness available
        self._active_hand_label = None   # "Left" or "Right" — primary tracking key
        self._activated         = False  # True once active hand is locked

    def process(self, frame: np.ndarray,
                baseline_ready: bool = False) -> HandDetection:
        """
        Run MediaPipe on one BGR frame.

        Args:
            frame:          undistorted BGR frame
            baseline_ready: True when HoleTracker has set baseline.
                            Before this, no activation is possible.

        Returns:
            HandDetection with active hand coords, or all-None.
        """
        h, w = frame.shape[:2]
        sx1, sy1, sx2, sy2 = STORAGE_BBOX_PX

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return HandDetection(wrist=None, thumb_tip=None,
                                 index_tip=None, all_landmarks=None)

        n_hands = len(results.multi_hand_landmarks)

        def to_px(lm, idx: int) -> np.ndarray:
            """Convert normalized landmark to pixel coordinates."""
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        def in_bbox(lm, idx, x1, y1, x2, y2) -> bool:
            """True if landmark idx is inside bbox."""
            lx = lm[idx].x * w
            ly = lm[idx].y * h
            return x1 <= lx <= x2 and y1 <= ly <= y2

        def make_detection(hand_lm) -> HandDetection:
            """Build HandDetection from MediaPipe hand landmarks object."""
            lm = hand_lm.landmark
            return HandDetection(
                wrist=     to_px(lm, LANDMARKS_OF_INTEREST["wrist"]),
                thumb_tip= to_px(lm, LANDMARKS_OF_INTEREST["thumb_tip"]),
                index_tip= to_px(lm, LANDMARKS_OF_INTEREST["index_tip"]),
                all_landmarks=hand_lm,
            )

        # ── Already activated — find same hand by label ──────────────────────
        if self._activated:
            # Primary: match by handedness label (robust against index swaps)
            if self._active_hand_label and results.multi_handedness:
                for i, handedness in enumerate(results.multi_handedness):
                    if handedness.classification[0].label == self._active_hand_label:
                        if i < n_hands:
                            return make_detection(results.multi_hand_landmarks[i])

            # Fallback: use saved index if no handedness available
            elif self._active_hand_idx is not None and self._active_hand_idx < n_hands:
                return make_detection(
                    results.multi_hand_landmarks[self._active_hand_idx]
                )

            # Temporarily lost — Kalman will interpolate
            return HandDetection(wrist=None, thumb_tip=None,
                                 index_tip=None, all_landmarks=None)

        # ── Not yet activated — wait for baseline ────────────────────────────
        if not baseline_ready:
            return HandDetection(wrist=None, thumb_tip=None,
                                 index_tip=None, all_landmarks=None)

        # ── Baseline ready — look for hand entering storage bbox ─────────────
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            lm = hand_lm.landmark
            index_in = in_bbox(lm, LANDMARKS_OF_INTEREST["index_tip"],
                               sx1, sy1, sx2, sy2)
            thumb_in  = in_bbox(lm, LANDMARKS_OF_INTEREST["thumb_tip"],
                                sx1, sy1, sx2, sy2)

            if index_in or thumb_in:
                # Save handedness label for robust tracking across frames
                if results.multi_handedness:
                    self._active_hand_label = (
                        results.multi_handedness[i].classification[0].label
                    )
                self._active_hand_idx = i
                self._activated       = True
                print(f"[HandDetector] Active hand locked: "
                      f"index={i}, label={self._active_hand_label or '?'}")
                return make_detection(hand_lm)

        return HandDetection(wrist=None, thumb_tip=None,
                             index_tip=None, all_landmarks=None)

    def draw(self, frame: np.ndarray, detection: HandDetection) -> np.ndarray:
        """Draw skeleton + landmark dots on frame."""
        if detection.all_landmarks is None:
            return frame

        out = frame.copy()
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
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()