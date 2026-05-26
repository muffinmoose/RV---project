# detection/hand_detector.py
# MediaPipe-based hand detector za 9HPT kinematično analizo
#
# Spremembe:
#   - Dodан index_mcp (landmark 5) — členek kazalca na dlani
#     Bolj stabilen od index_tip za trajektorijo (manj skače)
#   - Ostalo nespremenjeno: aktivacija preko STORAGE_BBOX_PX,
#     tracking po handedness label

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
    Rezultat detekcije aktivne roke v enem frameu.
    Vse None če aktivna roka ni najdena.
    index_mcp = landmark 5 (MCP členek kazalca) — za trajektorijo
    """
    wrist:         Optional[np.ndarray]   # landmark 0
    thumb_tip:     Optional[np.ndarray]   # landmark 4
    index_tip:     Optional[np.ndarray]   # landmark 8
    index_mcp:     Optional[np.ndarray]   # landmark 5 — členek kazalca na dlani
    all_landmarks: Optional[object]


class HandDetector:
    """
    Stateful hand detector.
    Čaka na HoleTracker baseline pred aktivacijo.
    Zaklene se na prvo roko ki vstopi v storage bbox.
    Sledi po handedness label — robustno proti MediaPipe index swapom.
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

        self._active_hand_idx   = None   # fallback index
        self._active_hand_label = None   # "Left" ali "Right" — primarni tracking
        self._activated         = False  # True ko je aktivna roka zaklenjena

    def process(self, frame: np.ndarray,
                baseline_ready: bool = False) -> HandDetection:
        """
        Požene MediaPipe na enem BGR frameu.

        Args:
            frame:          undistorted BGR frame
            baseline_ready: True ko HoleTracker nastavi baseline.
                            Pred tem aktivacija ni možna.

        Returns:
            HandDetection z koordinatami aktivne roke, ali vse-None.
        """
        h, w = frame.shape[:2]
        sx1, sy1, sx2, sy2 = STORAGE_BBOX_PX

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        # Prazen rezultat za None detekcijo
        empty = HandDetection(wrist=None, thumb_tip=None,
                              index_tip=None, index_mcp=None,
                              all_landmarks=None)

        if not results.multi_hand_landmarks:
            return empty

        n_hands = len(results.multi_hand_landmarks)

        def to_px(lm, idx: int) -> np.ndarray:
            """Normalizirane koordinate → pixel."""
            return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)

        def in_bbox(lm, idx, x1, y1, x2, y2) -> bool:
            """True če je landmark idx znotraj bbox."""
            lx = lm[idx].x * w
            ly = lm[idx].y * h
            return x1 <= lx <= x2 and y1 <= ly <= y2

        def make_detection(hand_lm) -> HandDetection:
            """Zgradi HandDetection iz MediaPipe landmarks objekta."""
            lm = hand_lm.landmark
            return HandDetection(
                wrist=     to_px(lm, LANDMARKS_OF_INTEREST["wrist"]),
                thumb_tip= to_px(lm, LANDMARKS_OF_INTEREST["thumb_tip"]),
                index_tip= to_px(lm, LANDMARKS_OF_INTEREST["index_tip"]),
                index_mcp= to_px(lm, 5),   # MCP členek kazalca — stabilnejši za trajektorijo
                all_landmarks=hand_lm,
            )

        # ── Že aktivirano — najdi isto roko po labelu ────────────────────────
        if self._activated:
            if self._active_hand_label and results.multi_handedness:
                for i, handedness in enumerate(results.multi_handedness):
                    if handedness.classification[0].label == self._active_hand_label:
                        if i < n_hands:
                            return make_detection(results.multi_hand_landmarks[i])
            elif self._active_hand_idx is not None and self._active_hand_idx < n_hands:
                return make_detection(
                    results.multi_hand_landmarks[self._active_hand_idx])
            return empty

        # ── Ni aktivirano — čakaj na baseline ───────────────────────────────
        if not baseline_ready:
            return empty

        # ── Baseline ready — išči roko v storage bbox ───────────────────────
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            lm = hand_lm.landmark
            index_in = in_bbox(lm, LANDMARKS_OF_INTEREST["index_tip"],
                               sx1, sy1, sx2, sy2)
            thumb_in = in_bbox(lm, LANDMARKS_OF_INTEREST["thumb_tip"],
                               sx1, sy1, sx2, sy2)

            if index_in or thumb_in:
                if results.multi_handedness:
                    self._active_hand_label = (
                        results.multi_handedness[i].classification[0].label)
                self._active_hand_idx = i
                self._activated       = True
                print(f"[HandDetector] Active hand locked: "
                      f"index={i}, label={self._active_hand_label or '?'}")
                return make_detection(hand_lm)

        return empty

    def draw(self, frame: np.ndarray, detection: HandDetection) -> np.ndarray:
        """Nariše skeleton + pike landmarkov na frame."""
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

        # Označi ključne točke
        for name, coords in [
            ("W",   detection.wrist),
            ("TH",  detection.thumb_tip),
            ("IX",  detection.index_tip),
            ("MCP", detection.index_mcp),
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