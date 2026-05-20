# utils/visualizer.py
# OpenCV-based UI
# Layout: 1/3 left = stats panel | 2/3 right = video feed
# Keyboard: L = toggle landmarks, Q = quit, SPACE = pause

import cv2
import numpy as np
from typing import Optional

from config import STATS_PANEL_RATIO, LANDMARK_COLOR, SKELETON_COLOR
from analysis.phase_detector import Phase, PHASE_COLORS


# ── Layout constants ──────────────────────────────────────────────────────────

PANEL_BG      = (30, 30, 30)      # dark grey background
TEXT_COLOR    = (220, 220, 220)   # light grey text
ACCENT_COLOR  = (0, 200, 255)     # yellow accent
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL    = 0.45
FONT_MEDIUM   = 0.55
FONT_LARGE    = 0.75
LINE_HEIGHT   = 22                # px between text lines


class Visualizer:
    """
    Builds and displays the composite UI frame each tick.

    Usage:
        viz = Visualizer(frame_width=640, frame_height=480)
        key = viz.show(
            frame      = video_frame,
            states     = kinematic_states,
            phase      = current_phase,
            peg_count  = detector.peg_count,
            show_landmarks = True,
            detection  = hand_detection,
            detector   = hand_detector,
        )
        if key == ord('q'):
            break
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.fh = frame_height
        self.fw = frame_width

        # Panel widths
        self.panel_w = int(frame_width * STATS_PANEL_RATIO)
        self.video_w = frame_width - self.panel_w

        # Total window size
        self.win_w = self.panel_w + self.video_w
        self.win_h = frame_height

        self.window_name = "9HPT Analysis"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)

    # ── Main render ───────────────────────────────────────────────────────────

    def show(
        self,
        frame:          np.ndarray,
        states:         dict,
        phase:          Phase,
        peg_count:      int,
        show_landmarks: bool,
        detection,                    # HandDetection
        detector,                     # HandDetector (for draw())
        patient_id:     str = "",
        frame_idx:      int = 0,
        events:         list = [],
    ) -> int:
        """
        Render one UI frame and display it.

        Returns the key pressed (or -1 if none).
        """
        # Optionally draw landmarks on video frame
        video_frame = frame.copy()
        if show_landmarks and detection is not None:
            video_frame = detector.draw(video_frame, detection)

        # Draw phase color bar at top of video
        video_frame = self._draw_phase_bar(video_frame, phase)

        # Resize video to fit right panel
        video_resized = cv2.resize(video_frame, (self.video_w, self.win_h))

        # Build stats panel
        panel = self._build_panel(
            states, phase, peg_count, patient_id, frame_idx, events)

        # Combine side by side
        composite = np.hstack([panel, video_resized])

        cv2.imshow(self.window_name, composite)
        return cv2.waitKey(1) & 0xFF

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _build_panel(
        self,
        states:     dict,
        phase:      Phase,
        peg_count:  int,
        patient_id: str,
        frame_idx:  int,
        events:     list,
    ) -> np.ndarray:

        panel = np.full((self.win_h, self.panel_w, 3), PANEL_BG, dtype=np.uint8)
        y = 20

        def text(s, color=TEXT_COLOR, scale=FONT_SMALL, bold=False):
            nonlocal y
            thickness = 2 if bold else 1
            cv2.putText(panel, s, (10, y), FONT, scale, color, thickness,
                        cv2.LINE_AA)
            y += LINE_HEIGHT

        def divider():
            nonlocal y
            cv2.line(panel, (10, y), (self.panel_w - 10, y),
                     (70, 70, 70), 1)
            y += 8

        # ── Header ────────────────────────────────────────────────────────────
        text("9HPT Analysis", ACCENT_COLOR, FONT_LARGE, bold=True)
        if patient_id:
            text(f"Patient: {patient_id}", scale=FONT_SMALL)
        text(f"Frame:   {frame_idx}", scale=FONT_SMALL)
        divider()

        # ── Current phase ─────────────────────────────────────────────────────
        phase_color = PHASE_COLORS.get(phase, TEXT_COLOR)
        text("PHASE", ACCENT_COLOR, FONT_SMALL, bold=True)
        text(phase.name, phase_color, FONT_MEDIUM, bold=True)
        y += 4
        divider()

        # ── Peg counter ───────────────────────────────────────────────────────
        text("PEGS DONE", ACCENT_COLOR, FONT_SMALL, bold=True)
        text(f"{peg_count} / 9", scale=FONT_MEDIUM, bold=True)
        divider()

        # ── Kinematics ────────────────────────────────────────────────────────
        text("KINEMATICS (wrist)", ACCENT_COLOR, FONT_SMALL, bold=True)
        wrist = states.get("wrist") if states else None
        if wrist:
            vel = wrist.velocity
            acc = wrist.acceleration
            pl  = wrist.path_length
            text(f"Speed:  {vel:.1f} mm/s"  if vel is not None else "Speed:  --")
            text(f"Accel:  {acc:.1f} mm/s2" if acc is not None else "Accel:  --")
            text(f"Path:   {pl:.1f} mm")
        else:
            text("No hand detected")
        divider()

        # ── Pinch distance ────────────────────────────────────────────────────
        text("PINCH", ACCENT_COLOR, FONT_SMALL, bold=True)
        if states and states.get("thumb_tip") and states.get("index_tip"):
            thumb = states["thumb_tip"].position
            index = states["index_tip"].position
            if thumb is not None and index is not None:
                pinch = float(np.linalg.norm(thumb - index))
                text(f"{pinch:.1f} mm")
            else:
                text("--")
        else:
            text("--")
        divider()

        # ── Last 3 completed pegs ─────────────────────────────────────────────
        text("LAST PEGS", ACCENT_COLOR, FONT_SMALL, bold=True)
        recent = events[-3:] if events else []
        if recent:
            for ev in reversed(recent):
                text(f"Peg {ev.peg_number}: {ev.duration_s:.2f}s")
        else:
            text("none yet")
        divider()

        # ── Controls reminder ─────────────────────────────────────────────────
        y = self.win_h - 60
        text("[L] landmarks", scale=0.38)
        text("[SPACE] pause", scale=0.38)
        text("[Q] quit",      scale=0.38)

        return panel

    # ── Phase color bar ───────────────────────────────────────────────────────

    @staticmethod
    def _draw_phase_bar(frame: np.ndarray, phase: Phase) -> np.ndarray:
        """Draw a colored bar at the top of the video frame showing current phase."""
        color = PHASE_COLORS.get(phase, (100, 100, 100))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 6), color, -1)
        return frame

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()