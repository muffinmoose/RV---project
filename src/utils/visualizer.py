# utils/visualizer.py
# Renders analysis overlay onto video frames and writes to output .mp4
# No display needed — works headless on server
#
# Layout: 1/3 left = stats panel | 2/3 right = video feed
# Output: data/results/<patient_id>/<video_stem>_analyzed.mp4

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from config import STATS_PANEL_RATIO, LANDMARK_COLOR, SKELETON_COLOR
from analysis.phase_detector import Phase, PHASE_COLORS


# ── Layout constants ──────────────────────────────────────────────────────────

PANEL_BG     = (30, 30, 30)
TEXT_COLOR   = (220, 220, 220)
ACCENT_COLOR = (0, 200, 255)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL   = 0.45
FONT_MEDIUM  = 0.55
FONT_LARGE   = 0.75
LINE_HEIGHT  = 22


class Visualizer:
    """
    Renders composite frames (stats panel + video) and writes to output video.

    Usage:
        viz = Visualizer(frame_width, frame_height, fps, output_path)
        viz.write_frame(frame, states, phase, ...)
        viz.close()   # flushes and saves video
    """

    def __init__(self, frame_width: int, frame_height: int,
                 fps: float, output_path: str):

        self.fh = frame_height
        self.fw = frame_width

        self.panel_w = int(frame_width * STATS_PANEL_RATIO)
        self.video_w = frame_width - self.panel_w
        self.win_w   = self.panel_w + self.video_w
        self.win_h   = frame_height

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.win_w, self.win_h))

        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {output_path}")

        self.output_path = output_path
        print(f"[Visualizer] Writing to {output_path}")

    # ── Main render ───────────────────────────────────────────────────────────

    def write_frame(
        self,
        frame:          np.ndarray,
        states:         dict,
        phase:          Phase,
        peg_count:      int,
        show_landmarks: bool,
        detection,                  # HandDetection
        detector,                   # HandDetector
        patient_id:     str = "",
        frame_idx:      int = 0,
        events:         list = [],
    ):
        """Render one frame and write it to the output video."""

        video_frame = frame.copy()

        if show_landmarks and detection is not None:
            video_frame = detector.draw(video_frame, detection)

        video_frame = self._draw_phase_bar(video_frame, phase)

        video_resized = cv2.resize(video_frame, (self.video_w, self.win_h))

        panel = self._build_panel(
            states, phase, peg_count, patient_id, frame_idx, events)

        composite = np.hstack([panel, video_resized])
        self.writer.write(composite)

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _build_panel(self, states, phase, peg_count,
                     patient_id, frame_idx, events) -> np.ndarray:

        panel = np.full((self.win_h, self.panel_w, 3), PANEL_BG, dtype=np.uint8)
        y = 20

        def text(s, color=TEXT_COLOR, scale=FONT_SMALL, bold=False):
            nonlocal y
            cv2.putText(panel, s, (10, y), FONT, scale, color,
                        2 if bold else 1, cv2.LINE_AA)
            y += LINE_HEIGHT

        def divider():
            nonlocal y
            cv2.line(panel, (10, y), (self.panel_w - 10, y), (70, 70, 70), 1)
            y += 8

        # Header
        text("9HPT Analysis", ACCENT_COLOR, FONT_LARGE, bold=True)
        if patient_id:
            text(f"Patient: {patient_id}")
        text(f"Frame:   {frame_idx}")
        divider()

        # Phase
        phase_color = PHASE_COLORS.get(phase, TEXT_COLOR)
        text("PHASE", ACCENT_COLOR, bold=True)
        text(phase.name, phase_color, FONT_MEDIUM, bold=True)
        y += 4
        divider()

        # Peg counter
        text("PEGS DONE", ACCENT_COLOR, bold=True)
        text(f"{peg_count} / 9", FONT_MEDIUM, bold=True)
        divider()

        # Kinematics
        text("KINEMATICS (wrist)", ACCENT_COLOR, bold=True)
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

        # Pinch
        text("PINCH", ACCENT_COLOR, bold=True)
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

        # Last pegs
        text("LAST PEGS", ACCENT_COLOR, bold=True)
        recent = events[-3:] if events else []
        if recent:
            for ev in reversed(recent):
                text(f"Peg {ev.peg_number}: {ev.duration_s:.2f}s")
        else:
            text("none yet")

        return panel

    # ── Phase bar ─────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_phase_bar(frame: np.ndarray, phase: Phase) -> np.ndarray:
        color = PHASE_COLORS.get(phase, (100, 100, 100))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 6), color, -1)
        return frame

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self.writer.release()
        print(f"[Visualizer] Video saved → {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()