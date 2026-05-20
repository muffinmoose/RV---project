# src/utils/visualizer.py
# Renders minimal overlay onto video frames and writes to output .mp4
# Layout: full video frame + top-left overlay (patient, frame, pegs)

import cv2
import numpy as np
from pathlib import Path

from config import LANDMARK_COLOR, SKELETON_COLOR
from analysis.phase_detector import Phase, PHASE_COLORS

FONT     = cv2.FONT_HERSHEY_SIMPLEX
ACCENT   = (0, 200, 255)
WHITE    = (255, 255, 255)
BG_COLOR = (20, 20, 20)


class Visualizer:
    def __init__(self, frame_width: int, frame_height: int,
                 fps: float, output_path: str):

        self.fw = frame_width
        self.fh = frame_height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height))

        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at {output_path}")

        self.output_path = output_path
        print(f"[Visualizer] Writing to {output_path}")

    def write_frame(self, frame, states, phase, peg_count,
                    show_landmarks, detection, detector,
                    hole_tracker=None,
                    patient_id="", frame_idx=0, events=[]):

        out = frame.copy()

        if show_landmarks and detection is not None:
            out = detector.draw(out, detection)

        # Hole circles
        if hole_tracker is not None:
            hole_tracker.draw(out)

        # Phase bar top edge
        color = PHASE_COLORS.get(phase, (100, 100, 100))
        cv2.rectangle(out, (0, 0), (self.fw, 5), color, -1)

        # Overlay box top-left
        lines = [
            (f"Patient: {patient_id}", WHITE),
            (f"Frame:   {frame_idx}",  WHITE),
            (f"Pegs:    {peg_count}/9", ACCENT),
        ]
        self._draw_overlay(out, lines)

        self.writer.write(out)

    def _draw_overlay(self, frame, lines):
        x0, y0 = 10, 10
        lh, pad = 22, 8
        box_h = pad * 2 + lh * len(lines)
        box_w = 260

        roi = frame[y0:y0+box_h, x0:x0+box_w]
        bg  = np.full_like(roi, BG_COLOR)
        cv2.addWeighted(bg, 0.6, roi, 0.4, 0, roi)
        frame[y0:y0+box_h, x0:x0+box_w] = roi

        for i, (txt, color) in enumerate(lines):
            y = y0 + pad + lh * i + 14
            cv2.putText(frame, txt, (x0 + pad, y),
                        FONT, 0.5, color, 1, cv2.LINE_AA)

    def close(self):
        self.writer.release()
        print(f"[Visualizer] Saved → {self.output_path}")

    def __enter__(self): return self
    def __exit__(self, *_): self.close()