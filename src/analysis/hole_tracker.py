# src/analysis/hole_tracker.py
# Detects filled/empty holes in the 9HPT board.
# Holes are detected ONCE on the first frame, then tracked by brightness.
# Blue circle = empty, Green circle = filled

import cv2
import numpy as np
from config import (
    LEFT_HOLES_PX, RIGHT_HOLES_PX,
    HOLE_RADIUS_PX, HOLE_SPACING_PX,
)

SAMPLE_RADIUS = 6   # px radius za vzorčenje barve luknjice


class HoleTracker:
    """
    Tracks filled/empty state of all 18 holes.

    Usage:
        ht = HoleTracker()
        ht.initialize(first_frame)          # enkrat na začetku
        filled = ht.update(frame)           # vsak frame
        ht.draw(frame)                      # nariše circles na frame
    """

    def __init__(self):
        self.holes = []          # list of {px, py, side, idx, filled, baseline}
        self._initialized = False
        self.filled_count = 0

    def initialize(self, frame: np.ndarray):
        """
        Called once on first frame.
        Samples baseline brightness for each hole.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.holes = []

        all_holes = (
            [("L", i, px) for i, px in enumerate(LEFT_HOLES_PX)] +
            [("R", i, px) for i, px in enumerate(RIGHT_HOLES_PX)]
        )

        for side, idx, (x, y) in all_holes:
            brightness = self._sample_brightness(gray, x, y)
            self.holes.append({
                "x":         x,
                "y":         y,
                "side":      side,
                "idx":       idx,
                "filled":    False,
                "baseline":  brightness,   # brightness when empty (bright/white)
            })

        self._initialized = True
        print(f"[HoleTracker] Initialized {len(self.holes)} holes")
        baselines = [h["baseline"] for h in self.holes]
        print(f"[HoleTracker] Baseline brightness: min={min(baselines):.0f} max={max(baselines):.0f}")

    def update(self, frame: np.ndarray) -> int:
        """
        Update filled/empty state for each hole.
        Returns number of filled holes.
        """
        if not self._initialized:
            return 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for h in self.holes:
            brightness = self._sample_brightness(gray, h["x"], h["y"])
            # Hole is filled if brightness dropped significantly from baseline
            threshold = h["baseline"] * 0.65
            h["filled"] = brightness < threshold

        self.filled_count = sum(1 for h in self.holes if h["filled"])
        return self.filled_count

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw circles on holes — blue=empty, green=filled."""
        if not self._initialized:
            return frame

        r = int(HOLE_RADIUS_PX)
        for h in self.holes:
            x, y = h["x"], h["y"]
            if h["filled"]:
                color     = (0, 255, 0)    # green
                thickness = -1             # filled
            else:
                color     = (255, 100, 0)  # blue
                thickness = 2              # outline

            cv2.circle(frame, (x, y), r, color, thickness)

        return frame

    def get_state(self) -> list:
        """Returns list of (side, idx, filled) for all holes."""
        return [(h["side"], h["idx"], h["filled"]) for h in self.holes]

    @staticmethod
    def _sample_brightness(gray: np.ndarray, x: int, y: int) -> float:
        """Average brightness in a small region around (x, y)."""
        h, w = gray.shape
        x1 = max(0, x - SAMPLE_RADIUS)
        x2 = min(w, x + SAMPLE_RADIUS)
        y1 = max(0, y - SAMPLE_RADIUS)
        y2 = min(h, y + SAMPLE_RADIUS)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi))