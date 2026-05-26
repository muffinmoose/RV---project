# src/analysis/hole_tracker.py
# Detects filled/empty holes in the 9HPT board.
#
# Strategy:
#   1. Record initial brightness (all LEDs on)
#   2. Wait for LED drop (negative front) — one sector turns off
#   3. Wait for comeback (positive front) — active sector turns back on
#   4. Set baseline: holes that recovered = active, stayed dark = inactive
#   5. Hysteresis: CONFIRM_FRAMES consecutive dark frames to mark filled
#   6. Group change guard: >MAX_SIMULTANEOUS changes = roka/LED event, ignore
#   7. Sector guard: any hand landmark inside sector bbox = skip that sector
#
# active_side property:
#   'L' = left sector active  → patient uses RIGHT hand
#   'R' = right sector active → patient uses LEFT hand

import cv2
import numpy as np
from config import LEFT_HOLES_PX, RIGHT_HOLES_PX, HOLE_RADIUS_PX

SAMPLE_RADIUS       = 6
STABILIZE_THRESHOLD = 3.0
STABILIZE_WINDOW    = 10
DROP_THRESHOLD      = 0.25
COMEBACK_WINDOW     = 50
FILL_THRESHOLD      = 0.65
CONFIRM_FRAMES      = 5
MAX_SIMULTANEOUS    = 2
PICK_CONFIRM_FRAMES = 8  # consecutive bright frames to confirm pin picked


class HoleTracker:
    def __init__(self):
        self.holes               = []
        self._initialized        = False
        self._frame_count        = 0
        self.filled_count        = 0
        self._baseline_set       = False
        self._left_bbox          = None
        self._right_bbox         = None
        self._initial_brightness = None
        self._drop_detected      = False
        self._drop_frame         = 0
        self._min_brightness     = None
        self._prev_mean          = None
        self._stable_count       = 0
        self.fill_order          = []   # (side, idx) tuples in fill sequence
        self.pick_order          = []   # (side, idx) tuples in pick sequence

    def initialize(self, frame: np.ndarray):
        """Called once on first frame. Registers hole positions and sector bboxes."""
        self.holes = []
        all_holes = (
            [("L", i, px) for i, px in enumerate(LEFT_HOLES_PX)] +
            [("R", i, px) for i, px in enumerate(RIGHT_HOLES_PX)]
        )
        for side, idx, (x, y) in all_holes:
            self.holes.append({
                "x": x, "y": y, "side": side, "idx": idx,
                "filled": False, "baseline": None,
                "led_off": False, "dark_count": 0, "bright_count": 0,
            })

        margin = int(HOLE_RADIUS_PX * 3)
        lx = [h["x"] for h in self.holes if h["side"] == "L"]
        ly = [h["y"] for h in self.holes if h["side"] == "L"]
        rx = [h["x"] for h in self.holes if h["side"] == "R"]
        ry = [h["y"] for h in self.holes if h["side"] == "R"]
        self._left_bbox  = (min(lx)-margin, min(ly)-margin,
                            max(lx)+margin, max(ly)+margin)
        self._right_bbox = (min(rx)-margin, min(ry)-margin,
                            max(rx)+margin, max(ry)+margin)

        self._initialized        = True
        self._frame_count        = 0
        self._baseline_set       = False
        self._drop_detected      = False
        self._drop_frame         = 0
        self._min_brightness     = None
        self._initial_brightness = None
        self._prev_mean          = None
        self._stable_count       = 0

        print(f"[HoleTracker] Initialized {len(self.holes)} holes")
        print(f"[HoleTracker] Left bbox:  {self._left_bbox}")
        print(f"[HoleTracker] Right bbox: {self._right_bbox}")

    def update(self, frame: np.ndarray, all_landmarks=None) -> int:
        """Update filled/empty state. Returns number of filled holes."""
        if not self._initialized:
            return 0

        self._frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self._baseline_set:
            self._check_stabilization(gray)
            return 0

        left_blocked  = self._sector_blocked(all_landmarks, "L")
        right_blocked = self._sector_blocked(all_landmarks, "R")

        candidates = []
        for h in self.holes:
            if h["baseline"] is None or h["led_off"]:
                continue
            if h["side"] == "L" and left_blocked:
                h["dark_count"] = 0
                continue
            if h["side"] == "R" and right_blocked:
                h["dark_count"] = 0
                continue
            brightness = self._sample_brightness(gray, h["x"], h["y"])
            is_dark    = brightness < h["baseline"] * FILL_THRESHOLD
            candidates.append((h, is_dark))

        would_change = sum(1 for h, is_dark in candidates if is_dark != h["filled"])
        if would_change > MAX_SIMULTANEOUS:
            for h, _ in candidates:
                h["dark_count"] = 0
            self.filled_count = sum(1 for h in self.holes if h["filled"])
            return self.filled_count

        placing_done = len(self.fill_order) >= 8

        for h, is_dark in candidates:
            if is_dark:
                h["dark_count"] += 1
                h["bright_count"] = 0
                key = (h["side"], h["idx"])
                if h["dark_count"] >= CONFIRM_FRAMES and not h["filled"]:
                    if key not in self.fill_order:
                        h["filled"] = True
                        self.fill_order.append(key)
            else:
                h["dark_count"] = 0
                if h["filled"] and placing_done:
                    h["bright_count"] += 1
                    if h["bright_count"] >= PICK_CONFIRM_FRAMES:
                        h["filled"] = False
                        h["bright_count"] = 0
                        self.pick_order.append((h["side"], h["idx"]))
                else:
                    h["bright_count"] = 0
                    if not placing_done:
                        h["filled"] = False

        self.filled_count = sum(1 for h in self.holes if h["filled"])
        return self.filled_count

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw circles: gray=stabilizing, blue=empty, green=filled."""
        if not self._initialized:
            return frame
        r = int(HOLE_RADIUS_PX)
        for h in self.holes:
            x, y = h["x"], h["y"]
            if not self._baseline_set:
                color, thickness = (128, 128, 128), 2
            elif h["led_off"]:
                continue
            elif h["filled"]:
                color, thickness = (0, 255, 0), -1
            else:
                color, thickness = (255, 100, 0), 2
            cv2.circle(frame, (x, y), r, color, thickness)
        return frame

    def get_state(self) -> list:
        return [(h["side"], h["idx"], h["filled"]) for h in self.holes]

    @property
    def active_side(self):
        """Majority vote — side with more active holes wins."""
        if not self._baseline_set:
            return None
        left_count  = sum(1 for h in self.holes if h["side"] == "L" and not h["led_off"])
        right_count = sum(1 for h in self.holes if h["side"] == "R" and not h["led_off"])
        if left_count > right_count:
            return "L"
        if right_count > left_count:
            return "R"
        return None

    def _check_stabilization(self, gray: np.ndarray):
        """Three-phase: initial → drop → comeback → baseline."""
        mean_b = np.mean([
            self._sample_brightness(gray, h["x"], h["y"])
            for h in self.holes
        ])

        if self._initial_brightness is None:
            self._initial_brightness = mean_b
            self._prev_mean          = mean_b
            return

        if not self._drop_detected:
            drop_ratio = (self._initial_brightness - mean_b) / self._initial_brightness
            if drop_ratio > DROP_THRESHOLD:
                self._drop_detected  = True
                self._drop_frame     = self._frame_count
                self._min_brightness = mean_b
                self._stable_count   = 0
                print(f"[HoleTracker] LED drop at frame {self._frame_count} "
                      f"({self._initial_brightness:.0f} → {mean_b:.0f})")
            self._prev_mean = mean_b
            return

        if mean_b < self._min_brightness:
            self._min_brightness = mean_b

        frames_since_drop = self._frame_count - self._drop_frame
        comeback_ratio    = (mean_b - self._min_brightness) / max(self._initial_brightness, 1)

        if comeback_ratio > 0.15 or frames_since_drop > COMEBACK_WINDOW:
            change = abs(mean_b - self._prev_mean)
            if change < STABILIZE_THRESHOLD:
                self._stable_count += 1
            else:
                self._stable_count = 0
            if self._stable_count >= STABILIZE_WINDOW:
                self._set_baseline(gray)

        self._prev_mean = mean_b

    def _set_baseline(self, gray: np.ndarray):
        """Set baseline — activate exactly 9 holes from the brighter sector."""
        brightnesses = [
            self._sample_brightness(gray, h["x"], h["y"])
            for h in self.holes
        ]
        for h, b in zip(self.holes, brightnesses):
            h["baseline"] = b

        left_holes  = [(h, b) for h, b in zip(self.holes, brightnesses) if h["side"] == "L"]
        right_holes = [(h, b) for h, b in zip(self.holes, brightnesses) if h["side"] == "R"]
        left_mean   = np.mean([b for _, b in left_holes])
        right_mean  = np.mean([b for _, b in right_holes])
        active_holes = left_holes if left_mean > right_mean else right_holes

        for h in self.holes:
            h["led_off"] = True
            h["filled"]  = False

        for h, b in active_holes:
            h["led_off"]  = False
            h["baseline"] = b

        self._baseline_set = True
        active = sum(1 for h in self.holes if not h["led_off"])
        print(f"[HoleTracker] Baseline set at frame {self._frame_count}")
        print(f"[HoleTracker] Active holes: {active}/18")
        print(f"[HoleTracker] Active side: {self.active_side} "
              f"→ patient uses {'RIGHT' if self.active_side == 'L' else 'LEFT'} hand")

    def _sector_blocked(self, all_landmarks, side: str) -> bool:
        """True if any landmark is inside the sector bounding box."""
        if all_landmarks is None:
            return False
        bbox = self._left_bbox if side == "L" else self._right_bbox
        if bbox is None:
            return False
        x1, y1, x2, y2 = bbox
        for (lx, ly) in all_landmarks:
            if x1 <= lx <= x2 and y1 <= ly <= y2:
                return True
        return False

    @staticmethod
    def _sample_brightness(gray: np.ndarray, x: int, y: int) -> float:
        """Average brightness in small region around (x, y)."""
        h, w = gray.shape
        x1 = max(0, x - SAMPLE_RADIUS)
        x2 = min(w, x + SAMPLE_RADIUS)
        y1 = max(0, y - SAMPLE_RADIUS)
        y2 = min(h, y + SAMPLE_RADIUS)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(np.mean(roi))