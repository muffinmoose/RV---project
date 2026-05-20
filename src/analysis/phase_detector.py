# analysis/phase_detector.py
# Rule-based phase detection for 9HPT
#
# The 9HPT cycle per peg:
#   IDLE       → hand is resting, test not started
#   REACHING   → hand moving toward peg hole (high velocity, moving to target)
#   GRASPING   → hand slowing down near hole (low velocity, pinch forming)
#   TRANSPORTING → hand moving peg to destination hole
#   PLACING    → hand slowing down at destination (low velocity)
#   RETURNING  → hand moving back to peg storage area
#
# Detection is purely rule-based:
#   - velocity thresholds  (from KinematicState)
#   - pinch distance       (thumb_tip ↔ index_tip distance in mm)
#   - position proximity   to known hole locations

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List

from config import FPS, HOLE_SPACING_MM, HOLE_ROWS, HOLE_COLS


# ── Phase definitions ─────────────────────────────────────────────────────────

class Phase(Enum):
    IDLE         = auto()
    REACHING     = auto()
    GRASPING     = auto()
    TRANSPORTING = auto()
    PLACING      = auto()
    RETURNING    = auto()


# ── Thresholds (tune these after seeing real data) ────────────────────────────

# Velocity thresholds in mm/s
VEL_MOVING_THRESHOLD  = 15.0   # above this → hand is moving
VEL_SLOW_THRESHOLD    = 8.0    # below this → hand is slowing / stationary

# Pinch distance thresholds in mm
PINCH_CLOSED_MM = 25.0   # thumb-index distance below this → closed pinch (holding peg)
PINCH_OPEN_MM   = 40.0   # above this → open hand (not holding)

# How many consecutive frames must confirm a phase before switching
PHASE_CONFIRM_FRAMES = 4


# ── Event logged when a peg cycle completes ───────────────────────────────────

@dataclass
class PegEvent:
    """One complete peg pick-and-place cycle."""
    peg_number:      int
    frame_start:     int
    frame_end:       int
    duration_s:      float
    reach_frames:    int = 0
    grasp_frames:    int = 0
    transport_frames:int = 0
    place_frames:    int = 0


# ── Main detector ─────────────────────────────────────────────────────────────

class PhaseDetector:
    """
    Stateful rule-based phase detector.
    Feed it one frame at a time via update().

    Usage:
        pd = PhaseDetector()
        for frame_idx, states in enumerate(kinematic_states):
            phase, event = pd.update(states, frame_idx)
            if event:
                print(f"Peg {event.peg_number} done in {event.duration_s:.2f}s")
        pd.reset()
    """

    def __init__(self):
        self.current_phase:   Phase = Phase.IDLE
        self._candidate:      Phase = Phase.IDLE
        self._candidate_count: int  = 0

        self._phase_start_frame: int = 0
        self._peg_count:         int = 0

        # Frame counters per phase in current peg cycle
        self._reach_frames:     int = 0
        self._grasp_frames:     int = 0
        self._transport_frames: int = 0
        self._place_frames:     int = 0

        self._cycle_start_frame: int = 0

        # History for UI / logging
        self.phase_history: List[Phase] = []
        self.events:        List[PegEvent] = []

    # ── Main update ───────────────────────────────────────────────────────────

    def update(self, states: dict, frame_idx: int) -> tuple:
        """
        Args:
            states:    dict of KinematicState — output of MultiLandmarkKinematics
            frame_idx: current frame number

        Returns:
            (Phase, Optional[PegEvent])
            PegEvent is not None when a full peg cycle just completed.
        """
        # Extract features
        wrist     = states["wrist"]
        thumb     = states["thumb_tip"]
        index     = states["index_tip"]

        velocity  = wrist.velocity or 0.0
        pinch_dist = self._pinch_distance(thumb.position, index.position)

        # Determine candidate phase from rules
        candidate = self._classify(velocity, pinch_dist)

        # Require PHASE_CONFIRM_FRAMES consecutive frames before switching
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate       = candidate
            self._candidate_count = 1

        event = None
        if self._candidate_count >= PHASE_CONFIRM_FRAMES:
            if candidate != self.current_phase:
                event = self._on_phase_change(candidate, frame_idx)
                self.current_phase = candidate

        # Count frames per phase in current cycle
        self._count_phase_frame(self.current_phase)

        self.phase_history.append(self.current_phase)
        return self.current_phase, event

    # ── Rule-based classification ─────────────────────────────────────────────

    def _classify(self, velocity: float, pinch_dist: float) -> Phase:
        """
        Simple decision tree based on velocity + pinch distance.
        Tune thresholds in constants above after reviewing real data.
        """
        pinch_closed = pinch_dist < PINCH_CLOSED_MM
        moving       = velocity   > VEL_MOVING_THRESHOLD
        slow         = velocity   < VEL_SLOW_THRESHOLD

        if self.current_phase == Phase.IDLE:
            if moving:
                return Phase.REACHING
            return Phase.IDLE

        if self.current_phase == Phase.REACHING:
            if slow and not pinch_closed:
                return Phase.GRASPING
            if slow and pinch_closed:
                return Phase.TRANSPORTING
            return Phase.REACHING

        if self.current_phase == Phase.GRASPING:
            if pinch_closed and moving:
                return Phase.TRANSPORTING
            if not moving and not pinch_closed:
                return Phase.IDLE
            return Phase.GRASPING

        if self.current_phase == Phase.TRANSPORTING:
            if slow and pinch_closed:
                return Phase.PLACING
            return Phase.TRANSPORTING

        if self.current_phase == Phase.PLACING:
            if not pinch_closed and moving:
                return Phase.RETURNING
            if not pinch_closed and slow:
                # Peg placed, hand releasing
                return Phase.RETURNING
            return Phase.PLACING

        if self.current_phase == Phase.RETURNING:
            if slow:
                return Phase.IDLE
            return Phase.RETURNING

        return Phase.IDLE

    # ── Phase transition handler ──────────────────────────────────────────────

    def _on_phase_change(self, new_phase: Phase, frame_idx: int) -> Optional[PegEvent]:
        """
        Called when phase switches. Returns a PegEvent if a full cycle completed.
        """
        event = None

        # A new peg cycle starts when we leave IDLE into REACHING
        if new_phase == Phase.REACHING and self.current_phase == Phase.IDLE:
            self._cycle_start_frame = frame_idx
            self._reach_frames      = 0
            self._grasp_frames      = 0
            self._transport_frames  = 0
            self._place_frames      = 0

        # A peg cycle completes when we return to IDLE after RETURNING
        if new_phase == Phase.IDLE and self.current_phase == Phase.RETURNING:
            self._peg_count += 1
            duration = (frame_idx - self._cycle_start_frame) / FPS
            event = PegEvent(
                peg_number=       self._peg_count,
                frame_start=      self._cycle_start_frame,
                frame_end=        frame_idx,
                duration_s=       duration,
                reach_frames=     self._reach_frames,
                grasp_frames=     self._grasp_frames,
                transport_frames= self._transport_frames,
                place_frames=     self._place_frames,
            )
            self.events.append(event)

        return event

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pinch_distance(thumb_mm: np.ndarray, index_mm: np.ndarray) -> float:
        """Euclidean distance between thumb tip and index tip in mm."""
        if thumb_mm is None or index_mm is None:
            return float("inf")
        return float(np.linalg.norm(thumb_mm - index_mm))

    def _count_phase_frame(self, phase: Phase):
        if phase == Phase.REACHING:
            self._reach_frames     += 1
        elif phase == Phase.GRASPING:
            self._grasp_frames     += 1
        elif phase == Phase.TRANSPORTING:
            self._transport_frames += 1
        elif phase == Phase.PLACING:
            self._place_frames     += 1

    def reset(self):
        """Reset all state between patients/videos."""
        self.current_phase    = Phase.IDLE
        self._candidate       = Phase.IDLE
        self._candidate_count = 0
        self._phase_start_frame = 0
        self._peg_count       = 0
        self._reach_frames    = 0
        self._grasp_frames    = 0
        self._transport_frames= 0
        self._place_frames    = 0
        self._cycle_start_frame = 0
        self.phase_history    = []
        self.events           = []

    @property
    def peg_count(self) -> int:
        return self._peg_count


# ── Phase color map for visualizer ───────────────────────────────────────────
# BGR colors for each phase overlay in OpenCV

PHASE_COLORS = {
    Phase.IDLE:         (100, 100, 100),   # grey
    Phase.REACHING:     (0,   200, 255),   # yellow
    Phase.GRASPING:     (0,   140, 255),   # orange
    Phase.TRANSPORTING: (0,   255,   0),   # green
    Phase.PLACING:      (255, 100,   0),   # blue
    Phase.RETURNING:    (255,   0, 150),   # purple
}