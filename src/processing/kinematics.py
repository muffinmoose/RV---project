# processing/kinematics.py
# Computes kinematic features from smoothed mm coordinates:
#   - velocity (mm/s)
#   - acceleration (mm/s²)
#   - total path length (mm)
#   - displacement (mm)
#
# Input:  smoothed [x_mm, y_mm] from Kalman filter (one point per frame)
# Output: KinematicState dataclass with all features for that frame

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import FPS, SLIDING_WINDOW_FRAMES


@dataclass
class KinematicState:
    """
    All kinematic features for a single frame.
    All values are in mm and mm/s and mm/s².
    None means not enough frames yet to compute that feature.
    """
    position:     np.ndarray          # [x_mm, y_mm] current position
    velocity:     Optional[float] = None   # scalar speed mm/s
    velocity_vec: Optional[np.ndarray] = None  # [vx, vy] mm/s
    acceleration: Optional[float] = None   # scalar mm/s²
    path_length:  float = 0.0         # total path length so far (mm)
    displacement: Optional[float] = None   # straight-line from start (mm)


class KinematicsCalculator:
    """
    Computes kinematics from a stream of mm coordinates (one per frame).
    Uses a sliding window for velocity/acceleration smoothing.

    Usage:
        kin = KinematicsCalculator()
        for frame in video:
            smooth = hk.update(detection)          # from HandKalman
            mm_pos = cal.pixel_to_mm(smooth["wrist"])
            state  = kin.update(mm_pos)
            print(state.velocity, state.acceleration)
        kin.reset()   # between videos
    """

    def __init__(self, fps: float = FPS, window: int = SLIDING_WINDOW_FRAMES):
        self.fps    = fps
        self.dt     = 1.0 / fps
        self.window = window

        # Rolling buffer of recent mm positions
        self._positions: deque = deque(maxlen=window + 1)

        # Rolling buffer of recent velocities (for acceleration)
        self._velocities: deque = deque(maxlen=window + 1)

        # Cumulative path length
        self._path_length: float = 0.0

        # Starting position (for displacement)
        self._start_pos: Optional[np.ndarray] = None

    def update(self, mm_pos: np.ndarray) -> KinematicState:
        """
        Feed one frame's smoothed mm position, get back full KinematicState.

        Args:
            mm_pos: np.ndarray shape (2,) — [x_mm, y_mm]

        Returns:
            KinematicState for this frame
        """
        mm_pos = mm_pos.astype(np.float64)

        # Record start position
        if self._start_pos is None:
            self._start_pos = mm_pos.copy()

        # Update path length
        if len(self._positions) > 0:
            step = float(np.linalg.norm(mm_pos - self._positions[-1]))
            self._path_length += step

        self._positions.append(mm_pos.copy())

        # ── Velocity ──────────────────────────────────────────────────────────
        velocity     = None
        velocity_vec = None

        if len(self._positions) >= 2:
            # Central difference over the available window
            pts  = list(self._positions)
            n    = len(pts)

            if n >= 3:
                # Central difference: v = (pos[+1] - pos[-1]) / (2*dt) — smoothest
                vel_vec = (pts[-1] - pts[-3]) / (2 * self.dt)
            else:
                # Only 2 points — forward difference
                vel_vec = (pts[-1] - pts[-2]) / self.dt

            velocity_vec = vel_vec
            velocity     = float(np.linalg.norm(vel_vec))
            self._velocities.append(velocity)

        # ── Acceleration ──────────────────────────────────────────────────────
        acceleration = None

        if len(self._velocities) >= 2:
            vels = list(self._velocities)
            n    = len(vels)

            if n >= 3:
                acceleration = (vels[-1] - vels[-3]) / (2 * self.dt)
            else:
                acceleration = (vels[-1] - vels[-2]) / self.dt

        # ── Displacement ──────────────────────────────────────────────────────
        displacement = float(np.linalg.norm(mm_pos - self._start_pos)) \
                       if self._start_pos is not None else None

        return KinematicState(
        position=mm_pos / 1000.0,           # mm → m
        velocity=velocity / 1000.0 if velocity is not None else None,
        velocity_vec=velocity_vec / 1000.0 if velocity_vec is not None else None,
        acceleration=acceleration / 1000.0 if acceleration is not None else None,
        path_length=self._path_length / 1000.0,
        displacement=displacement / 1000.0 if displacement is not None else None,
    )

    def reset(self):
        """Reset all state — call between videos/patients."""
        self._positions.clear()
        self._velocities.clear()
        self._path_length = 0.0
        self._start_pos   = None


class MultiLandmarkKinematics:
    """
    Runs KinematicsCalculator for all three landmarks simultaneously.

    Usage:
        mk = MultiLandmarkKinematics()
        states = mk.update(smooth_mm)   # dict of KinematicState per landmark
        mk.reset()
    """

    def __init__(self, fps: float = FPS, window: int = SLIDING_WINDOW_FRAMES):
        self.calculators = {
            "wrist":     KinematicsCalculator(fps, window),
            "thumb_tip": KinematicsCalculator(fps, window),
            "index_tip": KinematicsCalculator(fps, window),
        }

    def update(self, smooth_mm: dict) -> dict:
        """
        Args:
            smooth_mm: dict {"wrist": np.ndarray, "thumb_tip": ..., "index_tip": ...}
                       All values are [x_mm, y_mm] — output of Calibrator.pixel_to_mm()

        Returns:
            dict {"wrist": KinematicState, "thumb_tip": ..., "index_tip": ...}
        """
        return {
            name: self.calculators[name].update(smooth_mm[name])
            for name in self.calculators
        }

    def reset(self):
        for calc in self.calculators.values():
            calc.reset()


# ── Utility: compute summary stats over a completed trial ─────────────────────

def trial_summary(states: list) -> dict:
    """
    Compute summary statistics from a list of KinematicState objects
    collected over one full trial (e.g. one peg placement).

    Args:
        states: list of KinematicState from one trial

    Returns:
        dict with summary stats
    """
    velocities     = [s.velocity     for s in states if s.velocity     is not None]
    accelerations  = [s.acceleration for s in states if s.acceleration is not None]
    path_length    = states[-1].path_length if states else 0.0
    displacement   = states[-1].displacement if states else None

    return {
        "duration_s":       len(states) / FPS,
        "path_length_mm":   path_length,
        "displacement_mm":  displacement,
        "mean_velocity":    float(np.mean(velocities))    if velocities    else None,
        "max_velocity":     float(np.max(velocities))     if velocities    else None,
        "mean_acceleration":float(np.mean(np.abs(accelerations))) if accelerations else None,
        "max_acceleration": float(np.max(np.abs(accelerations)))  if accelerations else None,
    }