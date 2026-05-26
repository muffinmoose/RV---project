# processing/kinematics.py
# Računa kinematične značilke iz zglajenh mm koordinat:
#   - velocity     (m/s)
#   - acceleration (m/s²)
#   - path length  (m)
#   - displacement (m)
#
# Vhod:  zglajen [x_mm, y_mm] iz Kalman filtra
# Izhod: KinematicState dataclass — VSE vrednosti v metrih in m/s
#
# OPOMBA: interno računamo v mm, na koncu delimo z 1000 → m
# PhaseDetector in graphs.py pričakujejo metre.

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from config import FPS, SLIDING_WINDOW_FRAMES


@dataclass
class KinematicState:
    """
    Vse kinematične značilke za en frame.
    VSE vrednosti so v metrih (m) in m/s in m/s².
    None pomeni premalo frameov za izračun.
    """
    position:     np.ndarray               # [x_m, y_m]
    velocity:     Optional[float] = None   # skalar m/s
    velocity_vec: Optional[np.ndarray] = None  # [vx, vy] m/s
    acceleration: Optional[float] = None   # skalar m/s²
    path_length:  float = 0.0              # skupna pot m
    displacement: Optional[float] = None   # premik od starta m


class KinematicsCalculator:
    """
    Računa kinematiko iz toka mm koordinat (ena na frame).
    Sliding window za glajenje velocity/acceleration.

    Uporaba:
        kin = KinematicsCalculator()
        for frame in video:
            smooth = hk.update(detection)
            mm_pos = cal.pixel_to_mm(smooth["wrist"])
            state  = kin.update(mm_pos)   # vrača m/s, m
        kin.reset()
    """

    def __init__(self, fps: float = FPS, window: int = SLIDING_WINDOW_FRAMES):
        self.fps    = fps
        self.dt     = 1.0 / fps   # časovni korak med framei
        self.window = window

        # Drsni buffer zadnjih mm pozicij (za velocity)
        self._positions:  deque = deque(maxlen=window + 1)

        # Drsni buffer zadnjih hitrosti (za acceleration)
        self._velocities: deque = deque(maxlen=window + 1)

        # Kumulativna dolžina poti v mm (delimo z 1000 na koncu)
        self._path_length: float = 0.0

        # Začetna pozicija za displacement
        self._start_pos: Optional[np.ndarray] = None

    def update(self, mm_pos: np.ndarray) -> KinematicState:
        """
        En frame — zglajen mm položaj → KinematicState v metrih.

        Args:
            mm_pos: np.ndarray shape (2,) — [x_mm, y_mm] iz Calibrator.pixel_to_mm()

        Returns:
            KinematicState — position v m, velocity v m/s, acceleration v m/s²
        """
        mm_pos = mm_pos.astype(np.float64)

        # Zapiši začetno pozicijo
        if self._start_pos is None:
            self._start_pos = mm_pos.copy()

        # Posodobi dolžino poti (v mm, delimo na koncu)
        if len(self._positions) > 0:
            step = float(np.linalg.norm(mm_pos - self._positions[-1]))
            self._path_length += step

        self._positions.append(mm_pos.copy())

        # ── Velocity ──────────────────────────────────────────────────────────
        # Centralna diferenca čez window — bolj gladko kot forward diferenca
        velocity     = None
        velocity_vec = None

        if len(self._positions) >= 2:
            pts = list(self._positions)
            n   = len(pts)

            if n >= 3:
                # Centralna diferenca: v = (pos[n-1] - pos[n-3]) / (2*dt)
                vel_vec = (pts[-1] - pts[-3]) / (2 * self.dt)
            else:
                # Samo 2 točki — forward diferenca
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

        # ── Vrni v metrih — VSE delimo z 1000 ────────────────────────────────
        # PhaseDetector, graphs.py in visualizer.py pričakujejo metre
        return KinematicState(
            position=     mm_pos / 1000.0,
            velocity=     velocity     / 1000.0 if velocity     is not None else None,
            velocity_vec= velocity_vec / 1000.0 if velocity_vec is not None else None,
            acceleration= acceleration / 1000.0 if acceleration is not None else None,
            path_length=  self._path_length / 1000.0,   # bil bug — zdaj konsistentno
            displacement= displacement / 1000.0          if displacement is not None else None,
        )

    def reset(self):
        """Reset med videi/pacienti."""
        self._positions.clear()
        self._velocities.clear()
        self._path_length = 0.0
        self._start_pos   = None


class MultiLandmarkKinematics:
    """
    Poganja KinematicsCalculator za vse tri landmarks hkrati.
    Kliče se iz main.py za vsak frame.

    Uporaba:
        mk = MultiLandmarkKinematics()
        states = mk.update(smooth_mm)   # dict KinematicState
        mk.reset()
    """

    def __init__(self, fps: float = FPS, window: int = SLIDING_WINDOW_FRAMES):
        # En kalkulator na landmark
        self.calculators = {
            "wrist":     KinematicsCalculator(fps, window),
            "thumb_tip": KinematicsCalculator(fps, window),
            "index_tip": KinematicsCalculator(fps, window),
        }

    def update(self, smooth_mm: dict) -> dict:
        """
        Args:
            smooth_mm: dict {"wrist": np.ndarray, "thumb_tip": ..., "index_tip": ...}
                       Vrednosti so [x_mm, y_mm] — izhod Calibrator.pixel_to_mm()

        Returns:
            dict {"wrist": KinematicState, "thumb_tip": ..., "index_tip": ...}
            Vse vrednosti v metrih in m/s.
        """
        return {
            name: self.calculators[name].update(smooth_mm[name])
            for name in self.calculators
        }

    def reset(self):
        for calc in self.calculators.values():
            calc.reset()


# ── Povzetek poskusa ──────────────────────────────────────────────────────────

def trial_summary(states: list) -> dict:
    """
    Izračuna povzetne statistike iz liste KinematicState za en poskus.

    Args:
        states: lista KinematicState iz enega cikla (en pin)

    Returns:
        dict s povzetnimi statistikami — vse v metrih in m/s
    """
    velocities    = [s.velocity     for s in states if s.velocity     is not None]
    accelerations = [s.acceleration for s in states if s.acceleration is not None]
    path_length   = states[-1].path_length  if states else 0.0
    displacement  = states[-1].displacement if states else None

    return {
        "duration_s":        len(states) / FPS,
        "path_length_m":     path_length,
        "displacement_m":    displacement,
        "mean_velocity":     float(np.mean(velocities))              if velocities    else None,
        "max_velocity":      float(np.max(velocities))               if velocities    else None,
        "mean_acceleration": float(np.mean(np.abs(accelerations)))   if accelerations else None,
        "max_acceleration":  float(np.max(np.abs(accelerations)))    if accelerations else None,
    }