# analysis/phase_detector.py
# HoleTracker-based phase detection za 9HPT
#
# Strategija (namesto rule-based velocity/pinch):
#   filled_count raste 0→9  = PLACING faza  (pacient polni luknjice)
#   filled_count pada  9→0  = RETURNING faza (pacient jemlje pine nazaj)
#   Vsaka nova zapolnjena luknjica = en PegEvent
#
# PhaseDetector.update() sedaj sprejme tudi filled_count iz HoleTracker.
# Kinematični states (velocity, pinch) se še vedno beležijo za grafe —
# samo štetje pegov in faz temelji na luknjicah.

import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List

from config import FPS


# ── Phase definitions ─────────────────────────────────────────────────────────

class Phase(Enum):
    IDLE      = auto()   # čakanje pred testom / med testi
    PLACING   = auto()   # pacient polni luknjice (filled_count raste)
    RETURNING = auto()   # pacient jemlje pine nazaj (filled_count pada)


# ── Peg event ─────────────────────────────────────────────────────────────────

@dataclass
class PegEvent:
    """En zapolnjen pin — od prejšnjega fill do tega fill."""
    peg_number:   int     # zaporedna številka pina (1-9)
    frame_start:  int     # frame ko je bil prejšnji pin zapolnjen
    frame_end:    int     # frame ko je bil ta pin zapolnjen
    duration_s:   float   # čas med dvema zaporednima filloma (sekunde)
    # Kinematični povzetki za ta interval (iz states)
    mean_velocity:    Optional[float] = None   # m/s
    max_velocity:     Optional[float] = None   # m/s
    mean_acceleration:Optional[float] = None   # m/s²
    path_length:      Optional[float] = None   # m


# ── Main detector ─────────────────────────────────────────────────────────────

class PhaseDetector:
    """
    HoleTracker-based phase detector.

    Kliči update() za vsak frame iz main.py.
    Posreduj filled_count iz HoleTracker — vse ostalo je avtomatsko.

    Uporaba:
        pd = PhaseDetector()
        phase, event = pd.update(states, frame_idx, filled_count=ht.filled_count)
        if event:
            print(f"Pin {event.peg_number} → {event.duration_s:.2f}s")
    """

    def __init__(self):
        self.current_phase = Phase.IDLE

        self._peg_count        = 0          # koliko pinov zapolnjenih
        self._last_fill_frame  = 0          # frame zadnjega fill eventa
        self._last_filled      = 0          # prejšnji filled_count
        self._placing_started  = False      # ali je PLACING faza začeta
        self._returning_started= False      # ali je RETURNING faza začeta

        # Kinematični buffer za trenutni interval med dvema filloma
        self._interval_velocities:     List[float] = []
        self._interval_accelerations:  List[float] = []
        self._interval_path_start:     float = 0.0

        self.phase_history: List[Phase]    = []
        self.events:        List[PegEvent] = []

    # ── Glavni update ─────────────────────────────────────────────────────────

    def update(self, states: dict, frame_idx: int,
               filled_count: int = 0) -> tuple:
        """
        Kliče se vsak frame iz main.py.

        Args:
            states:       dict KinematicState iz MultiLandmarkKinematics
            frame_idx:    trenutna številka framea
            filled_count: ht.filled_count iz HoleTracker (0-9)

        Returns:
            (Phase, Optional[PegEvent]) — nikoli None
        """
        try:
            wrist = states.get("wrist")

            # Zberaj kinematiko za trenutni interval
            if wrist is not None:
                if wrist.velocity is not None:
                    self._interval_velocities.append(wrist.velocity)
                if wrist.acceleration is not None:
                    self._interval_accelerations.append(abs(wrist.acceleration))

            # ── Določi fazo iz filled_count ───────────────────────────────────
            if filled_count > 0 and not self._placing_started:
                # Začetek PLACING — prvi pin zapolnjen
                self._placing_started  = True
                self._returning_started= False
                self.current_phase     = Phase.PLACING
                self._last_fill_frame  = frame_idx
                self._last_filled      = filled_count
                print(f"[PhaseDetector] PLACING started at frame {frame_idx}")

            elif filled_count == 9 and self._placing_started and not self._returning_started:
                # Vsi pini zapolnjeni — čakamo na začetek RETURNING
                self.current_phase = Phase.PLACING

            elif filled_count < self._last_filled and self._placing_started:
                if not self._returning_started:
                    # Začetek RETURNING — prvi pin vzet nazaj
                    self._returning_started = True
                    self.current_phase      = Phase.RETURNING
                    print(f"[PhaseDetector] RETURNING started at frame {frame_idx}")
                else:
                    self.current_phase = Phase.RETURNING

            elif filled_count == 0 and self._returning_started:
                # Konec — vsi pini vzeti nazaj
                self.current_phase      = Phase.IDLE
                self._placing_started   = False
                self._returning_started = False
                print(f"[PhaseDetector] Test complete at frame {frame_idx}")

            # ── Zazaj fill event (nov pin zapolnjen) ──────────────────────────
            event = None
            if (filled_count > self._last_filled
                    and self.current_phase == Phase.PLACING):
                event = self._on_new_fill(frame_idx, states)

            self._last_filled = filled_count
            self.phase_history.append(self.current_phase)
            return self.current_phase, event

        except Exception as e:
            print(f"[PhaseDetector] ERROR frame {frame_idx}: {e}")
            return self.current_phase, None

    # ── Fill event handler ────────────────────────────────────────────────────

    def _on_new_fill(self, frame_idx: int, states: dict) -> PegEvent:
        """
        Kliče se ko HoleTracker zazna novo zapolnjeno luknjico.
        Izračuna trajanje in kinematične povzetke za ta interval.
        """
        self._peg_count += 1

        duration = (frame_idx - self._last_fill_frame) / FPS

        # Kinematični povzetki za interval od zadnjega fill do tega
        mean_vel  = float(np.mean(self._interval_velocities))  \
                    if self._interval_velocities  else None
        max_vel   = float(np.max(self._interval_velocities))   \
                    if self._interval_velocities  else None
        mean_acc  = float(np.mean(self._interval_accelerations))\
                    if self._interval_accelerations else None

        # Path length — razlika od zadnjega fill
        wrist = states.get("wrist")
        path  = None
        if wrist is not None and wrist.path_length is not None:
            path = wrist.path_length - self._interval_path_start
            self._interval_path_start = wrist.path_length

        event = PegEvent(
            peg_number=        self._peg_count,
            frame_start=       self._last_fill_frame,
            frame_end=         frame_idx,
            duration_s=        duration,
            mean_velocity=     mean_vel,
            max_velocity=      max_vel,
            mean_acceleration= mean_acc,
            path_length=       path,
        )
        self.events.append(event)

        mean_str = f"{mean_vel:.3f}" if mean_vel is not None else "N/A"
        print(f"[PhaseDetector] Pin {self._peg_count}/9 → {duration:.2f}s vel_mean={mean_str}m/s")

        # Reset interval buffer za naslednji pin
        self._interval_velocities    = []
        self._interval_accelerations = []
        self._last_fill_frame        = frame_idx

        return event

    # ── Helpers ───────────────────────────────────────────────────────────────

    def reset(self):
        """Reset med videi/pacienti."""
        self.current_phase          = Phase.IDLE
        self._peg_count             = 0
        self._last_fill_frame       = 0
        self._last_filled           = 0
        self._placing_started       = False
        self._returning_started     = False
        self._interval_velocities   = []
        self._interval_accelerations= []
        self._interval_path_start   = 0.0
        self.phase_history          = []
        self.events                 = []

    @property
    def peg_count(self) -> int:
        return self._peg_count


# ── Phase barve za visualizer (BGR za OpenCV) ─────────────────────────────────

PHASE_COLORS = {
    Phase.IDLE:      (100, 100, 100),   # siva
    Phase.PLACING:   (0,   255,   0),   # zelena
    Phase.RETURNING: (255,   0, 150),   # vijolična
}