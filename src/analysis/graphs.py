# analysis/graphs.py
# Generates kinematic graphs for a processed video session.
#
# Outputs a .png with:
#   - Top plot:    velocity magnitude + vx + vy over time (m/s)
#   - Bottom plot: acceleration over time (m/s²)
#   - Colored vertical bands showing active phase at each frame
#
# Usage (called automatically from main.py after processing):
#   from analysis.graphs import save_graphs
#   save_graphs(history, output_path)

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional

from analysis.phase_detector import Phase


# ── Phase colors for matplotlib (RGB 0-1) ────────────────────────────────────

PHASE_COLORS_MPL = {
    Phase.IDLE:          (0.4,  0.4,  0.4,  0.15),
    Phase.REACHING:      (1.0,  0.8,  0.0,  0.20),
    Phase.GRASPING:      (1.0,  0.5,  0.0,  0.20),
    Phase.TRANSPORTING:  (0.0,  0.8,  0.0,  0.20),
    Phase.PLACING:       (0.0,  0.4,  1.0,  0.20),
    Phase.RETURNING:     (0.6,  0.0,  0.8,  0.20),
}

PHASE_LABEL_COLORS = {
    Phase.IDLE:          "#666666",
    Phase.REACHING:      "#ccaa00",
    Phase.GRASPING:      "#ff8800",
    Phase.TRANSPORTING:  "#00bb00",
    Phase.PLACING:       "#0066ff",
    Phase.RETURNING:     "#9900cc",
}


# ── Data container filled frame by frame in main.py ──────────────────────────

class KinematicHistory:
    """
    Accumulates per-frame kinematic data during video processing.
    Pass an instance to main.py and call record() each frame.
    After processing, pass to save_graphs().
    """

    def __init__(self, fps: float):
        self.fps = fps

        self.times:        List[float] = []   # seconds
        self.vel_mag:      List[float] = []   # |v| m/s
        self.vel_x:        List[float] = []   # vx m/s
        self.vel_y:        List[float] = []   # vy m/s
        self.accel:        List[float] = []   # |a| m/s²
        self.phases:       List[Phase] = []   # phase at each frame
        self.path_lengths: List[float] = []   # cumulative path m

    def record(self, frame_idx: int, states: dict, phase: Phase):
        """Call once per frame from main.py."""
        t     = frame_idx / self.fps
        wrist = states.get("wrist")

        self.times.append(t)
        self.phases.append(phase)

        if wrist and wrist.velocity is not None:
            self.vel_mag.append(wrist.velocity)
        else:
            self.vel_mag.append(0.0)

        if wrist and wrist.velocity_vec is not None:
            self.vel_x.append(float(wrist.velocity_vec[0]))
            self.vel_y.append(float(wrist.velocity_vec[1]))
        else:
            self.vel_x.append(0.0)
            self.vel_y.append(0.0)

        if wrist and wrist.acceleration is not None:
            self.accel.append(abs(wrist.acceleration))
        else:
            self.accel.append(0.0)

        self.path_lengths.append(wrist.path_length if wrist else 0.0)


# ── Graph generation ──────────────────────────────────────────────────────────

def save_graphs(history: KinematicHistory,
                output_path: str,
                patient_id: str = "",
                video_name: str = "") -> None:
    """
    Generate and save kinematic graphs as a .png file.

    Args:
        history:     KinematicHistory filled during processing
        output_path: path to save the .png
        patient_id:  for title
        video_name:  for title
    """
    t      = np.array(history.times)
    v_mag  = np.array(history.vel_mag)
    v_x    = np.array(history.vel_x)
    v_y    = np.array(history.vel_y)
    a_mag  = np.array(history.accel)
    phases = history.phases

    if len(t) == 0:
        print("[Graphs] No data to plot.")
        return

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#1e1e1e")

    title = f"Kinematic Analysis"
    if patient_id:
        title += f" — {patient_id}"
    if video_name:
        title += f"\n{video_name}"
    fig.suptitle(title, color="white", fontsize=13, y=0.98)

    # ── Draw phase bands on all axes ──────────────────────────────────────────
    _draw_phase_bands(axes, t, phases)

    # ── Plot 1: Velocity magnitude + components ───────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#2d2d2d")
    ax1.plot(t, v_mag, color="#ffcc00", linewidth=1.5, label="|v| (magnitude)", zorder=3)
    ax1.plot(t, v_x,   color="#00bfff", linewidth=1.0, label="vx",             zorder=3, alpha=0.8)
    ax1.plot(t, v_y,   color="#ff6eb4", linewidth=1.0, label="vy",             zorder=3, alpha=0.8)
    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Velocity (m/s)", color="white")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#555555")
    ax1.legend(loc="upper right", facecolor="#3d3d3d",
               labelcolor="white", fontsize=8)
    ax1.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Plot 2: Acceleration magnitude ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#2d2d2d")
    ax2.plot(t, a_mag, color="#ff4444", linewidth=1.5, label="|a|", zorder=3)
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Acceleration (m/s²)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#555555")
    ax2.legend(loc="upper right", facecolor="#3d3d3d",
               labelcolor="white", fontsize=8)
    ax2.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Plot 3: Cumulative path length ────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#2d2d2d")
    path = np.array(history.path_lengths)
    ax3.plot(t, path, color="#44ff88", linewidth=1.5, label="Path length (m)", zorder=3)
    ax3.set_ylabel("Path length (m)", color="white")
    ax3.set_xlabel("Time (s)", color="white")
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#555555")
    ax3.legend(loc="upper left", facecolor="#3d3d3d",
               labelcolor="white", fontsize=8)
    ax3.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Phase legend ──────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=PHASE_LABEL_COLORS[p], label=p.name)
        for p in Phase
    ]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               facecolor="#2d2d2d", labelcolor="white",
               fontsize=8, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Graphs] Saved → {output_path}")


def _draw_phase_bands(axes, t: np.ndarray, phases: List[Phase]):
    """Draw colored vertical bands for each phase on all axes."""
    if len(t) == 0:
        return

    current_phase = phases[0]
    start_t       = t[0]

    for i in range(1, len(phases)):
        if phases[i] != current_phase or i == len(phases) - 1:
            end_t = t[i]
            color = PHASE_COLORS_MPL.get(current_phase, (0.5, 0.5, 0.5, 0.1))
            for ax in axes:
                ax.axvspan(start_t, end_t, color=color, zorder=1)
            current_phase = phases[i]
            start_t       = t[i]