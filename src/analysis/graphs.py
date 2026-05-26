# analysis/graphs.py
# Generates all output figures for a processed 9HPT video session.
#
# Exported functions:
#   save_graphs(history, output_path, patient_id, video_name)
#       → kinematic graphs (velocity, acceleration, path length) + FFT tremor
#   save_board_figure(place_order, pick_order, output_path, patient_id, side,
#                     trajectory_px, frame_w, frame_h)
#       → 9HPT board with pin order + hand trajectory overlay

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional, Tuple

from analysis.phase_detector import Phase, PHASE_COLORS
from config import LEFT_HOLES_PX, RIGHT_HOLES_PX, BOARD_HOLES_BBOX_PX


# ── Phase colors ──────────────────────────────────────────────────────────────

# Matplotlib barve za grafe — (R, G, B, alpha)
PHASE_COLORS_MPL = {
    Phase.IDLE:      (0.4, 0.4, 0.4, 0.15),   # siva
    Phase.PLACING:   (0.0, 0.8, 0.0, 0.20),   # zelena
    Phase.RETURNING: (0.6, 0.0, 0.8, 0.20),   # vijolična
}

PHASE_LABEL_COLORS = {
    Phase.IDLE:      "#666666",
    Phase.PLACING:   "#00bb00",
    Phase.RETURNING: "#9900cc",
}

# Frames to skip after baseline (Kalman init spike suppression)
WARMUP_FRAMES = 15


# ── Data container ────────────────────────────────────────────────────────────

class KinematicHistory:
    """
    Accumulates per-frame kinematic + pixel data during video processing.
    Call record() each frame from main.py.
    Pass to save_graphs() and save_board_figure() after processing.
    """

    def __init__(self, fps: float):
        self.fps = fps

        self.times:        List[float] = []
        self.vel_mag:      List[float] = []   # |v| m/s  (wrist)
        self.vel_x:        List[float] = []   # vx m/s
        self.vel_y:        List[float] = []   # vy m/s
        self.accel:        List[float] = []   # |a| m/s²
        self.phases:       List[Phase] = []
        self.path_lengths: List[float] = []   # cumulative path m

        # Pixel trajectory of index_tip for board overlay
        # Each entry is (x_px, y_px) or None if not detected
        self.index_px:     List[Optional[Tuple[float, float]]] = []

        # Frame offset where active phase begins (set after baseline)
        self._baseline_frame: int = -1

    def set_baseline_frame(self, frame_idx: int):
        """Call once from main.py when HoleTracker baseline is set."""
        if self._baseline_frame < 0:
            self._baseline_frame = frame_idx

    def record(self, frame_idx: int, states: dict, phase: Phase,
               index_tip_px: Optional[Tuple[float, float]] = None):
        """
        Call once per frame from main.py.

        Args:
            frame_idx:    current frame number
            states:       dict of KinematicState from MultiLandmarkKinematics
            phase:        current Phase from PhaseDetector
            index_tip_px: (x, y) pixel coords of index tip, or None
        """
        t     = frame_idx / self.fps
        wrist = states.get("wrist")

        # Suppress Kalman warmup spike — zero out first WARMUP_FRAMES after baseline
        in_warmup = (self._baseline_frame >= 0 and
                     frame_idx < self._baseline_frame + WARMUP_FRAMES)

        self.times.append(t)
        self.phases.append(phase)
        self.index_px.append(index_tip_px)

        if in_warmup or wrist is None:
            self.vel_mag.append(0.0)
            self.vel_x.append(0.0)
            self.vel_y.append(0.0)
            self.accel.append(0.0)
            self.path_lengths.append(
                self.path_lengths[-1] if self.path_lengths else 0.0)
            return

        self.vel_mag.append(wrist.velocity if wrist.velocity is not None else 0.0)

        if wrist.velocity_vec is not None:
            self.vel_x.append(float(wrist.velocity_vec[0]))
            self.vel_y.append(float(wrist.velocity_vec[1]))
        else:
            self.vel_x.append(0.0)
            self.vel_y.append(0.0)

        self.accel.append(abs(wrist.acceleration)
                          if wrist.acceleration is not None else 0.0)
        self.path_lengths.append(wrist.path_length if wrist else 0.0)


# ── Kinematic graphs ──────────────────────────────────────────────────────────

def save_graphs(history: "KinematicHistory",
                output_path: str,
                patient_id: str = "",
                video_name: str = "") -> None:
    """
    Save kinematic graphs (velocity, acceleration, path length, FFT tremor).

    Spike suppression: first WARMUP_FRAMES after baseline are zeroed in record().
    FFT tremor analysis: applied to wrist velocity signal, shows dominant
    tremor frequencies. Pathological tremor typically 4-12 Hz (Parkinson 4-6 Hz,
    essential tremor 8-12 Hz).
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

    # ── Figure: 4 subplots (3 kinematic + 1 FFT) ─────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=False)
    fig.patch.set_facecolor("#1e1e1e")

    title = "Kinematic Analysis"
    if patient_id:
        title += f" — {patient_id}"
    if video_name:
        title += f"\n{video_name}"
    fig.suptitle(title, color="white", fontsize=13, y=0.99)

    # Share x-axis only for first 3 plots
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])

    _draw_phase_bands(axes[:3], t, phases)

    # ── Plot 1: Velocity ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#2d2d2d")
    ax1.plot(t, v_mag, color="#ffcc00", linewidth=1.5, label="|v|", zorder=3)
    ax1.plot(t, v_x,   color="#00bfff", linewidth=1.0, label="vx",  zorder=3, alpha=0.8)
    ax1.plot(t, v_y,   color="#ff6eb4", linewidth=1.0, label="vy",  zorder=3, alpha=0.8)
    ax1.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax1.set_ylabel("Velocity (m/s)", color="white")
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#555555")
    ax1.legend(loc="upper right", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax1.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Plot 2: Acceleration ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#2d2d2d")
    ax2.plot(t, a_mag, color="#ff4444", linewidth=1.5, label="|a|", zorder=3)
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Acceleration (m/s²)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#555555")
    ax2.legend(loc="upper right", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax2.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Plot 3: Path length ───────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#2d2d2d")
    path = np.array(history.path_lengths)
    ax3.plot(t, path, color="#44ff88", linewidth=1.5, label="Path length (m)", zorder=3)
    ax3.set_ylabel("Path length (m)", color="white")
    ax3.set_xlabel("Time (s)", color="white")
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#555555")
    ax3.legend(loc="upper left", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax3.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Plot 4: FFT tremor analysis ───────────────────────────────────────────
    # Use velocity magnitude signal — tremor appears as periodic oscillation.
    # Only analyse the active portion (after baseline, non-zero signal).
    ax4 = axes[3]
    ax4.set_facecolor("#2d2d2d")

    fps = history.fps
    signal = v_mag.copy()

    # Remove DC component (mean) before FFT
    signal = signal - np.mean(signal)

    if len(signal) > 10:
        fft_vals  = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)

        # Only show 0-25 Hz (relevant for tremor)
        mask = fft_freqs <= 25.0
        freqs = fft_freqs[mask]
        power = fft_vals[mask]

        # Normalize power
        if power.max() > 0:
            power = power / power.max()

        ax4.plot(freqs, power, color="#bb88ff", linewidth=1.5, label="FFT power", zorder=3)

        # Highlight tremor band 4-12 Hz
        ax4.axvspan(4, 12, color=(0.8, 0.3, 0.3, 0.15), zorder=1,
                    label="Tremor band (4-12 Hz)")

        # Mark peak frequency in tremor band
        tremor_mask = (freqs >= 4) & (freqs <= 12)
        if tremor_mask.any():
            peak_freq = freqs[tremor_mask][np.argmax(power[tremor_mask])]
            peak_pow  = power[tremor_mask].max()
            ax4.axvline(peak_freq, color="#ff4444", linewidth=1.0,
                        linestyle="--", alpha=0.8)
            ax4.text(peak_freq + 0.2, peak_pow * 0.95,
                     f"peak: {peak_freq:.1f} Hz",
                     color="#ff4444", fontsize=8)

    ax4.set_ylabel("Normalised power", color="white")
    ax4.set_xlabel("Frequency (Hz)", color="white")
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#555555")
    ax4.legend(loc="upper right", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax4.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # ── Phase legend ──────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=PHASE_LABEL_COLORS[p], label=p.name) for p in Phase]
    fig.legend(handles=patches, loc="lower center", ncol=6,
               facecolor="#2d2d2d", labelcolor="white",
               fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Graphs] Saved → {output_path}")


def _draw_phase_bands(axes, t: np.ndarray, phases: List[Phase]):
    """Draw colored vertical phase bands on a list of axes."""
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


# ── Board figure ──────────────────────────────────────────────────────────────

def save_board_figure(place_order:    List[int],
                      pick_order:     List[int],
                      output_path:    str,
                      patient_id:     str = "",
                      side:           str = "right",
                      trajectory_px:  Optional[List[Optional[Tuple[float,float]]]] = None,
                      frame_w:        int = 640,
                      frame_h:        int = 480) -> None:
    """
    Generate 9HPT board visualization with pin order and hand trajectory.

    Args:
        place_order:   camera hole indices (0-8) in fill sequence
        pick_order:    camera hole indices (0-8) in pick sequence
        output_path:   save path for .png
        patient_id:    for title
        side:          "right" or "left" hand
        trajectory_px: list of (x,y) pixel coords of index_tip per frame
                       (from KinematicHistory.index_px). None entries skipped.
        frame_w/h:     original video dimensions for coordinate normalisation

    Camera → patient orientation mapping:
        Camera: 3 6 9    Patient: 1 2 3
                2 5 8  →          4 5 6
                1 4 7             7 8 9
    """

    # Camera idx → patient grid position (0=top-left, 8=bottom-right)
    CAM_TO_PATIENT = {0: 0, 3: 1, 6: 2,
                      1: 3, 4: 4, 7: 5,
                      2: 6, 5: 7, 8: 8}

    pad = 0.15
    hole_pos = {}
    for cam_idx, pat_idx in CAM_TO_PATIENT.items():
        col = pat_idx % 3
        row = pat_idx // 3
        nx  = pad + col / 2 * (1 - 2*pad)
        ny  = 1.0 - pad - row / 2 * (1 - 2*pad)
        hole_pos[cam_idx] = (nx, ny)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_board, ax_table) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.patch.set_facecolor("#1e1e1e")

    title = "9HPT Board — Pin Order"
    if patient_id:
        title += f"  |  {patient_id}"
    if side:
        title += f"  |  {side.capitalize()} hand"
    fig.suptitle(title, color="white", fontsize=12, y=1.01)

    # ── Board axis ────────────────────────────────────────────────────────────
    ax_board.set_facecolor("#1a4a6b")
    ax_board.set_xlim(0, 1)
    ax_board.set_ylim(0, 1)
    ax_board.set_aspect("equal")
    ax_board.axis("off")

    HOLE_R = 0.07

    # ── Trajectory overlay ────────────────────────────────────────────────────
    # Normalise pixel coords to board axes [0,1] using BOARD_HOLES_BBOX_PX
    if trajectory_px:
        bx1, by1, bx2, by2 = BOARD_HOLES_BBOX_PX
        bw = max(bx2 - bx1, 1)
        bh = max(by2 - by1, 1)

        traj_x, traj_y = [], []
        for pt in trajectory_px:
            if pt is None:
                # Break trajectory on None (hand not detected)
                if traj_x:
                    ax_board.plot(traj_x, traj_y,
                                  color="#00ffcc", linewidth=0.8,
                                  alpha=0.5, zorder=1)
                traj_x, traj_y = [], []
                continue
            px, py = pt
            # Normalise into board bbox, flip y
            nx = (px - bx1) / bw
            ny = 1.0 - (py - by1) / bh
            traj_x.append(np.clip(nx, 0, 1))
            traj_y.append(np.clip(ny, 0, 1))

        if traj_x:
            ax_board.plot(traj_x, traj_y,
                          color="#00ffcc", linewidth=0.8,
                          alpha=0.5, zorder=1, label="Trajectory")

    # ── Holes ─────────────────────────────────────────────────────────────────
    place_rank = {hole: rank+1 for rank, hole in enumerate(place_order)}
    pick_rank  = {hole: rank+1 for rank, hole in enumerate(pick_order)}

    for cam_idx, (x, y) in hole_pos.items():
        circle = plt.Circle((x, y), HOLE_R, color="white", zorder=2)
        ax_board.add_patch(circle)

        if cam_idx in place_rank:
            ax_board.text(x, y + 0.012, str(place_rank[cam_idx]),
                          ha="center", va="center",
                          fontsize=11, fontweight="bold",
                          color="black", zorder=3)

        if cam_idx in pick_rank:
            ax_board.text(x, y - HOLE_R - 0.045, str(pick_rank[cam_idx]),
                          ha="center", va="center",
                          fontsize=9, fontweight="bold",
                          color="#ff4444", zorder=3)

    ax_board.text(0.02, -0.06, "● black = placing order",
                  color="white", fontsize=7, transform=ax_board.transAxes)
    ax_board.text(0.02, -0.10, "● red = picking order",
                  color="#ff4444", fontsize=7, transform=ax_board.transAxes)
    if trajectory_px:
        ax_board.text(0.02, -0.14, "— teal = hand trajectory",
                      color="#00ffcc", fontsize=7, transform=ax_board.transAxes)

    # ── Table ─────────────────────────────────────────────────────────────────
    ax_table.set_facecolor("#1e1e1e")
    ax_table.axis("off")

    def label(h):
        if isinstance(h, int):
            pat = CAM_TO_PATIENT.get(h, h)
            return f"R{pat//3+1}C{pat%3+1}"
        return str(h)

    rows = []
    for rank in range(1, 10):
        p_hole = place_order[rank-1] if rank-1 < len(place_order) else "-"
        k_hole = pick_order[rank-1]  if rank-1 < len(pick_order)  else "-"
        rows.append([str(rank), label(p_hole), label(k_hole)])

    tbl = ax_table.table(
        cellText=rows,
        colLabels=["#", "Placing", "Picking"],
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#2d2d2d" if r > 0 else "#3a3a3a")
        cell.set_edgecolor("#555555")
        cell.set_text_props(color="white" if c != 2 else "#ff4444")
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")

    ax_table.set_title("Order summary", color="white", fontsize=9, pad=6)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Graphs] Board figure saved → {output_path}")