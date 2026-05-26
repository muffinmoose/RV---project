# analysis/graphs.py
# Generates all output figures for a processed 9HPT video session.
#
# Exported functions:
#   save_graphs(history, output_path, patient_id, video_name, events, hand_side)
#       → kinematic graphs + summary stats table
#   save_board_figure(...)
#       → 9HPT board with pin order + trajectory in mm coords

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Optional, Tuple

from analysis.phase_detector import Phase, PHASE_COLORS
from config import LEFT_HOLES_PX, RIGHT_HOLES_PX, BOARD_HOLES_BBOX_PX


# ── Phase colors ──────────────────────────────────────────────────────────────

PHASE_COLORS_MPL = {
    Phase.IDLE:      (0.4, 0.4, 0.4, 0.15),
    Phase.PLACING:   (0.0, 0.8, 0.0, 0.20),
    Phase.RETURNING: (0.6, 0.0, 0.8, 0.20),
}

PHASE_LABEL_COLORS = {
    Phase.IDLE:      "#666666",
    Phase.PLACING:   "#00bb00",
    Phase.RETURNING: "#9900cc",
}

WARMUP_FRAMES = 15


# ── Data container ────────────────────────────────────────────────────────────

class KinematicHistory:
    """
    Accumulates per-frame kinematic + pixel data during video processing.
    Call record() each frame from main.py.
    """

    def __init__(self, fps: float):
        self.fps = fps
        self.times:        List[float] = []
        self.vel_mag:      List[float] = []
        self.vel_x:        List[float] = []
        self.vel_y:        List[float] = []
        self.accel:        List[float] = []
        self.phases:       List[Phase] = []
        self.path_lengths: List[float] = []
        self.index_px:     List[Optional[Tuple[float, float]]] = []
        self._baseline_frame: int = -1

    def set_baseline_frame(self, frame_idx: int):
        """Call once from main.py when HoleTracker baseline is set."""
        if self._baseline_frame < 0:
            self._baseline_frame = frame_idx

    def record(self, frame_idx: int, states: dict, phase: Phase,
               index_tip_px: Optional[Tuple[float, float]] = None):
        """Call once per frame from main.py. Zabeleži kinematiko in fazo."""
        t     = frame_idx / self.fps
        wrist = states.get("wrist")

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


# ── Summary stats ─────────────────────────────────────────────────────────────

def _compute_summary(history: "KinematicHistory",
                     events: list,
                     hand_side: str) -> dict:
    """
    Izračuna povzetne statistike iz KinematicHistory in PegEvent liste.
    Uporablja se za tabelo desno od grafov.
    """
    v_mag  = np.array(history.vel_mag)
    phases = history.phases
    fps    = history.fps

    placing_frames   = sum(1 for p in phases if p == Phase.PLACING)
    returning_frames = sum(1 for p in phases if p == Phase.RETURNING)
    placing_time_s   = placing_frames  / fps
    returning_time_s = returning_frames / fps

    path_total = history.path_lengths[-1] if history.path_lengths else 0.0

    active_vel = v_mag[v_mag > 0.001]
    mean_vel   = float(np.mean(active_vel)) if len(active_vel) > 0 else 0.0
    max_vel    = float(np.max(active_vel))  if len(active_vel) > 0 else 0.0

    n_pegs = len(events)
    if n_pegs > 0:
        durations             = [ev.duration_s for ev in events]
        mean_dur              = float(np.mean(durations))
        min_dur               = float(np.min(durations))
        max_dur               = float(np.max(durations))
        pins_per_sec_placing  = n_pegs / placing_time_s   if placing_time_s  > 0 else 0.0
        pins_per_sec_returning= n_pegs / returning_time_s if returning_time_s > 0 else 0.0
    else:
        mean_dur = min_dur = max_dur = 0.0
        pins_per_sec_placing = pins_per_sec_returning = 0.0

    # FFT tremor peak
    peak_freq = peak_pow = None
    signal = v_mag.copy() - np.mean(v_mag)
    if len(signal) > 10:
        fft_vals  = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
        mask  = fft_freqs <= 25.0
        freqs = fft_freqs[mask]
        power = fft_vals[mask]
        if power.max() > 0:
            power = power / power.max()
        tremor_mask = (freqs >= 4) & (freqs <= 12)
        if tremor_mask.any():
            peak_freq = float(freqs[tremor_mask][np.argmax(power[tremor_mask])])
            peak_pow  = float(power[tremor_mask].max())

    return {
        "hand":                hand_side.capitalize(),
        "n_pegs":              n_pegs,
        "placing_time_s":      placing_time_s,
        "returning_time_s":    returning_time_s,
        "mean_pin_time_s":     mean_dur,
        "min_pin_time_s":      min_dur,
        "max_pin_time_s":      max_dur,
        "pins_per_sec_place":  pins_per_sec_placing,
        "pins_per_sec_pick":   pins_per_sec_returning,
        "path_total_m":        path_total,
        "mean_vel_ms":         mean_vel,
        "max_vel_ms":          max_vel,
        "tremor_peak_hz":      peak_freq,
        "tremor_peak_pow":     peak_pow,
    }


def _draw_summary_table(ax, stats: dict):
    """
    Nariše tabelo povzetkov na matplotlib os (desni stolpec figure).
    Sekcije: PLACING, RETURNING, KINEMATICS, TREMOR.
    """
    ax.set_facecolor("#1e1e1e")
    ax.axis("off")

    def fmt(val, decimals=2, unit=""):
        if val is None:
            return "N/A"
        return f"{val:.{decimals}f}{unit}"

    rows = [
        ("Hand",             stats["hand"]),
        ("Pegs detected",    str(stats["n_pegs"]) + " / 9"),
        ("── PLACING ──",    ""),
        ("  Total time",     fmt(stats["placing_time_s"],  1, " s")),
        ("  Mean per pin",   fmt(stats["mean_pin_time_s"], 2, " s")),
        ("  Min per pin",    fmt(stats["min_pin_time_s"],  2, " s")),
        ("  Max per pin",    fmt(stats["max_pin_time_s"],  2, " s")),
        ("  Pins / sec",     fmt(stats["pins_per_sec_place"], 2, " /s")),
        ("── RETURNING ──",  ""),
        ("  Total time",     fmt(stats["returning_time_s"], 1, " s")),
        ("  Pins / sec",     fmt(stats["pins_per_sec_pick"], 2, " /s")),
        ("── KINEMATICS ──", ""),
        ("  Path length",    fmt(stats["path_total_m"],  2, " m")),
        ("  Mean velocity",  fmt(stats["mean_vel_ms"],   3, " m/s")),
        ("  Max velocity",   fmt(stats["max_vel_ms"],    3, " m/s")),
        ("── TREMOR ──",     ""),
        ("  Peak freq",      fmt(stats["tremor_peak_hz"], 1, " Hz") if stats["tremor_peak_hz"] else "N/A"),
        ("  Peak power",     fmt(stats["tremor_peak_pow"], 2)        if stats["tremor_peak_pow"] else "N/A"),
    ]

    y_start, dy = 0.97, 0.054
    for i, (lbl, val) in enumerate(rows):
        y = y_start - i * dy
        if lbl.startswith("──"):
            ax.text(0.02, y, lbl, transform=ax.transAxes,
                    fontsize=8, color="#aaaaaa", fontweight="bold", va="top")
        else:
            ax.text(0.04, y, lbl, transform=ax.transAxes,
                    fontsize=8.5, color="#cccccc", va="top")
            ax.text(0.98, y, val, transform=ax.transAxes,
                    fontsize=8.5, color="#ffffff", va="top",
                    ha="right", fontweight="bold")

    ax.set_title("Summary", color="white", fontsize=9, pad=6)


# ── Kinematic graphs ──────────────────────────────────────────────────────────

def save_graphs(history: "KinematicHistory",
                output_path: str,
                patient_id:  str  = "",
                video_name:  str  = "",
                events:      list = None,
                hand_side:   str  = "") -> None:
    """
    Shrani kinematične grafe (velocity, acceleration, path length, FFT tremor)
    + tabelo povzetkov desno.

    Args:
        history:     KinematicHistory iz main.py
        output_path: pot za .png
        patient_id:  za naslov
        video_name:  za naslov
        events:      seznam PegEvent iz PhaseDetector
        hand_side:   "right" ali "left" iz HoleTracker
    """
    if events is None:
        events = []

    t      = np.array(history.times)
    v_mag  = np.array(history.vel_mag)
    v_x    = np.array(history.vel_x)
    v_y    = np.array(history.vel_y)
    a_mag  = np.array(history.accel)
    phases = history.phases

    if len(t) == 0:
        print("[Graphs] No data to plot.")
        return

    # Layout: 4 vrstice grafov levo + tabela desno
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#1e1e1e")

    gs  = GridSpec(4, 2, figure=fig, width_ratios=[3, 1],
                   hspace=0.35, wspace=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax_s= fig.add_subplot(gs[:, 1])   # tabela — cela desna stran

    title = "Kinematic Analysis"
    if patient_id:
        title += f" — {patient_id}"
    if video_name:
        title += f"\n{video_name}"
    fig.suptitle(title, color="white", fontsize=13, y=0.99)

    ax2.sharex(ax1)
    ax3.sharex(ax1)
    _draw_phase_bands([ax1, ax2, ax3], t, phases)

    # Plot 1: Velocity
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

    # Plot 2: Acceleration
    ax2.set_facecolor("#2d2d2d")
    ax2.plot(t, a_mag, color="#ff4444", linewidth=1.5, label="|a|", zorder=3)
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("Acceleration (m/s²)", color="white")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#555555")
    ax2.legend(loc="upper right", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax2.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # Plot 3: Path length
    ax3.set_facecolor("#2d2d2d")
    path = np.array(history.path_lengths)
    ax3.plot(t, path, color="#44ff88", linewidth=1.5, label="Path length (m)", zorder=3)
    ax3.set_ylabel("Path length (m)", color="white")
    ax3.set_xlabel("Time (s)", color="white")
    ax3.tick_params(colors="white")
    ax3.spines[:].set_color("#555555")
    ax3.legend(loc="upper left", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax3.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # Plot 4: FFT tremor
    ax4.set_facecolor("#2d2d2d")
    signal = v_mag.copy() - np.mean(v_mag)
    if len(signal) > 10:
        fft_vals  = np.abs(np.fft.rfft(signal))
        fft_freqs = np.fft.rfftfreq(len(signal), d=1.0/history.fps)
        mask  = fft_freqs <= 25.0
        freqs = fft_freqs[mask]
        power = fft_vals[mask]
        if power.max() > 0:
            power = power / power.max()
        ax4.plot(freqs, power, color="#bb88ff", linewidth=1.5, label="FFT power", zorder=3)
        ax4.axvspan(4, 12, color=(0.8, 0.3, 0.3, 0.15), zorder=1,
                    label="Tremor band (4-12 Hz)")
        tremor_mask = (freqs >= 4) & (freqs <= 12)
        if tremor_mask.any():
            peak_freq = freqs[tremor_mask][np.argmax(power[tremor_mask])]
            peak_pow  = power[tremor_mask].max()
            ax4.axvline(peak_freq, color="#ff4444", linewidth=1.0,
                        linestyle="--", alpha=0.8)
            ax4.text(peak_freq + 0.2, peak_pow * 0.95,
                     f"peak: {peak_freq:.1f} Hz", color="#ff4444", fontsize=8)
    ax4.set_ylabel("Normalised power", color="white")
    ax4.set_xlabel("Frequency (Hz)", color="white")
    ax4.tick_params(colors="white")
    ax4.spines[:].set_color("#555555")
    ax4.legend(loc="upper right", facecolor="#3d3d3d", labelcolor="white", fontsize=8)
    ax4.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

    # Phase legenda
    patches = [mpatches.Patch(color=PHASE_LABEL_COLORS[p], label=p.name) for p in Phase]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#2d2d2d", labelcolor="white",
               fontsize=8, bbox_to_anchor=(0.38, 0.0))

    # Tabela povzetkov
    stats = _compute_summary(history, events, hand_side)
    _draw_summary_table(ax_s, stats)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Graphs] Saved → {output_path}")


def _draw_phase_bands(axes, t: np.ndarray, phases: List[Phase]):
    """Nariše barvne vertikalne fazne pasove na seznam osi."""
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

def save_board_figure(place_order:   List[int],
                      pick_order:    List[int],
                      output_path:   str,
                      patient_id:    str = "",
                      side:          str = "right",
                      trajectory_px: Optional[List[Optional[Tuple[float,float]]]] = None,
                      frame_w:       int = 640,
                      frame_h:       int = 480,
                      calibrator=    None) -> None:
    """
    9HPT board vizualizacija v mm koordinatah via calibrator.
    Trajektorija index_mcp v istih mm koordinatah.
    Storage krog med obema sektorjema.
    """
    from config import LEFT_HOLES_PX, RIGHT_HOLES_PX, HOLE_SPACING_MM

    def holes_to_mm(holes_px, cal):
        """Pretvori pixel koordinate luknjic v mm. Fallback na px če ni cal."""
        if cal is not None and cal.is_ready():
            return [cal.pixel_to_mm(np.array(pt, dtype=np.float32)) for pt in holes_px]
        return [np.array(pt, dtype=np.float32) for pt in holes_px]

    left_mm  = holes_to_mm(LEFT_HOLES_PX,  calibrator)
    right_mm = holes_to_mm(RIGHT_HOLES_PX, calibrator)
    all_holes_mm = left_mm + right_mm

    active_mm   = left_mm  if side == "right" else right_mm
    inactive_mm = right_mm if side == "right" else left_mm

    # Pretvori trajektorijo v mm
    traj_mm = []
    if trajectory_px and calibrator is not None and calibrator.is_ready():
        for pt in trajectory_px:
            if pt is None:
                traj_mm.append(None)
            else:
                mm = calibrator.pixel_to_mm(np.array(pt, dtype=np.float32))
                traj_mm.append((float(mm[0]), float(mm[1])))
    elif trajectory_px:
        traj_mm = trajectory_px

    # Bounds
    all_x = [pt[0] for pt in all_holes_mm]
    all_y = [pt[1] for pt in all_holes_mm]
    if traj_mm:
        tx = [pt[0] for pt in traj_mm if pt is not None]
        ty = [pt[1] for pt in traj_mm if pt is not None]
        if tx:
            all_x += tx
            all_y += ty

    pad_mm = HOLE_SPACING_MM * 1.5
    x_min, x_max = min(all_x) - pad_mm, max(all_x) + pad_mm
    y_min, y_max = min(all_y) - pad_mm, max(all_y) + pad_mm

    fig, (ax_board, ax_table) = plt.subplots(
        1, 2, figsize=(13, 6),
        gridspec_kw={"width_ratios": [2.5, 1]}
    )
    fig.patch.set_facecolor("#1e1e1e")

    title = "9HPT Board — Pin Order"
    if patient_id:
        title += f"  |  {patient_id}"
    if side:
        title += f"  |  {side.capitalize()} hand"
    fig.suptitle(title, color="white", fontsize=12, y=1.01)

    ax_board.set_facecolor("#1a4a6b")
    ax_board.set_xlim(x_min, x_max)
    ax_board.set_ylim(y_max, y_min)   # y obrnjen
    ax_board.set_aspect("equal")
    ax_board.set_xlabel("x (mm)", color="white", fontsize=8)
    ax_board.set_ylabel("y (mm)", color="white", fontsize=8)
    ax_board.tick_params(colors="white", labelsize=7)
    ax_board.spines[:].set_color("#555555")

    HOLE_R_MM = HOLE_SPACING_MM * 0.28

    # Trajektorija
    if traj_mm:
        seg_x, seg_y = [], []
        for pt in traj_mm:
            if pt is None:
                if seg_x:
                    ax_board.plot(seg_x, seg_y, color="#00ffcc",
                                  linewidth=0.8, alpha=0.5, zorder=1)
                seg_x, seg_y = [], []
            else:
                seg_x.append(pt[0])
                seg_y.append(pt[1])
        if seg_x:
            ax_board.plot(seg_x, seg_y, color="#00ffcc",
                          linewidth=0.8, alpha=0.5, zorder=1)

    # Neaktivne luknjice
    for pt in inactive_mm:
        ax_board.add_patch(plt.Circle((pt[0], pt[1]), HOLE_R_MM * 0.6,
                                      color="#446688", zorder=2))

    # Aktivne luknjice z vrstnim redom
    place_rank = {idx: rank+1 for rank, idx in enumerate(place_order)}
    pick_rank  = {idx: rank+1 for rank, idx in enumerate(pick_order)}

    for i, pt in enumerate(active_mm):
        ax_board.add_patch(plt.Circle((pt[0], pt[1]), HOLE_R_MM,
                                      color="white", zorder=3))
        if i in place_rank:
            ax_board.text(pt[0], pt[1] + HOLE_R_MM * 0.15,
                          str(place_rank[i]),
                          ha="center", va="center",
                          fontsize=10, fontweight="bold",
                          color="black", zorder=4)
        if i in pick_rank:
            ax_board.text(pt[0], pt[1] + HOLE_R_MM + 3.5,
                          str(pick_rank[i]),
                          ha="center", va="center",
                          fontsize=8, fontweight="bold",
                          color="#ff4444", zorder=4)

    # Storage krog
    all_x_h = [pt[0] for pt in all_holes_mm]
    all_y_h = [pt[1] for pt in all_holes_mm]
    sx = (min(all_x_h) + max(all_x_h)) / 2
    sy = (min(all_y_h) + max(all_y_h)) / 2
    sr = HOLE_SPACING_MM * 1.0
    ax_board.add_patch(plt.Circle((sx, sy), sr, color="#9932cc",
                                   alpha=0.4, fill=True, zorder=2))
    ax_board.add_patch(plt.Circle((sx, sy), sr, color="white",
                                   fill=False, linewidth=1.5, zorder=2))

    ax_board.text(0.02, -0.06, "● črna = placing order",
                  color="white", fontsize=7, transform=ax_board.transAxes)
    ax_board.text(0.02, -0.10, "● rdeča = picking order",
                  color="#ff4444", fontsize=7, transform=ax_board.transAxes)
    if traj_mm:
        ax_board.text(0.02, -0.14, "— teal = MCP trajektorija (členek kazalca)",
                      color="#00ffcc", fontsize=7, transform=ax_board.transAxes)

    # Tabela vrstnega reda
    ax_table.set_facecolor("#1e1e1e")
    ax_table.axis("off")

    CAM_TO_PATIENT = {0: 0, 3: 1, 6: 2,
                      1: 3, 4: 4, 7: 5,
                      2: 6, 5: 7, 8: 8}

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