# testing/analyze.py
# Agregirana analiza 9HPT — po pacientih, posnetkih, timestamps
#
# Generira v testing/analysis/:
#   all_patients.csv          — en zapis na posnetek
#   plot_01_pin_time.png      — mean pin time po pacientih
#   plot_02_distribution.png  — porazdelitev časov
#   plot_03_velocity_time.png — velocity vs pin time scatter
#   plot_04_fatigue.png       — trend utrujenosti (pin 2→8)
#   plot_05_velocity.png      — mean velocity po pacientih
#   plot_06_path.png          — path length distribucija
#   plot_07_learning.png      — učenje/utrujenost skozi čas (timestamp)
#   plot_08_within_patient.png— variabilnost znotraj pacienta

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import re
from datetime import datetime

RESULTS_DIR = Path("/workdir/testing/results")
OUT_DIR     = Path("/workdir/testing/analysis")
OUT_DIR.mkdir(exist_ok=True)

MIN_PEGS = 6

# ── Pomožne funkcije ──────────────────────────────────────────────────────────

def style(ax, title, xlabel, ylabel):
    """Stilizira matplotlib os — dark theme."""
    ax.set_facecolor("#2d2d2d")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555555")
    ax.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

def save_fig(fig, name):
    """Shrani figuro in zapre."""
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {path}")

def parse_timestamp(video_stem: str) -> datetime:
    """Izvleče timestamp iz imena videa npr. patient_003camP_1_20231005_14_13_43."""
    match = re.search(r'(\d{8}_\d{2}_\d{2}_\d{2})$', video_stem)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H_%M_%S")
    return None

# ── Zberi vse CSV-je ──────────────────────────────────────────────────────────
records = []
for csv_path in sorted(RESULTS_DIR.rglob("*_results.csv")):
    try:
        df = pd.read_csv(csv_path)
        if len(df) < MIN_PEGS:
            continue

        # Izpusti pin 1 — Kalman spike
        df_clean = df[df["peg_number"] > 1].copy()

        patient_id = csv_path.parts[-3]
        video_stem = csv_path.parts[-2]
        ts         = parse_timestamp(video_stem)

        # Utrujenost: razlika zadnji - prvi pin
        fatigue = df_clean.iloc[-1]["duration_s"] - df_clean.iloc[0]["duration_s"]

        records.append({
            "patient_id":      patient_id,
            "video":           video_stem,
            "timestamp":       ts,
            "n_pegs":          len(df_clean) + 1,
            "mean_duration_s": df_clean["duration_s"].mean(),
            "min_duration_s":  df_clean["duration_s"].min(),
            "max_duration_s":  df_clean["duration_s"].max(),
            "std_duration_s":  df_clean["duration_s"].std(),
            "first_pin_s":     df_clean.iloc[0]["duration_s"],
            "last_pin_s":      df_clean.iloc[-1]["duration_s"],
            "fatigue_delta_s": fatigue,   # pozitivno = upočasnitev, negativno = pohitritev
            "mean_velocity":   df_clean["mean_velocity"].mean(),
            "max_velocity":    df_clean["max_velocity"].max(),
            "mean_accel":      df_clean["mean_acceleration"].mean(),
            "total_path_m":    df_clean["path_length"].sum(),
        })
    except Exception as e:
        print(f"SKIP {csv_path}: {e}")

summary = pd.DataFrame(records)
summary.to_csv(OUT_DIR / "all_patients.csv", index=False)
print(f"Loaded {len(summary)} valid recordings from {summary['patient_id'].nunique()} patients")

# ── Plot 1: Mean pin time po pacientih ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#1e1e1e")
patient_means = summary.groupby("patient_id")["mean_duration_s"].mean().sort_values()
ax.barh(range(len(patient_means)), patient_means.values, color="#00bfff", alpha=0.8)
ax.set_yticks(range(len(patient_means)))
ax.set_yticklabels(patient_means.index, fontsize=8, color="white")
ax.axvline(patient_means.mean(), color="#ff4444", linewidth=1.5,
           linestyle="--", label=f"Mean: {patient_means.mean():.2f}s")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Mean pin time per patient", "Time (s)", "Patient")
save_fig(fig, "plot_01_pin_time.png")

# ── Plot 2: Porazdelitev časov ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1e1e1e")
all_durations = []
for csv_path in RESULTS_DIR.rglob("*_results.csv"):
    try:
        df = pd.read_csv(csv_path)
        df = df[df["peg_number"] > 1]
        all_durations.extend(df["duration_s"].tolist())
    except:
        pass
ax.hist(all_durations, bins=30, color="#ffcc00", alpha=0.8, edgecolor="#333333")
ax.axvline(np.mean(all_durations), color="#ff4444", linewidth=2,
           label=f"Mean: {np.mean(all_durations):.2f}s")
ax.axvline(np.median(all_durations), color="#00ffcc", linewidth=2,
           linestyle="--", label=f"Median: {np.median(all_durations):.2f}s")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Distribution of pin placement times", "Time (s)", "Count")
save_fig(fig, "plot_02_distribution.png")

# ── Plot 3: Velocity vs Pin Time ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1e1e1e")
ax.scatter(summary["mean_duration_s"], summary["mean_velocity"],
           alpha=0.7, color="#44ff88", s=50, edgecolors="#222222")
# Trend linija
mask = summary["mean_velocity"].notna() & summary["mean_duration_s"].notna()
z = np.polyfit(summary.loc[mask, "mean_duration_s"],
               summary.loc[mask, "mean_velocity"], 1)
p = np.poly1d(z)
x_line = np.linspace(summary["mean_duration_s"].min(),
                     summary["mean_duration_s"].max(), 100)
ax.plot(x_line, p(x_line), color="#ff4444", linewidth=1.5,
        linestyle="--", label="Trend")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Mean velocity vs Mean pin time", "Mean pin time (s)", "Mean velocity (m/s)")
save_fig(fig, "plot_03_velocity_time.png")

# ── Plot 4: Trend utrujenosti ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1e1e1e")
pin_times = {i: [] for i in range(2, 10)}
for csv_path in RESULTS_DIR.rglob("*_results.csv"):
    try:
        df = pd.read_csv(csv_path)
        if len(df) < MIN_PEGS:
            continue
        for _, row in df.iterrows():
            pn = int(row["peg_number"])
            if 2 <= pn <= 9:
                pin_times[pn].append(row["duration_s"])
    except:
        pass
pins  = sorted(pin_times.keys())
means = [np.mean(pin_times[p]) for p in pins]
stds  = [np.std(pin_times[p])  for p in pins]
ax.errorbar(pins, means, yerr=stds, fmt='o-', color="#ffcc00",
            ecolor="#ff8800", linewidth=2, capsize=5, markersize=8)
ax.fill_between(pins,
                [m-s for m,s in zip(means,stds)],
                [m+s for m,s in zip(means,stds)],
                color="#ffcc00", alpha=0.15)
style(ax, "Fatigue trend — mean time per pin (±1 std)", "Pin number", "Mean time (s)")
save_fig(fig, "plot_04_fatigue.png")

# ── Plot 5: Mean velocity po pacientih ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#1e1e1e")
patient_vel = summary.groupby("patient_id")["mean_velocity"].mean().sort_values()
ax.barh(range(len(patient_vel)), patient_vel.values, color="#bb88ff", alpha=0.8)
ax.set_yticks(range(len(patient_vel)))
ax.set_yticklabels(patient_vel.index, fontsize=8, color="white")
ax.axvline(patient_vel.mean(), color="#ff4444", linewidth=1.5,
           linestyle="--", label=f"Mean: {patient_vel.mean():.3f} m/s")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Mean wrist velocity per patient", "Velocity (m/s)", "Patient")
save_fig(fig, "plot_05_velocity.png")

# ── Plot 6: Path length distribucija ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1e1e1e")
ax.hist(summary["total_path_m"].dropna(), bins=20,
        color="#ff6eb4", alpha=0.8, edgecolor="#333333")
ax.axvline(summary["total_path_m"].mean(), color="#ff4444", linewidth=2,
           label=f"Mean: {summary['total_path_m'].mean():.2f}m")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Total hand path length distribution", "Path (m)", "Count")
save_fig(fig, "plot_06_path.png")

# ── Plot 7: Learning curve — čas skozi ponovitve ──────────────────────────────
# Za vsakega pacienta ki ima >1 posnetek: ali se izboljša skozi čas?
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#1e1e1e")

multi = summary[summary.groupby("patient_id")["patient_id"].transform("count") > 1].copy()
multi = multi.sort_values(["patient_id", "timestamp"])

colors = plt.cm.tab20(np.linspace(0, 1, multi["patient_id"].nunique()))
for (pid, grp), col in zip(multi.groupby("patient_id"), colors):
    grp = grp.sort_values("timestamp")
    x   = range(len(grp))
    ax.plot(x, grp["mean_duration_s"].values, 'o-',
            color=col, linewidth=1.5, markersize=6,
            label=pid, alpha=0.8)

ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=7,
          bbox_to_anchor=(1.01, 1), loc="upper left")
style(ax, "Learning curve — mean pin time across sessions",
      "Session number (chronological)", "Mean pin time (s)")
save_fig(fig, "plot_07_learning.png")

# ── Plot 8: Variabilnost znotraj pacienta ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1e1e1e")
fig.suptitle("Within-patient variability", color="white", fontsize=12)

# Levo: std deviation med posnetki
patient_std = summary.groupby("patient_id")["mean_duration_s"].std().dropna().sort_values()
axes[0].barh(range(len(patient_std)), patient_std.values, color="#ff8800", alpha=0.8)
axes[0].set_yticks(range(len(patient_std)))
axes[0].set_yticklabels(patient_std.index, fontsize=7, color="white")
style(axes[0], "Std dev of pin time (between sessions)", "Std (s)", "Patient")

# Desno: fatigue delta (last pin - first pin) po pacientih
fatigue_mean = summary.groupby("patient_id")["fatigue_delta_s"].mean().sort_values()
colors_fat   = ["#ff4444" if v > 0 else "#44ff88" for v in fatigue_mean.values]
axes[1].barh(range(len(fatigue_mean)), fatigue_mean.values, color=colors_fat, alpha=0.8)
axes[1].set_yticks(range(len(fatigue_mean)))
axes[1].set_yticklabels(fatigue_mean.index, fontsize=7, color="white")
axes[1].axvline(0, color="white", linewidth=1, alpha=0.5)
style(axes[1], "Fatigue delta (last pin - first pin)\nRed=slower, Green=faster",
      "Delta (s)", "Patient")

save_fig(fig, "plot_08_within_patient.png")

# ── Plot 9: Korelacija pot ↔ čas ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#1e1e1e")
ax.scatter(summary["total_path_m"], summary["mean_duration_s"],
           alpha=0.7, color="#00ffcc", s=50, edgecolors="#222222")
mask = summary["total_path_m"].notna() & summary["mean_duration_s"].notna()
z = np.polyfit(summary.loc[mask, "total_path_m"],
               summary.loc[mask, "mean_duration_s"], 1)
p = np.poly1d(z)
x_line = np.linspace(summary["total_path_m"].min(),
                     summary["total_path_m"].max(), 100)
ax.plot(x_line, p(x_line), color="#ff4444", linewidth=1.5,
        linestyle="--", label="Trend")
ax.legend(facecolor="#3d3d3d", labelcolor="white", fontsize=9)
style(ax, "Path length vs Mean pin time", "Total path (m)", "Mean pin time (s)")
save_fig(fig, "plot_09_path_vs_time.png")

# ── Plot 10: Boxplot časov po pacientih ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor("#1e1e1e")
all_data, labels = [], []
for pid in sorted(summary["patient_id"].unique()):
    vals = summary[summary["patient_id"] == pid]["mean_duration_s"].values
    if len(vals) > 0:
        all_data.append(vals)
        labels.append(pid)
bp = ax.boxplot(all_data, vert=False, patch_artist=True,
                medianprops=dict(color="#ff4444", linewidth=2))
for patch in bp["boxes"]:
    patch.set_facecolor("#00bfff")
    patch.set_alpha(0.7)
ax.set_yticks(range(1, len(labels)+1))
ax.set_yticklabels(labels, fontsize=7, color="white")
style(ax, "Pin time distribution per patient (boxplot)", "Time (s)", "Patient")
save_fig(fig, "plot_10_boxplot.png")

# ── Plot 11: Top 3 najhitrejši vs Top 3 najpočasnejši ────────────────────────
patient_mean = summary.groupby("patient_id")["mean_duration_s"].mean().sort_values()
top3_fast    = patient_mean.head(3).index.tolist()
top3_slow    = patient_mean.tail(3).index.tolist()
selected     = top3_fast + top3_slow

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#1e1e1e")
fig.suptitle("Top 3 fastest vs Top 3 slowest patients", color="white", fontsize=12)

for ax, group, title, color in zip(
        axes,
        [top3_fast, top3_slow],
        ["3 Fastest", "3 Slowest"],
        ["#44ff88", "#ff4444"]):
    pin_data = {i: [] for i in range(2, 10)}
    for pid in group:
        for csv_path in RESULTS_DIR.rglob("*_results.csv"):
            if pid not in str(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if len(df) < MIN_PEGS:
                    continue
                for _, row in df.iterrows():
                    pn = int(row["peg_number"])
                    if 2 <= pn <= 9:
                        pin_data[pn].append(row["duration_s"])
            except:
                pass
    pins  = sorted(pin_data.keys())
    means = [np.mean(pin_data[p]) if pin_data[p] else 0 for p in pins]
    ax.plot(pins, means, 'o-', color=color, linewidth=2, markersize=8)
    ax.set_facecolor("#2d2d2d")
    ax.set_title(f"{title}: {', '.join(group)}", color="white", fontsize=9)
    ax.set_xlabel("Pin number", color="white", fontsize=9)
    ax.set_ylabel("Mean time (s)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#555555")
    ax.grid(True, color="#444444", linewidth=0.5, alpha=0.5)

save_fig(fig, "plot_11_fast_vs_slow.png")

# ── Končni povzetek ───────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"COHORT SUMMARY ({len(summary)} recordings, {summary['patient_id'].nunique()} patients)")
print(f"{'='*50}")
print(f"  Mean pin time:     {summary['mean_duration_s'].mean():.2f} ± {summary['mean_duration_s'].std():.2f} s")
print(f"  Mean velocity:     {summary['mean_velocity'].mean():.3f} ± {summary['mean_velocity'].std():.3f} m/s")
print(f"  Mean path length:  {summary['total_path_m'].mean():.2f} ± {summary['total_path_m'].std():.2f} m")
print(f"  Mean fatigue Δ:    {summary['fatigue_delta_s'].mean():.3f} s (+ = slower at end)")
print(f"  Fastest patient:   {summary.groupby('patient_id')['mean_duration_s'].mean().idxmin()}")
print(f"  Slowest patient:   {summary.groupby('patient_id')['mean_duration_s'].mean().idxmax()}")