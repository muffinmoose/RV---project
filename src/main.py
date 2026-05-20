# main.py
# Entry point for 9HPT Kinematic Analysis
#
# Flow:
#   1. Select patient folder
#   2. Select video file (camP_1 preferred)
#   3. Load calibration (auto from filename)
#   4. Compute homography from first frames
#   5. Process video frame by frame
#   6. Save results to CSV

import cv2
import csv
import os
import sys
from pathlib import Path

from config import (
    DATA_ROOT,
    CALIB_FILE,
    RESULTS_DIR,
    PRIMARY_CAMERA,
    FPS,
)

from detection.hand_detector   import HandDetector
from processing.kalman_filter  import HandKalman
from processing.calibration    import Calibrator
from processing.kinematics     import MultiLandmarkKinematics
from analysis.phase_detector   import PhaseDetector, Phase
from utils.visualizer          import Visualizer


# ── Patient / video selection ─────────────────────────────────────────────────

def list_patients(data_root: str) -> list:
    root = Path(data_root)
    patients = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return patients


def list_videos(patient_dir: Path, preferred_cam: str = PRIMARY_CAMERA) -> list:
    videos = sorted(patient_dir.glob("*.mp4"))
    # Sort: preferred camera first
    preferred = [v for v in videos if preferred_cam in v.name]
    others    = [v for v in videos if preferred_cam not in v.name]
    return preferred + others


def select_from_list(items: list, label: str) -> int:
    print(f"\n── {label} ──────────────────────────")
    for i, item in enumerate(items):
        marker = " ★" if i < len(items) and PRIMARY_CAMERA in str(item) else ""
        print(f"  [{i}] {item}{marker}")
    print()
    while True:
        try:
            idx = int(input(f"Select {label} (0-{len(items)-1}): "))
            if 0 <= idx < len(items):
                return idx
            print(f"  Enter a number between 0 and {len(items)-1}")
        except ValueError:
            print("  Invalid input")


# ── Homography bootstrap ──────────────────────────────────────────────────────

def bootstrap_homography(cap: cv2.VideoCapture, cal: Calibrator,
                          max_frames: int = 60) -> bool:
    """
    Try to find checkerboard in first max_frames frames.
    Returns True if homography was computed successfully.
    """
    print("[Main] Searching for checkerboard to compute homography...")
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if cal.compute_homography(frame):
            print(f"[Main] Homography found on frame {i}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
            return True
    print("[Main] WARNING: checkerboard not found in first "
          f"{max_frames} frames. Measurements will be in pixels only.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return False


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(patient_id: str, video_name: str,
                 events: list, results_dir: str) -> Path:
    out_dir = Path(results_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stem     = Path(video_name).stem
    out_path = out_dir / f"{stem}_results.csv"

    fieldnames = [
        "peg_number", "duration_s",
        "reach_frames", "grasp_frames",
        "transport_frames", "place_frames",
        "frame_start", "frame_end",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow({
                "peg_number":       ev.peg_number,
                "duration_s":       round(ev.duration_s, 3),
                "reach_frames":     ev.reach_frames,
                "grasp_frames":     ev.grasp_frames,
                "transport_frames": ev.transport_frames,
                "place_frames":     ev.place_frames,
                "frame_start":      ev.frame_start,
                "frame_end":        ev.frame_end,
            })

    print(f"[Main] Results saved → {out_path}")
    return out_path


# ── Main loop ─────────────────────────────────────────────────────────────────

def run():
    # 1. Select patient
    patients = list_patients(DATA_ROOT)
    if not patients:
        print(f"No patient folders found in {DATA_ROOT}")
        sys.exit(1)

    idx        = select_from_list(patients, "Patient")
    patient_id = patients[idx]
    patient_dir = Path(DATA_ROOT) / patient_id

    # 2. Select video
    videos = list_videos(patient_dir)
    if not videos:
        print(f"No .mp4 files found in {patient_dir}")
        sys.exit(1)

    vidx       = select_from_list([v.name for v in videos], "Video")
    video_path = videos[vidx]

    print(f"\n[Main] Patient : {patient_id}")
    print(f"[Main] Video   : {video_path.name}")

    # 3. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Main] ERROR: cannot open video {video_path}")
        sys.exit(1)

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Main] {frame_w}x{frame_h} @ {actual_fps:.1f}fps  "
          f"({total_frames} frames)")

    # 4. Calibration
    cal = Calibrator()
    cal.load_intrinsics_for_video(str(video_path))
    homography_ok = bootstrap_homography(cap, cal)

    # 5. Init pipeline modules
    detector = HandDetector()
    hk       = HandKalman(fps=actual_fps)
    mk       = MultiLandmarkKinematics(fps=actual_fps)
    pd       = PhaseDetector()
    viz      = Visualizer(frame_w, frame_h)

    show_landmarks = True
    paused         = False
    frame_idx      = 0

    print("\n[Main] Running — press Q to quit, L to toggle landmarks, "
          "SPACE to pause\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Main] End of video.")
                break

            # ── Pipeline ──────────────────────────────────────────────────────
            detection = detector.process(frame)

            smooth_px = hk.update(detection)

            # Convert to mm if calibration is ready, else keep pixels
            if homography_ok:
                smooth_mm = {
                    k: cal.pixel_to_mm(v) for k, v in smooth_px.items()
                }
            else:
                smooth_mm = smooth_px   # fallback: pixels

            states = mk.update(smooth_mm)

            phase, event = pd.update(states, frame_idx)

            if event:
                print(f"  Peg {event.peg_number} completed — "
                      f"{event.duration_s:.2f}s")

            frame_idx += 1

        # ── UI ────────────────────────────────────────────────────────────────
        key = viz.show(
            frame=          frame,
            states=         states if not paused else {},
            phase=          phase  if not paused else pd.current_phase,
            peg_count=      pd.peg_count,
            show_landmarks= show_landmarks,
            detection=      detection if not paused else None,
            detector=       detector,
            patient_id=     patient_id,
            frame_idx=      frame_idx,
            events=         pd.events,
        )

        # ── Key handling ──────────────────────────────────────────────────────
        if key == ord('q') or key == 27:    # Q or ESC
            print("[Main] Quit.")
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"[Main] Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord(' '):
            paused = not paused
            print(f"[Main] {'Paused' if paused else 'Resumed'}")

    # 6. Cleanup + save
    cap.release()
    detector.close()
    viz.close()

    if pd.events:
        save_results(patient_id, video_path.name, pd.events, RESULTS_DIR)
        print(f"\n[Main] Total pegs detected: {pd.peg_count}")
    else:
        print("\n[Main] No peg events detected — check thresholds.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()