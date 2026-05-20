# main.py
# Entry point for 9HPT Kinematic Analysis
# Runs headless on server — outputs annotated video + CSV results

import cv2
import csv
import sys
from pathlib import Path

from config import (
    DATA_ROOT,
    RESULTS_DIR,
    PRIMARY_CAMERA,
    FPS,
)

from detection.hand_detector   import HandDetector
from processing.kalman_filter  import HandKalman
from processing.calibration    import Calibrator
from processing.kinematics     import MultiLandmarkKinematics
from analysis.phase_detector   import PhaseDetector
from utils.visualizer          import Visualizer
from utils.logger              import setup_logger


# ── Patient / video selection ─────────────────────────────────────────────────

def list_patients(data_root: str) -> list:
    root = Path(data_root)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def list_videos(patient_dir: Path, preferred_cam: str = PRIMARY_CAMERA) -> list:
    videos    = sorted(patient_dir.glob("*.mp4"))
    preferred = [v for v in videos if preferred_cam in v.name]
    others    = [v for v in videos if preferred_cam not in v.name]
    return preferred + others


def select_from_list(items: list, label: str) -> int:
    print(f"\n── {label} ──────────────────────────")
    for i, item in enumerate(items):
        star = " ★" if PRIMARY_CAMERA in str(item) else ""
        print(f"  [{i}] {item}{star}")
    print()
    while True:
        try:
            idx = int(input(f"Select {label} (0-{len(items)-1}): "))
            if 0 <= idx < len(items):
                return idx
            print(f"  Enter a number between 0 and {len(items)-1}")
        except ValueError:
            print("  Invalid input")


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(patient_id: str, video_name: str,
                 events: list, results_dir: str, log) -> Path:
    out_dir  = Path(results_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(video_name).stem}_results.csv"

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

    log.info(f"Results saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # 1. Select patient + video
    patients = list_patients(DATA_ROOT)
    if not patients:
        print(f"No patient folders found in {DATA_ROOT}")
        sys.exit(1)

    patient_id  = patients[select_from_list(patients, "Patient")]
    patient_dir = Path(DATA_ROOT) / patient_id

    videos = list_videos(patient_dir)
    if not videos:
        print(f"No .mp4 files found in {patient_dir}")
        sys.exit(1)

    video_path = videos[select_from_list([v.name for v in videos], "Video")]

    # 2. Logger
    log = setup_logger(patient_id, video_path.name)
    log.info(f"Patient : {patient_id}")
    log.info(f"Video   : {video_path.name}")

    # 3. Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    actual_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"{frame_w}x{frame_h} @ {actual_fps:.1f}fps ({total_frames} frames)")

    # 4. Calibration — loads K, D, H from JSON (no checkerboard needed)
    cal = Calibrator()
    cal.load_intrinsics_for_video(str(video_path))

    if not cal.is_ready():
        log.error("Calibration not ready — run compute_homography.py first!")
        sys.exit(1)

    log.info("Calibration ready — measurements will be in mm")

    # 5. Output video path
    out_dir   = Path(RESULTS_DIR) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = str(out_dir / f"{video_path.stem}_analyzed.mp4")

    # 6. Init modules
    detector = HandDetector()
    hk       = HandKalman(fps=actual_fps)
    mk       = MultiLandmarkKinematics(fps=actual_fps)
    pd       = PhaseDetector()
    viz      = Visualizer(frame_w, frame_h, actual_fps, out_video)

    log.info("Processing video...")
    frame_idx      = 0
    show_landmarks = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Pipeline ──────────────────────────────────────────────────────────
        detection = detector.process(frame)
        smooth_px = hk.update(detection)

        # Convert to mm using calibration
        smooth_mm = {k: cal.pixel_to_mm(v) for k, v in smooth_px.items()}

        states       = mk.update(smooth_mm)
        phase, event = pd.update(states, frame_idx)

        if event:
            log.info(f"Peg {event.peg_number} completed — {event.duration_s:.2f}s")

        # ── Write frame ───────────────────────────────────────────────────────
        viz.write_frame(
            frame=          frame,
            states=         states,
            phase=          phase,
            peg_count=      pd.peg_count,
            show_landmarks= show_landmarks,
            detection=      detection,
            detector=       detector,
            patient_id=     patient_id,
            frame_idx=      frame_idx,
            events=         pd.events,
        )

        if frame_idx % 100 == 0:
            log.info(f"Frame {frame_idx}/{total_frames}")

        frame_idx += 1

    # 7. Cleanup + save
    cap.release()
    detector.close()
    viz.close()

    if pd.events:
        save_results(patient_id, video_path.name, pd.events, RESULTS_DIR, log)
        log.info(f"Total pegs detected: {pd.peg_count}")
    else:
        log.warning("No peg events detected — check thresholds.")

    log.info("Done.")


if __name__ == "__main__":
    run()