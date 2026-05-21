# src/main.py
# Entry point for 9HPT Kinematic Analysis
# Runs headless on server — outputs annotated video + graphs + CSV
#
# Usage:
#   Interactive:  python3 src/main.py
#   Batch:        python3 src/main.py /path/video1.mp4 /path/video2.mp4

import cv2
import csv
import sys
from pathlib import Path

from config import (
    DATA_ROOT,
    RESULTS_DIR,
    PRIMARY_CAMERA,
    FPS,
    BOARD_HOLES_BBOX_PX,
)

from detection.hand_detector   import HandDetector
from processing.kalman_filter  import HandKalman
from processing.calibration    import Calibrator
from processing.kinematics     import MultiLandmarkKinematics
from analysis.phase_detector   import PhaseDetector
from analysis.hole_tracker     import HoleTracker
from analysis.graphs           import KinematicHistory, save_graphs
from utils.visualizer          import Visualizer
from utils.logger              import setup_logger


# ── Patient / video selection ─────────────────────────────────────────────────

def list_patients(data_root: str) -> list:
    """Return sorted list of patient folder names."""
    root = Path(data_root)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def list_videos(patient_dir: Path, preferred_cam: str = PRIMARY_CAMERA) -> list:
    """Return only primary camera (camP_1) videos, sorted by name."""
    videos = sorted(patient_dir.glob("*.mp4"))
    return [v for v in videos if preferred_cam in v.name]


def select_patient_interactive() -> Path:
    """Interactive mode — type patient ID, select video."""
    patients = list_patients(DATA_ROOT)
    if not patients:
        print(f"No patients found in {DATA_ROOT}")
        sys.exit(1)

    while True:
        patients = list_patients(DATA_ROOT)
        pid = input(f"Enter patient ID (patient_001 — patient_{len(patients):03d}): ").strip()
        if pid in patients:
            break
        matches = [p for p in patients if pid in p]
        if len(matches) == 1:
            pid = matches[0]
            print(f"  → Matched: {pid}")
            break
        elif len(matches) > 1:
            print(f"  Multiple matches: {matches}")
        else:
            print(f"  Not found. Try again.")

    patient_dir = Path(DATA_ROOT) / pid
    videos = list_videos(patient_dir)
    if not videos:
        print(f"No camP_1 videos found for {pid}")
        sys.exit(1)

    print(f"\nVideos for {pid} (camP_1 only):")
    for i, v in enumerate(videos):
        print(f"  [{i}] {v.name}")

    while True:
        try:
            idx = int(input(f"Select video (0-{len(videos)-1}): "))
            if 0 <= idx < len(videos):
                return videos[idx]
        except ValueError:
            pass
        print("  Invalid input")


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(patient_id, video_stem, events, out_dir, log):
    """Save peg events to CSV in the video-specific output folder."""
    out_path = Path(out_dir) / f"{video_stem}_results.csv"

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


# ── Core processing ───────────────────────────────────────────────────────────

def process_video(video_path: Path):
    """
    Process a single video file.
    Output structure: results/<patient_id>/<video_stem>/
    Called for both interactive and batch mode.
    """
    patient_id = video_path.parent.name
    stem       = video_path.stem

    # Each video gets its own subfolder
    out_dir = Path(RESULTS_DIR) / patient_id / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(patient_id, video_path.name, log_dir=str(out_dir))
    log.info(f"Patient : {patient_id}")
    log.info(f"Video   : {video_path.name}")
    log.info(f"Output  : {out_dir}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return

    actual_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"{frame_w}x{frame_h} @ {actual_fps:.1f}fps ({total_frames} frames)")

    # Load calibration (K, D, H) for this camera
    cal = Calibrator()
    cal.load_intrinsics_for_video(str(video_path))
    if not cal.is_ready():
        log.error("Calibration not ready!")
        cap.release()
        return
    log.info("Calibration ready")

    # Output file paths
    out_video = str(out_dir / f"{stem}_analyzed.mp4")
    out_graph = str(out_dir / f"{stem}_graphs.png")

    # Init all modules
    detector = HandDetector()
    hk       = HandKalman(fps=actual_fps)
    mk       = MultiLandmarkKinematics(fps=actual_fps)
    pd       = PhaseDetector()
    ht       = HoleTracker()
    viz      = Visualizer(frame_w, frame_h, actual_fps, out_video)
    history  = KinematicHistory(fps=actual_fps)

    log.info("Processing video...")
    log.info(f"Active hand bbox: {BOARD_HOLES_BBOX_PX}")
    frame_idx      = 0
    show_landmarks = True
    ht_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort frame for accurate pixel coordinates
        frame = cal.undistort_frame(frame)

        # Initialize hole tracker on first frame
        if not ht_initialized:
            ht.initialize(frame)
            ht_initialized = True

        # ── Pipeline ──────────────────────────────────────────────────────────

        # Hand detection — activation waits for HoleTracker baseline
        # baseline_ready=True unlocks storage bbox trigger in HandDetector
        detection = detector.process(frame, baseline_ready=ht._baseline_set)

        smooth_px    = hk.update(detection)
        smooth_mm    = {k: cal.pixel_to_mm(v) for k, v in smooth_px.items()}
        states       = mk.update(smooth_mm)
        phase, event = pd.update(states, frame_idx)

        # Get all landmark coordinates for hole tracker sector guard
        # Only passed when active hand is detected
        all_lm_px = None
        if detection.all_landmarks is not None:
            h_f, w_f = frame.shape[:2]
            all_lm_px = [
                (lm.x * w_f, lm.y * h_f)
                for lm in detection.all_landmarks.landmark
            ]

        # Update hole filled/empty state
        ht.update(frame, all_lm_px)

        if event:
            log.info(f"Peg {event.peg_number} completed — {event.duration_s:.2f}s")

        history.record(frame_idx, states, phase)

        # Write annotated frame to output video
        viz.write_frame(
            frame=          frame,
            states=         states,
            phase=          phase,
            peg_count=      ht.filled_count,
            show_landmarks= show_landmarks,
            detection=      detection,
            detector=       detector,
            hole_tracker=   ht,
            patient_id=     patient_id,
            frame_idx=      frame_idx,
            events=         pd.events,
        )

        if frame_idx % 100 == 0:
            log.info(f"Frame {frame_idx}/{total_frames}")

        frame_idx += 1

    # ── Cleanup & save outputs ────────────────────────────────────────────────
    cap.release()
    detector.close()
    viz.close()

    log.info("Generating graphs...")
    save_graphs(history, out_graph, patient_id, video_path.name)

    if pd.events:
        save_results(patient_id, stem, pd.events, out_dir, log)
        log.info(f"Total pegs detected: {pd.peg_count}")
    else:
        log.warning("No peg events detected — check thresholds.")

    log.info(f"Done. Results in {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    args = sys.argv[1:]

    if args:
        # Batch mode — process all provided video paths
        print(f"Batch mode — {len(args)} video(s)")
        for path_str in args:
            video_path = Path(path_str)
            if not video_path.exists():
                print(f"  SKIP: not found — {path_str}")
                continue
            print(f"\n── Processing: {video_path.name} ──")
            process_video(video_path)
        print("\nBatch done.")
    else:
        # Interactive mode
        video_path = select_patient_interactive()
        process_video(video_path)


if __name__ == "__main__":
    run()