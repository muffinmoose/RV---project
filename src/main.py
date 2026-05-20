# src/main.py
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
from analysis.hole_tracker     import HoleTracker
from analysis.graphs           import KinematicHistory, save_graphs
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


def select_patient_interactive() -> Path:
    """Interactive mode — type patient ID, select video."""
    patients = list_patients(DATA_ROOT)
    if not patients:
        print(f"No patients found in {DATA_ROOT}")
        sys.exit(1)

    while True:
        patients = list_patients(DATA_ROOT)
        pid = input(f"Enter patient ID (from 001 to — {len(patients):03d}): ").strip()
        if pid in patients:
            break
        # Try partial match
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
        print(f"No videos found for {pid}")
        sys.exit(1)

    print(f"\nVideos for {pid}:")
    for i, v in enumerate(videos):
        star = " ★" if PRIMARY_CAMERA in v.name else ""
        print(f"  [{i}] {v.name}{star}")

    while True:
        try:
            idx = int(input(f"Select video (0-{len(videos)-1}): "))
            if 0 <= idx < len(videos):
                return videos[idx]
        except ValueError:
            pass
        print("  Invalid input")


# ── Results saving ────────────────────────────────────────────────────────────

def save_results(patient_id, video_name, events, results_dir, log):
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


# ── Core processing ───────────────────────────────────────────────────────────

def process_video(video_path: Path):
    """Process a single video file. Called for both interactive and batch mode."""

    patient_id = video_path.parent.name

    log = setup_logger(patient_id, video_path.name)
    log.info(f"Patient : {patient_id}")
    log.info(f"Video   : {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {video_path}")
        return

    actual_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"{frame_w}x{frame_h} @ {actual_fps:.1f}fps ({total_frames} frames)")

    cal = Calibrator()
    cal.load_intrinsics_for_video(str(video_path))
    if not cal.is_ready():
        log.error("Calibration not ready!")
        cap.release()
        return
    log.info("Calibration ready")

    out_dir   = Path(RESULTS_DIR) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem      = video_path.stem
    out_video = str(out_dir / f"{stem}_analyzed.mp4")
    out_graph = str(out_dir / f"{stem}_graphs.png")

    # Skip if already processed
    if Path(out_video).exists():
        log.info(f"Already processed — skipping. Delete output to reprocess.")
        cap.release()
        return

    detector = HandDetector()
    hk       = HandKalman(fps=actual_fps)
    mk       = MultiLandmarkKinematics(fps=actual_fps)
    pd       = PhaseDetector()
    ht       = HoleTracker()
    viz      = Visualizer(frame_w, frame_h, actual_fps, out_video)
    history  = KinematicHistory(fps=actual_fps)

    log.info("Processing video...")
    frame_idx      = 0
    show_landmarks = True
    ht_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not ht_initialized:
            ht.initialize(frame)
            ht_initialized = True

        detection    = detector.process(frame)
        smooth_px    = hk.update(detection)
        smooth_mm    = {k: cal.pixel_to_mm(v) for k, v in smooth_px.items()}
        states       = mk.update(smooth_mm)
        phase, event = pd.update(states, frame_idx)

        ht.update(frame)

        if event:
            log.info(f"Peg {event.peg_number} completed — {event.duration_s:.2f}s")

        history.record(frame_idx, states, phase)

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

    cap.release()
    detector.close()
    viz.close()

    log.info("Generating graphs...")
    save_graphs(history, out_graph, patient_id, video_path.name)

    if pd.events:
        save_results(patient_id, video_path.name, pd.events, RESULTS_DIR, log)
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