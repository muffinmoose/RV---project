# compute_homography.py
# ONE-TIME SCRIPT — run once per camera to compute homography H
# and save it into ukc_calibration.json permanently.
#
# After running this, calibration.py will load H directly from JSON
# and never need to search for a checkerboard in patient videos.
#
# Run with:
#   python3 src/compute_homography.py

import cv2
import json
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CALIB_JSON  = "/media/FastDataMama/zigab/calibration_photos/ukc_calibration.json"

CALIB_PHOTOS = {
    "left":  "/media/FastDataMama/zigab/calibration_photos/cam_left_resized2",
    "mid":   "/media/FastDataMama/zigab/calibration_photos/cam_mid_resized2",
    "right": "/media/FastDataMama/zigab/calibration_photos/cam_right_resized2",
}

# Checkerboard: 9x6 inner corners, 20x20mm squares
BOARD_COLS   = 9
BOARD_ROWS   = 6
SQUARE_MM    = 20.0

# ── World points (same for all cameras) ───────────────────────────────────────

world_pts = np.array([
    [c * SQUARE_MM, r * SQUARE_MM]
    for r in range(BOARD_ROWS)
    for c in range(BOARD_COLS)
], dtype=np.float32)


# ── Per-camera processing ─────────────────────────────────────────────────────

def compute_H_for_camera(camera_name: str, photo_dir: str,
                          K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Find checkerboard in calibration photos, compute and return homography H.
    Uses all photos and picks the one with the most inliers.
    """
    photos = sorted(Path(photo_dir).glob("*.jpg"))
    if not photos:
        photos = sorted(Path(photo_dir).glob("*.png"))
    if not photos:
        raise FileNotFoundError(f"No images found in {photo_dir}")

    print(f"\n[{camera_name}] Found {len(photos)} calibration photos")

    best_H       = None
    best_inliers = 0
    best_photo   = None

    for photo_path in photos:
        img  = cv2.imread(str(photo_path))
        if img is None:
            print(f"  WARNING: could not read {photo_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            (BOARD_COLS, BOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                  | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print(f"  {photo_path.name} — checkerboard NOT found")
            continue

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_pts  = corners.reshape(-1, 2).astype(np.float32)

        # Undistort points using camera intrinsics
        img_pts_undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2), K, D, P=K
        ).reshape(-1, 2)

        # Compute homography
        H, mask = cv2.findHomography(img_pts_undist, world_pts, cv2.RANSAC, 3.0)
        if H is None:
            print(f"  {photo_path.name} — homography FAILED")
            continue

        inliers = int(mask.sum()) if mask is not None else 0
        print(f"  {photo_path.name} — inliers: {inliers}/{len(world_pts)}")

        if inliers > best_inliers:
            best_inliers = inliers
            best_H       = H
            best_photo   = photo_path.name

    if best_H is None:
        raise RuntimeError(
            f"Could not compute homography for camera '{camera_name}'. "
            "No checkerboard found in any photo.")

    print(f"  ✓ Best: {best_photo} with {best_inliers} inliers")
    return best_H


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load existing calibration JSON
    with open(CALIB_JSON) as f:
        data = json.load(f)

    for camera_name, photo_dir in CALIB_PHOTOS.items():
        print(f"\n{'='*50}")
        print(f"Processing camera: {camera_name}")
        print(f"{'='*50}")

        if camera_name not in data:
            print(f"WARNING: '{camera_name}' not found in JSON, skipping.")
            continue

        K = np.array(data[camera_name]["cameraMatrix"],     dtype=np.float64)
        D = np.array(data[camera_name]["distortionCoeffs"], dtype=np.float64)

        H = compute_H_for_camera(camera_name, photo_dir, K, D)

        # Save H into JSON as a list of lists
        data[camera_name]["homography"] = H.tolist()
        print(f"  → Homography saved for '{camera_name}'")

    # Write updated JSON back
    with open(CALIB_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Done — homography saved to {CALIB_JSON}")
    print("You can now run main.py — no checkerboard needed in patient videos.")


if __name__ == "__main__":
    main()