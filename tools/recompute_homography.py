# recompute_homography.py
# Lokalno na Windowsih — izračuna homografijo za vse 3 kamere
# Usage: python recompute_homography.py
# Nastavi CALIB_ROOT na mapo kjer so cam_left_resized2, cam_mid_resized2, cam_right_resized2

import cv2
import json
import numpy as np
from pathlib import Path

# ── NASTAVI TO ────────────────────────────────────────────────────────────────
CALIB_ROOT = "C:\\Users\\matej\\calibration_photos"  # ← pot do mape
CALIB_JSON = "C:\\Users\\matej\\calibration_photos\\ukc_calibration.json"  # ← original JSON
OUTPUT_JSON = "C:\\Users\\matej\\calibration_photos\\ukc_calibration_new.json"  # ← output
# ─────────────────────────────────────────────────────────────────────────────

CALIB_PHOTOS = {
    "left":  "cam_left_resized2",
    "mid":   "cam_mid_resized2",
    "right": "cam_right_resized2",
}

BOARD_COLS = 9
BOARD_ROWS = 6
SQUARE_MM  = 20.0

world_pts = np.array([
    [c * SQUARE_MM, r * SQUARE_MM]
    for r in range(BOARD_ROWS)
    for c in range(BOARD_COLS)
], dtype=np.float32)


def compute_H(camera_name, photo_dir, K, D):
    photos = sorted(Path(photo_dir).glob("*.jpg"))
    if not photos:
        photos = sorted(Path(photo_dir).glob("*.png"))
    if not photos:
        print(f"  ERROR: ni slik v {photo_dir}")
        return None

    print(f"\n[{camera_name}] {len(photos)} slik")

    best_H, best_inliers, best_photo = None, 0, None
    results = []

    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"  {p.name}: šahovnica NI najdena")
            continue

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_pts  = corners.reshape(-1, 2).astype(np.float32)

        img_pts_undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2), K, D, P=K
        ).reshape(-1, 2)

        H, mask = cv2.findHomography(img_pts_undist, world_pts, cv2.RANSAC, 3.0)
        if H is None:
            print(f"  {p.name}: homografija FAILED")
            continue

        inliers = int(mask.sum()) if mask is not None else 0
        results.append((inliers, p.name))
        print(f"  {p.name}: {inliers}/{len(world_pts)} inliers")

        if inliers > best_inliers:
            best_inliers = inliers
            best_H       = H
            best_photo   = p.name

    if best_H is None:
        print(f"  ERROR: homografija ni bila izračunana za {camera_name}")
        return None

    print(f"  ✓ Najboljša: {best_photo} ({best_inliers} inliers)")

    # Verifikacija — preveri spacing
    test_pts = np.array([[[0.0, 0.0]], [[SQUARE_MM, 0.0]]], dtype=np.float32)
    # Inverz H: mm → px
    H_inv = np.linalg.inv(best_H)
    p1 = cv2.perspectiveTransform(np.array([[[0.0, 0.0]]]), H_inv)[0][0]
    p2 = cv2.perspectiveTransform(np.array([[[SQUARE_MM, 0.0]]]), H_inv)[0][0]
    px_per_square = np.linalg.norm(p2 - p1)
    print(f"  Verifikacija: {SQUARE_MM}mm = {px_per_square:.1f}px → {SQUARE_MM/px_per_square:.4f} mm/px")

    return best_H


def main():
    with open(CALIB_JSON) as f:
        data = json.load(f)

    for cam_name, subdir in CALIB_PHOTOS.items():
        print(f"\n{'='*50}")
        print(f"Kamera: {cam_name}")
        print(f"{'='*50}")

        photo_dir = Path(CALIB_ROOT) / subdir
        if not photo_dir.exists():
            print(f"  ERROR: mapa ne obstaja: {photo_dir}")
            continue

        if cam_name not in data:
            print(f"  ERROR: {cam_name} ni v JSON")
            continue

        K = np.array(data[cam_name]["cameraMatrix"],     dtype=np.float64)
        D = np.array(data[cam_name]["distortionCoeffs"], dtype=np.float64)

        H = compute_H(cam_name, photo_dir, K, D)
        if H is not None:
            data[cam_name]["homography"] = H.tolist()
            print(f"  → Homografija shranjena za '{cam_name}'")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Novo JSON shranjeno → {OUTPUT_JSON}")
    print("Preveri vrednosti, potem kopiraj na server:")
    print(f"  scp -P 3322 {OUTPUT_JSON} matejh@192.168.32.141:/media/FastDataMama/zigab/calibration_photos/ukc_calibration.json")


if __name__ == "__main__":
    main()