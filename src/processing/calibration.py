# processing/calibration.py
# Camera calibration + homography: converts pixel coordinates → real mm
#
# Flow:
#   1. Parse camera name from video filename (camP_0/1/2 → left/mid/right)
#   2. Load K, D, H from ukc_calibration.json for that specific camera
#   3. Every frame: pixel → undistort → homography → mm
#
# H is pre-computed once via compute_homography.py and stored in JSON.
# No checkerboard needed in patient videos.

import cv2
import json
import re
import numpy as np
from pathlib import Path
from typing import Optional

from config import (
    CALIB_FILE,
    CALIB_BOARD_COLS,
    CALIB_BOARD_ROWS,
    CAMERA_MAP,
    PRIMARY_CAMERA,
)


class Calibrator:
    """
    Handles all coordinate-space transformations:
        pixel (raw)  →  pixel (undistorted)  →  mm (real world)

    Loads K, D, H from ukc_calibration.json.
    H was pre-computed once from calibration photos via compute_homography.py.
    """

    def __init__(self):
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None

        self.camera_name: Optional[str] = None
        self.camera_id:   Optional[str] = None

        self._intrinsics_loaded = False
        self._homography_ready  = False

    # ── 1. Parse camera from filename ─────────────────────────────────────────

    @staticmethod
    def parse_camera_id(filename: str) -> str:
        match = re.search(r"(camP_\d+)", filename)
        if not match:
            raise ValueError(
                f"Could not parse camera ID from filename: {filename}\n"
                f"Expected pattern like 'camP_0', 'camP_1', 'camP_2'."
            )
        return match.group(1)

    # ── 2. Load K, D, H from JSON ─────────────────────────────────────────────

    def load_intrinsics_for_video(self, video_path: str) -> str:
        """
        Parse camera ID from video filename, load K, D, H from JSON.

        Args:
            video_path: full path or just filename of the video

        Returns:
            camera name string e.g. "mid"
        """
        filename         = Path(video_path).name
        self.camera_id   = self.parse_camera_id(filename)
        self.camera_name = CAMERA_MAP.get(self.camera_id)

        if self.camera_name is None:
            raise ValueError(
                f"Camera ID '{self.camera_id}' not found in CAMERA_MAP.\n"
                f"Known cameras: {list(CAMERA_MAP.keys())}"
            )

        self._load_from_json(self.camera_name)
        return self.camera_name

    def load_intrinsics_by_name(self, camera_name: str) -> None:
        """Load K, D, H directly by camera name ('left', 'mid', 'right')."""
        self._load_from_json(camera_name)

    def _load_from_json(self, camera_name: str) -> None:
        """Load K, D and H for a given camera from JSON."""
        calib_path = Path(CALIB_FILE)
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {CALIB_FILE}")

        with open(calib_path) as f:
            data = json.load(f)

        if camera_name not in data:
            raise KeyError(
                f"Camera '{camera_name}' not found in {CALIB_FILE}.\n"
                f"Available: {list(data.keys())}"
            )

        cam_data = data[camera_name]

        self.K = np.array(cam_data["cameraMatrix"],     dtype=np.float64)
        self.D = np.array(cam_data["distortionCoeffs"], dtype=np.float64)

        if self.K.shape != (3, 3):
            raise ValueError(f"cameraMatrix must be 3x3, got {self.K.shape}")

        self._intrinsics_loaded = True

        # Load H if available
        if "homography" in cam_data:
            self.H = np.array(cam_data["homography"], dtype=np.float64)
            self._homography_ready = True
            print(f"[Calibrator] Loaded K, D, H for camera: '{camera_name}'")
        else:
            self._homography_ready = False
            print(f"[Calibrator] WARNING: No homography in JSON for '{camera_name}'.")
            print(f"             Run compute_homography.py first!")

        print(f"             fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  "
              f"cx={self.K[0,2]:.2f}  cy={self.K[1,2]:.2f}")

    # ── 3. Undistortion ───────────────────────────────────────────────────────

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Remove lens distortion from a full frame."""
        self._check_intrinsics()
        return cv2.undistort(frame, self.K, self.D)

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Undistort an array of 2D points.
        Args:
            pts: shape (N, 2) float32
        Returns:
            shape (N, 2) float32
        """
        self._check_intrinsics()
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts_reshaped, self.K, self.D, P=self.K)
        return undist.reshape(-1, 2)

    # ── 4. Coordinate conversion ──────────────────────────────────────────────

    def pixel_to_mm(self, px: np.ndarray) -> np.ndarray:
        """
        Convert one pixel coordinate to mm.
        Args:
            px: shape (2,) — [x, y] raw distorted pixels
        Returns:
            shape (2,) — [x_mm, y_mm]
        """
        self._check_intrinsics()
        self._check_homography()

        pt_undist = self.undistort_points(px.reshape(1, 2)).reshape(2)
        pt_h      = np.array([pt_undist[0], pt_undist[1], 1.0], dtype=np.float64)
        mm_h      = self.H @ pt_h
        return (mm_h[:2] / mm_h[2]).astype(np.float32)

    def pixels_to_mm_batch(self, pts: np.ndarray) -> np.ndarray:
        """
        Convert array of pixel coordinates to mm — batch version.
        Args:
            pts: shape (N, 2)
        Returns:
            shape (N, 2) in mm
        """
        self._check_intrinsics()
        self._check_homography()

        undist = self.undistort_points(pts)
        ones   = np.ones((len(undist), 1), dtype=np.float32)
        pts_h  = np.hstack([undist, ones])
        mm_h   = (self.H @ pts_h.T).T
        return (mm_h[:, :2] / mm_h[:, 2:3]).astype(np.float32)

    # ── 5. Helpers ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._intrinsics_loaded and self._homography_ready

    def _check_intrinsics(self):
        if not self._intrinsics_loaded:
            raise RuntimeError(
                "Intrinsics not loaded. "
                "Call load_intrinsics_for_video() or load_intrinsics_by_name() first.")

    def _check_homography(self):
        if not self._homography_ready:
            raise RuntimeError(
                "Homography not ready. "
                "Run compute_homography.py first to save H to JSON.")