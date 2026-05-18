# processing/calibration.py
# Camera calibration + homography: converts pixel coordinates → real mm
#
# Flow:
#   1. Parse camera name from video filename (camP_0/1/2 → left/mid/right)
#   2. Load K, D from ukc_calibration.json for that specific camera
#   3. Compute homography H from first video frame (checkerboard)
#   4. Every frame: pixel → undistort → homography → mm
#
# Usage:
#   cal = Calibrator()
#   cal.load_intrinsics_for_video("patient_003camP_1_20231005_14_13_43.mp4")
#   cal.compute_homography(first_frame)
#   mm_xy = cal.pixel_to_mm(np.array([x, y]))

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
    CALIB_SQUARE_MM,
    CAMERA_MAP,
    PRIMARY_CAMERA,
)


class Calibrator:
    """
    Handles all coordinate-space transformations:
        pixel (raw)  →  pixel (undistorted)  →  mm (real world)

    One Calibrator instance per session.
    Call load_intrinsics_for_video() each time you open a new video —
    it parses the camera name from the filename and loads the correct K, D.
    Homography H is computed once from the first frame and reused.
    """

    def __init__(self):
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None

        self.camera_name: Optional[str] = None   # e.g. "mid"
        self.camera_id:   Optional[str] = None   # e.g. "camP_1"

        self._intrinsics_loaded = False
        self._homography_ready  = False

    # ── 1. Parse camera from filename ─────────────────────────────────────────

    @staticmethod
    def parse_camera_id(filename: str) -> str:
        """
        Extract camP_X from a filename like:
            patient_003camP_1_20231005_14_13_43.mp4

        Returns e.g. "camP_1"
        Raises ValueError if no camP pattern found.
        """
        match = re.search(r"(camP_\d+)", filename)
        if not match:
            raise ValueError(
                f"Could not parse camera ID from filename: {filename}\n"
                f"Expected pattern like 'camP_0', 'camP_1', 'camP_2'."
            )
        return match.group(1)

    # ── 2. Intrinsics ─────────────────────────────────────────────────────────

    def load_intrinsics_for_video(self, video_path: str) -> str:
        """
        Parse camera ID from video filename, load matching K and D from JSON.

        Args:
            video_path: full path or just filename of the video

        Returns:
            camera name string e.g. "mid"
        """
        filename = Path(video_path).name
        self.camera_id   = self.parse_camera_id(filename)
        self.camera_name = CAMERA_MAP.get(self.camera_id)

        if self.camera_name is None:
            raise ValueError(
                f"Camera ID '{self.camera_id}' not found in CAMERA_MAP.\n"
                f"Known cameras: {list(CAMERA_MAP.keys())}"
            )

        self._load_intrinsics_by_name(self.camera_name)
        return self.camera_name

    def load_intrinsics_by_name(self, camera_name: str) -> None:
        """
        Load K, D directly by camera name ('left', 'mid', 'right').
        Use this if you don't have a video filename yet.
        """
        self._load_intrinsics_by_name(camera_name)

    def _load_intrinsics_by_name(self, camera_name: str) -> None:
        """Internal: load K, D for a given camera name from JSON."""
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

        # Matches your actual JSON keys: "cameraMatrix" and "distortionCoeffs"
        self.K = np.array(cam_data["cameraMatrix"],     dtype=np.float64)
        self.D = np.array(cam_data["distortionCoeffs"], dtype=np.float64)

        if self.K.shape != (3, 3):
            raise ValueError(f"cameraMatrix must be 3x3, got {self.K.shape}")

        self._intrinsics_loaded = True
        self._homography_ready  = False   # reset H when switching cameras

        print(f"[Calibrator] Loaded intrinsics for camera: '{camera_name}'")
        print(f"             fx={self.K[0,0]:.2f}  fy={self.K[1,1]:.2f}  "
              f"cx={self.K[0,2]:.2f}  cy={self.K[1,2]:.2f}")

    # ── 3. Undistortion helpers ────────────────────────────────────────────────

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
            shape (N, 2) float32 — undistorted pixel coordinates
        """
        self._check_intrinsics()
        pts_reshaped = pts.reshape(-1, 1, 2).astype(np.float32)
        undist = cv2.undistortPoints(pts_reshaped, self.K, self.D, P=self.K)
        return undist.reshape(-1, 2)

    # ── 4. Homography ─────────────────────────────────────────────────────────

    def compute_homography(self, frame: np.ndarray) -> bool:
        """
        Auto-detect checkerboard in frame and compute homography H.
        Since camera is fixed, call this ONCE per session (on any clear frame).
        H is then reused for ALL subsequent frames and ALL patients.

        Pass frames one by one in a loop until it returns True:

            cap = cv2.VideoCapture(video_path)
            while not cal.compute_homography(frame):
                ret, frame = cap.read()

        Returns True if successful, False if board not found in this frame.
        """
        self._check_intrinsics()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            (CALIB_BOARD_COLS, CALIB_BOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
                  | cv2.CALIB_CB_NORMALIZE_IMAGE
                  | cv2.CALIB_CB_FAST_CHECK,
        )

        if not found:
            return False

        # Sub-pixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners  = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_pts  = corners.reshape(-1, 2).astype(np.float32)

        # Undistort before homography
        img_pts_undist = self.undistort_points(img_pts)

        # World points in mm (checkerboard inner corners)
        # Origin = top-left inner corner, x → right, y → down
        world_pts = np.array([
            [c * CALIB_SQUARE_MM, r * CALIB_SQUARE_MM]
            for r in range(CALIB_BOARD_ROWS)
            for c in range(CALIB_BOARD_COLS)
        ], dtype=np.float32)

        H, mask = cv2.findHomography(img_pts_undist, world_pts, cv2.RANSAC, 3.0)

        if H is None:
            print("[Calibrator] WARNING: findHomography failed.")
            return False

        self.H = H
        self._homography_ready = True
        inliers = int(mask.sum()) if mask is not None else "?"
        print(f"[Calibrator] Homography ready. Inliers: {inliers}/{len(world_pts)}")
        return True

    # ── 5. Coordinate conversion ──────────────────────────────────────────────

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
        Convert array of pixel coordinates to mm — batch version (faster).

        Args:
            pts: shape (N, 2)

        Returns:
            shape (N, 2) in mm
        """
        self._check_intrinsics()
        self._check_homography()

        undist = self.undistort_points(pts)
        ones   = np.ones((len(undist), 1), dtype=np.float32)
        pts_h  = np.hstack([undist, ones])        # (N, 3)
        mm_h   = (self.H @ pts_h.T).T             # (N, 3)
        return (mm_h[:, :2] / mm_h[:, 2:3]).astype(np.float32)

    # ── 6. Debug helpers ──────────────────────────────────────────────────────

    def draw_board_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected checkerboard corners on a frame copy. For debugging."""
        self._check_intrinsics()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (CALIB_BOARD_COLS, CALIB_BOARD_ROWS), None)
        out = frame.copy()
        if found:
            cv2.drawChessboardCorners(
                out, (CALIB_BOARD_COLS, CALIB_BOARD_ROWS), corners, found)
        return out

    def is_ready(self) -> bool:
        """True only when both intrinsics and homography are loaded."""
        return self._intrinsics_loaded and self._homography_ready

    # ── Internal checks ───────────────────────────────────────────────────────

    def _check_intrinsics(self):
        if not self._intrinsics_loaded:
            raise RuntimeError(
                "Intrinsics not loaded. "
                "Call load_intrinsics_for_video() or load_intrinsics_by_name() first.")

    def _check_homography(self):
        if not self._homography_ready:
            raise RuntimeError(
                "Homography not computed. Call compute_homography(frame) first.")