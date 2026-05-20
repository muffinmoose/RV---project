# config.py
# Central configuration for 9HPT Kinematic Analysis

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_ROOT       = "/media/FastDataMama/data_rv_26/Data"
VIDEO_ROOT      = "/media/FastDataMama/imep/"
CALIB_FILE = "/media/FastDataMama/zigab/calibration_photos/ukc_calibration.json"
RESULTS_DIR     = "data/results/"

# ─── CAMERA MAPPING ───────────────────────────────────────────────────────────
# Maps camP_X (from filename) → camera name (key in ukc_calibration.json)
CAMERA_MAP = {
    "camP_0": "left",
    "camP_1": "mid",
    "camP_2": "right",
}
PRIMARY_CAMERA = "camP_1"   # mid camera — best angle for analysis (nearly top-down)

# ─── CAMERA CALIBRATION BOARD ─────────────────────────────────────────────────
# Checkerboard pattern: 9x6 inner corners, 20x20 mm squares
CALIB_BOARD_COLS    = 9        # inner corners horizontal
CALIB_BOARD_ROWS    = 6        # inner corners vertical
CALIB_SQUARE_MM     = 20.0     # square size in mm

# ─── 9HPT BOARD DIMENSIONS ────────────────────────────────────────────────────
# TODO: fill in your actual board dimensions
BOARD_WIDTH_MM      = 0.0      # overall board width in mm   ← YOU FILL THIS
BOARD_HEIGHT_MM     = 0.0      # overall board height in mm  ← YOU FILL THIS
HOLE_DIAMETER_MM    = 0.0      # diameter of each peg hole   ← YOU FILL THIS
HOLE_SPACING_MM     = 0.0      # center-to-center spacing    ← YOU FILL THIS
HOLE_ROWS           = 3        # rows of holes
HOLE_COLS           = 3        # columns of holes

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
MP_MAX_HANDS            = 1     # only one hand in 9HPT
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE  = 0.5

# Which landmarks to track (MediaPipe indices)
# 0 = wrist, 4 = thumb tip, 8 = index tip
LANDMARKS_OF_INTEREST = {
    "wrist":     0,
    "thumb_tip": 4,
    "index_tip": 8,
}

# ─── KALMAN FILTER ────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE        = 1e-4
KALMAN_MEASUREMENT_NOISE    = 1e-2

# ─── KINEMATICS ───────────────────────────────────────────────────────────────
FPS                     = 30
SLIDING_WINDOW_FRAMES   = 5     # frames for sliding-window velocity smoothing

# ─── VISUALIZER / UI ──────────────────────────────────────────────────────────
STATS_PANEL_RATIO   = 1/3
LANDMARK_COLOR      = (0, 255, 0)
LANDMARK_RADIUS     = 5
SKELETON_COLOR      = (255, 255, 255)