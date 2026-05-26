# config.py
# Central configuration for 9HPT Kinematic Analysis

# ─── PATHS ────────────────────────────────────────────────────────────────────
DATA_ROOT    = "/media/FastDataMama/data_rv_26/Data"
VIDEO_ROOT   = "/media/FastDataMama/imep/"
CALIB_FILE   = "config/calibration.json"
RESULTS_DIR  = "results/"

# ─── CAMERA MAPPING ───────────────────────────────────────────────────────────
CAMERA_MAP = {
    "camP_0": "left",
    "camP_1": "mid",
    "camP_2": "right",
}
PRIMARY_CAMERA = "camP_1"

# ─── CALIBRATION BOARD (šahovnica) ────────────────────────────────────────────
CALIB_BOARD_COLS = 9
CALIB_BOARD_ROWS = 6
CALIB_SQUARE_MM  = 20.0

# ─── 9HPT BOARD DIMENSIONS ────────────────────────────────────────────────────
HOLE_SPACING_MM          = 32.0
HOLE_SPACING_PX          = 34.6
HOLE_DIAMETER_MM         = 10.0
HOLE_RADIUS_PX           = 8
HOLE_ROWS                = 3
HOLE_COLS                = 3

# Pixel koordinate luknjic (originalen frame, camP_1, undistorted)
# Levo polje (spodaj) — stolpci L→D, vrstice spodaj→zgoraj
LEFT_HOLES_PX = [
    (340, 409), (373, 403), (405, 397),
    (339, 371), (372, 367), (405, 361),
    (338, 333), (372, 330), (404, 326),
]

# Desno polje (zgoraj)
RIGHT_HOLES_PX = [
    (335, 111), (369, 114), (400, 116),
    (335,  73), (368,  78), (400,  81),
    (335,  37), (368,  42), (399,  47),
]

STORAGE_CENTER_PX        = (367, 221)
LEFT_FIELD_CENTER_PX     = (372, 367)
RIGHT_FIELD_CENTER_PX    = (368,  78)

# Board bboxes
BOARD_HOLES_BBOX_PX = (250, 10, 500, 490) #vse luknjice + malo prostora okoli
STORAGE_BBOX_PX = (250, 120, 480, 330) # samo krog kjer so pini

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
MP_MAX_HANDS             = 2
MP_DETECTION_CONFIDENCE  = 0.6   # bil 0.2 — preveč lažnih zaznav rok
MP_TRACKING_CONFIDENCE   = 0.5   # bil 0.2 — tracking preveč nestabilen

LANDMARKS_OF_INTEREST = {
    "wrist":     0,
    "thumb_tip": 4,
    "index_tip": 8,
}

# ─── KALMAN FILTER ────────────────────────────────────────────────────────────
KALMAN_PROCESS_NOISE     = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2

# ─── KINEMATICS ───────────────────────────────────────────────────────────────
FPS                   = 30
SLIDING_WINDOW_FRAMES = 5

# ─── VISUALIZER ───────────────────────────────────────────────────────────────
LANDMARK_COLOR  = (0, 255, 0)
LANDMARK_RADIUS = 5
SKELETON_COLOR  = (255, 255, 255)