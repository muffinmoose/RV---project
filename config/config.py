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
# Vrednosti iz click_holes.py meritev (originalen frame 640x480, brez rotacije)
HOLE_SPACING_MM          = 32.0
HOLE_SPACING_PX          = 34.6
HOLE_DIAMETER_MM         = 10.0
HOLE_RADIUS_PX           = 8       # za detekcijo filled/empty
HOLE_ROWS                = 3
HOLE_COLS                = 3

# Pixel koordinate luknjic (originalen frame, camP_1)
# Levo polje (spodaj na originalnem framu) — stolpci L→D, vrstice spodaj→zgoraj
LEFT_HOLES_PX = [
    (340, 409), (373, 403), (405, 397),   # stolpec 1: L1, L4, L7
    (339, 371), (372, 367), (405, 361),   # stolpec 2: L2, L5, L8
    (338, 333), (372, 330), (404, 326),   # stolpec 3: L3, L6, L9
]

# Desno polje (zgoraj na originalnem framu)
RIGHT_HOLES_PX = [
    (335, 111), (369, 114), (400, 116),   # stolpec 1: R1, R4, R7
    (335,  73), (368,  78), (400,  81),   # stolpec 2: R2, R5, R8
    (335,  37), (368,  42), (399,  47),   # stolpec 3: R3, R6, R9
]

STORAGE_CENTER_PX        = (367, 221)
LEFT_FIELD_CENTER_PX     = (372, 367)
RIGHT_FIELD_CENTER_PX    = (368,  78)

# ─── MEDIAPIPE ────────────────────────────────────────────────────────────────
MP_MAX_HANDS             = 1
MP_DETECTION_CONFIDENCE  = 0.5
MP_TRACKING_CONFIDENCE   = 0.5

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