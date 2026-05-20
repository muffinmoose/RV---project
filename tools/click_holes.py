import cv2
import numpy as np
import sys

TEST_VIDEO = "C:\\Users\\matej\\patient_003camP_1_20231005_14_13_43.mp4"

K = np.array([
    [478.66225232,   0.0,          346.2049993 ],
    [  0.0,        478.26439239,   211.95438758],
    [  0.0,          0.0,            1.0       ]
])
D = np.array([[-0.39941367, 0.20329422, -0.00136692, 0.00138915, -0.04521047]])
H = np.array([
    [ 0.19123533132041956, -0.5020142102957392,   88.51757836194574  ],
    [ 0.5084405195987515,   0.20367929151403272, -224.35398385452217 ],
    [-0.0002777484933742214, -4.245645372463294e-06, 1.0             ]
])

def px_to_mm(pt_px):
    pt = np.array([[[float(pt_px[0]), float(pt_px[1])]]], dtype=np.float32)
    return cv2.perspectiveTransform(pt, H)[0][0]

cap = cv2.VideoCapture(TEST_VIDEO)
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR")
    sys.exit(1)

frame = cv2.undistort(frame, K, D)
# BEZ rotacije
print(f"Frame resolucija: {frame.shape[1]}x{frame.shape[0]}")

ORIG_H, ORIG_W = frame.shape[:2]
SCALE = 2
display = cv2.resize(frame, (ORIG_W * SCALE, ORIG_H * SCALE))

clicks = []
labels = (
    [f"L{i+1}" for i in range(9)] +
    [f"R{i+1}" for i in range(9)] +
    ["CENTER",
     "BOARD_TL", "BOARD_TR", "BOARD_BL", "BOARD_BR"]
)

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = len(clicks)
        if idx < len(labels):
            rx, ry = x // SCALE, y // SCALE
            clicks.append((rx, ry))
            print(f"  [{labels[idx]}] → ({rx}, {ry})")

print("\nNAVODILA (originalen frame, brez rotacije):")
print("  1. Klikni 9x LEVO polje (L1-L9): vrstica po vrstica, levo→desno")
print("     (na originalnem framu je levo polje SPODAJ)")
print("  2. Klikni 9x DESNO polje (R1-R9): enako (ZGORAJ na originalnem framu)")
print("  3. CENTER kroga")
print("  4. Board corners: zgoraj-levo, zgoraj-desno, spodaj-levo, spodaj-desno")
print("  Pritisni Q za konec\n")

cv2.namedWindow("click_holes")
cv2.setMouseCallback("click_holes", mouse_cb)

while True:
    vis = display.copy()
    for i, (x, y) in enumerate(clicks):
        sx, sy = x * SCALE, y * SCALE
        color = (0, 255, 255) if i >= 19 else (0, 255, 0)
        cv2.circle(vis, (sx, sy), 8, color, -1)
        cv2.putText(vis, labels[i], (sx+8, sy-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if len(clicks) < len(labels):
        cv2.putText(vis, f"Klikni: {labels[len(clicks)]}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    else:
        cv2.putText(vis, "Vse kliknjeno! Pritisni Q",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("click_holes", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

if len(clicks) < 23:
    print(f"Prekinjeno — kliknjenih samo {len(clicks)}/23 točk")
    sys.exit(1)

left_px    = np.array(clicks[:9]).reshape(3, 3, 2)
right_px   = np.array(clicks[9:18]).reshape(3, 3, 2)
ctr_px     = np.array(clicks[18])
corners_px = np.array(clicks[19:23])

def avg_spacing(grid):
    dx = np.mean([np.linalg.norm(grid[r,c+1]-grid[r,c]) for r in range(3) for c in range(2)])
    dy = np.mean([np.linalg.norm(grid[r+1,c]-grid[r,c]) for r in range(2) for c in range(3)])
    return (dx + dy) / 2

spacing_px = np.mean([avg_spacing(left_px), avg_spacing(right_px)])
left_mm    = np.array([[px_to_mm(left_px[r,c])  for c in range(3)] for r in range(3)])
right_mm   = np.array([[px_to_mm(right_px[r,c]) for c in range(3)] for r in range(3)])
ctr_mm     = px_to_mm(ctr_px)
corners_mm = np.array([px_to_mm(c) for c in corners_px])
spacing_mm = np.mean([avg_spacing(left_mm), avg_spacing(right_mm)])

left_center_px  = left_px[1,1]
right_center_px = right_px[1,1]
field_dist_px   = np.linalg.norm(right_center_px - left_center_px)
field_dist_mm   = np.linalg.norm(px_to_mm(right_center_px) - px_to_mm(left_center_px))
px_per_mm       = spacing_px / spacing_mm

board_w_mm = np.mean([
    np.linalg.norm(corners_mm[1]-corners_mm[0]),
    np.linalg.norm(corners_mm[3]-corners_mm[2])
])
board_h_mm = np.mean([
    np.linalg.norm(corners_mm[2]-corners_mm[0]),
    np.linalg.norm(corners_mm[3]-corners_mm[1])
])

print(f"\n── REZULTATI ────────────────────────────────────────")
print(f"Spacing (px)              : {spacing_px:.1f} px")
print(f"Spacing (mm, homografija) : {spacing_mm:.1f} mm  (pričakovano: 32.0)")
print(f"px/mm                     : {px_per_mm:.4f}")
print(f"Razdalja L↔R (px)         : {field_dist_px:.1f} px")
print(f"Razdalja L↔R (mm)         : {field_dist_mm:.1f} mm")
print(f"Board širina (mm)         : {board_w_mm:.1f} mm")
print(f"Board višina (mm)         : {board_h_mm:.1f} mm")

print(f"\n── ZA CONFIG.PY ─────────────────────────────────────")
print(f"HOLE_SPACING_PX        = {spacing_px:.1f}")
print(f"HOLE_SPACING_MM        = {spacing_mm:.1f}")
print(f"LEFT_FIELD_CENTER_PX   = ({left_center_px[0]}, {left_center_px[1]})")
print(f"RIGHT_FIELD_CENTER_PX  = ({right_center_px[0]}, {right_center_px[1]})")
print(f"STORAGE_CENTER_PX      = ({ctr_px[0]}, {ctr_px[1]})")
print(f"BOARD_WIDTH_MM         = {board_w_mm:.1f}")
print(f"BOARD_HEIGHT_MM        = {board_h_mm:.1f}")
print(f"PX_PER_MM              = {px_per_mm:.4f}")

print(f"\n── VSE LUKNJICE (px → mm) ───────────────────────────")
print("Levo polje:")
for r in range(3):
    for c in range(3):
        print(f"  L{r*3+c+1}: px={left_px[r,c].tolist()}  mm=[{left_mm[r,c][0]:.1f}, {left_mm[r,c][1]:.1f}]")
print("Desno polje:")
for r in range(3):
    for c in range(3):
        print(f"  R{r*3+c+1}: px={right_px[r,c].tolist()}  mm=[{right_mm[r,c][0]:.1f}, {right_mm[r,c][1]:.1f}]")