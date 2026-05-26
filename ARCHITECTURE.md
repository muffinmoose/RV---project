# Arhitektura sistema — 9HPT Kinematična Analiza

Ta dokument opisuje tehnične odločitve, implementacijo posameznih modulov
in razloge za izbrane pristope.

---

## 1. Splošni pristop

### Zakaj MediaPipe Hands

MediaPipe Hands vrača 21 anatomskih landmarkov roke v vsakem frame-u brez
potrebe po treniranju lastnega modela. Za fiksen setup (kamera od zgoraj,
kontrolirana osvetlitev) je točnost zadostna. Alternativa bi bila YOLO
detekcija ali OpenPose, ampak MediaPipe je bistveno lažji za integracijo
in deluje zanesljivo na CPU brez GPU.

Ključna lastnost: sistem zaklene aktivno roko po `handedness` labelu
(Left/Right), kar je robustno proti MediaPipe index swapom med frameji.

### Zakaj Kalmanov filter

Surove MediaPipe koordinate so šumne — palec in kazalec "skačeta" med
frameji. Kalmanov filter z modelom konstantne hitrosti (state: x, y, vx, vy)
gladi trajektorijo in interpolira čez frame-e kjer MediaPipe ne zazna roke.
Alternativa bi bil Savitzky-Golay filter, ampak Kalman deluje online
(frame po frame) brez zakasnitve in omogoča predikcijo pri izpadlih frame-ih.

### Zakaj HoleTracker za zaznavo zatičev

Začetni pristop je bila rule-based zaznava faz iz velocity in pinch distance
thresholdov (REACHING → GRASPING → TRANSPORTING → PLACING → RETURNING).
Problem: velocity vrednosti so med pacienti zelo variabilne, thresholdi ne
generalizirajo med pacienti z različnimi motoričnimi sposobnostmi.

HoleTracker zaznava zapolnjene luknjice direktno iz svetlosti LED — zanesljivo,
pacienti-neodvisno. Vsaka nova zapolnjena luknjica sproži PegEvent s točnim
časom in kinematičnimi povzetki za ta interval.

---

## 2. Moduli

### 2.1 Calibration (processing/calibration.py)

Homografija H je predhodno izračunana iz kalibracijskih fotografij in shranjena
v `config/calibration.json`. Za vsak video se naloži ustrezna H glede na ime
kamere v imenu datoteke (camP_0/1/2 → left/mid/right).

Pipeline koordinatne pretvorbe na vsak frame:
```
raw pixel → cv2.undistort(K, D) → homografija H → mm
```

Vsak landmark iz MediaPipe se undistortira in pretvori v mm koordinate preden
gre v Kalmanov filter.

### 2.2 HoleTracker (analysis/hole_tracker.py)

LED plošča ima dve polji po 9 luknjic. Ob začetku testa ena stran ugasne
(LED drop) in prižge nazaj (comeback) — to je signal za nastavitev baseline.

Trifazni protokol stabilizacije:
1. **Initial** — zabeleži začetno svetlost vseh luknjic
2. **Drop** — zazna padec svetlosti > 25% → ena stran ugasne
3. **Comeback** — zazna vrnitev svetlosti → nastavi baseline za aktivni sektor

Po baselineu za vsako luknjico velja:
- `brightness < baseline × 0.65` za 5 zaporednih frameov → zapolnjena
- `brightness > baseline × 0.65` za 8 zaporednih frameov → prazna (picking)

Sektor guard: če je kateri landmark roke znotraj bounding boxa sektorja,
se detekcija za ta sektor preskoči — prepreči lažne spremembe ko je roka
v kadru.

### 2.3 Hand Detector (detection/hand_detector.py)

Aktivacija roke čaka na HoleTracker baseline — prepreči lažne detekcije
med LED stabilizacijo.

Po baselineu sistem išče roko ki vstopi v `STORAGE_BBOX_PX` (območje shrambe
zatičev). Ko jo najde, zaklene `_active_hand_label` (npr. "Left") in od tega
frame-a naprej sledi samo tej roki.

Tracking po labelu je robustnejši od trackinga po indeksu ker MediaPipe
pogosto zamenja indekse rok med frameji, label pa ostane stabilen.

Sledeni landmarki:
- `wrist` (0) — za velocity/acceleration
- `thumb_tip` (4) — za pinch detekcijo
- `index_tip` (8) — za pinch detekcijo
- `index_mcp` (5) — MCP členek kazalca, za trajektorijo (stabilnejši od tip-a)

### 2.4 Kalman Filter (processing/kalman_filter.py)

En `KalmanTracker` na landmark, skupaj 4 (wrist, thumb_tip, index_tip, index_mcp).

State vektor: `[x, y, vx, vy]`
Meritev: `[x, y]` v pikslih

Parametri:
- Process noise Q = 1e-4 (majhen → filter verjame modelu)
- Measurement noise R = 1e-2 (srednji → filter verjame meritvam)

Ko MediaPipe ne zazna roke (None), filter samo predvidi naprej brez
korekcijskega koraka — roka "leti" po zadnji znani hitrosti.

### 2.5 Kinematics (processing/kinematics.py)

Vhod: zglajen `[x_mm, y_mm]` iz Kalman filtra
Izhod: `KinematicState` z vsemi vrednostmi v **metrih** in **m/s**

Velocity: centralna diferenca čez sliding window
```
v = (pos[n-1] - pos[n-3]) / (2 * dt)
```

Acceleration: centralna diferenca hitrosti
```
a = (vel[n-1] - vel[n-3]) / (2 * dt)
```

Vse vrednosti se delijo z 1000 (mm → m) pred returnom za konsistentnost
z izhodi grafov in PhaseDetector-ja.

### 2.6 Phase Detector (analysis/phase_detector.py)

Temelji na `HoleTracker.filled_count`:

```
filled_count > 0 in prej = 0  →  začetek PLACING faze
filled_count = 9               →  vsi zatički postavljeni
filled_count < prejšnji        →  začetek RETURNING faze
filled_count = 0               →  test zaključen, IDLE
```

Ob vsaki novi zapolnjeni luknjici se sproži `PegEvent` z:
- `peg_number` — zaporedna številka (1–9)
- `frame_start`, `frame_end` — interval v frame-ih
- `duration_s` — trajanje v sekundah
- `mean_velocity`, `max_velocity` — iz kinematičnih bufferjev
- `mean_acceleration` — povprečni pospešek
- `path_length` — pot roke v tem intervalu

### 2.7 Graphs (analysis/graphs.py)

Generira tri izhodne datoteke:

**Kinematični grafi** (`*_graphs.png`):
- 4 grafi (velocity, acceleration, path length, FFT) + tabela povzetkov
- Faze obarvane kot pasovi v ozadju (zelena = PLACING, vijolična = RETURNING)
- FFT analiza velocity signala — peak v 4–12 Hz bandu je indikator tremora

**Board figura** (`*_board.png`):
- Luknjice narisane v mm koordinatah (via calibrator.pixel_to_mm)
- Trajektorija MCP členka kazalca v istih mm koordinatah
- Številke placing in picking vrstnega reda na vsaki luknjici

**CSV** (`*_results.csv`):
- Ena vrstica na zatič
- Čas, hitrost, pospešek, pot za vsak posamezen cikel

---

## 3. Tok podatkov

```
frame (BGR)
    │
    ├─→ HoleTracker.update(frame, landmarks)
    │       └─→ filled_count, fill_order, pick_order
    │
    ├─→ HandDetector.process(frame, baseline_ready)
    │       └─→ HandDetection(wrist, thumb_tip, index_tip, index_mcp)
    │
    ├─→ HandKalman.update(detection)
    │       └─→ smooth_px {wrist, thumb_tip, index_tip, index_mcp}
    │
    ├─→ cal.pixel_to_mm(smooth_px)
    │       └─→ smooth_mm {wrist, thumb_tip, index_tip, index_mcp}
    │
    ├─→ MultiLandmarkKinematics.update(smooth_mm)
    │       └─→ states {KinematicState per landmark}
    │
    ├─→ PhaseDetector.update(states, frame_idx, filled_count)
    │       └─→ (Phase, Optional[PegEvent])
    │
    └─→ KinematicHistory.record(frame_idx, states, phase, mcp_px)
            └─→ history (za grafe po koncu)
```

---

## 4. Konfiguracija

Vse nastavljive vrednosti so v `config/config.py`:

| Parameter | Vrednost | Opis |
|---|---|---|
| `HOLE_SPACING_MM` | 32.0 | Razdalja med luknjicami |
| `HOLE_RADIUS_PX` | 8 | Polmer vzorčenja za svetlost |
| `STORAGE_BBOX_PX` | (250,120,480,330) | Bbox shrambe zatičev |
| `BOARD_HOLES_BBOX_PX` | (250,10,500,490) | Bbox celotne plošče |
| `MP_DETECTION_CONFIDENCE` | 0.6 | MediaPipe detekcijska zaupnost |
| `MP_TRACKING_CONFIDENCE` | 0.5 | MediaPipe tracking zaupnost |
| `KALMAN_PROCESS_NOISE` | 1e-4 | Zaupanje modelu vs meritvam |
| `SLIDING_WINDOW_FRAMES` | 5 | Okno za glajenje kinematike |

---

## 5. Znane omejitve in možne izboljšave

**Omejitve:**
- Path length je precenjena zaradi Kalman interpolacije pri izpadlih frame-ih
- FFT analiza ne ločuje tremora od ritma postavljanja zatičev
- HoleTracker je občutljiv na nenadne spremembe osvetlitve
- Sistem predpostavlja en sektor aktiven (leva ali desna stran plošče)

**Možne izboljšave:**
- Ločena tremor analiza samo za mirujoče faze
- Multi-kamera fuzija za 3D trajektorijo
- Per-prst kinematika za vse 5 prstov
- ML classifier za fazno detekcijo namesto rule-based

---

Akademsko leto 2025/26 — Robotski vid, Fakulteta za elektrotehniko UL