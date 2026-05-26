# Kinematična analiza gibanja roke pri testu devetih zatičev (9HPT)

Avtomatski sistem za ekstrakcijo kinematičnih parametrov iz video posnetkov
**Nine-Hole Peg Testa (9HPT)** — standardiziranega kliničnega testa ročne
spretnosti, ki se uporablja pri diagnostiki multiple skleroze, možganske kapi
in drugih nevroloških obolenj.

---

## Opis problema

9HPT meri čas, ki ga pacient potrebuje za premikanje devetih zatičev iz
shrambe v luknjice plošče in nazaj. Standardna izvedba beleži samo skupni čas
— izgubi pa vse informacije o **načinu** gibanja: hitrost, pospeček, tremor,
trajektorija roke.

Ta sistem iz video posnetka avtomatsko ekstrahira te kinematične parametre
brez fizičnih markerjev in brez specialne opreme.

---

## Kaj sistem zazna in izmeri

- **Trajektorija roke** — pot MCP členka kazalca skozi čas (v mm)
- **Hitrost** (m/s) — prvi odvod pozicije, ločeno za zapestje, palec, kazalec
- **Pospešek** (m/s²) — drugi odvod pozicije
- **Faze testa** — PLACING (polnjenje luknjic) in RETURNING (jemanje nazaj)
- **Čas na zatič** — trajanje vsakega posameznega cikla
- **Vrstni red zatičev** — katera luknjica je bila zapolnjena kdaj
- **Dominantna frekvenca gibanja** — FFT analiza za detekcijo tremora (4–12 Hz)
- **Aktivna roka** — avtomatska detekcija leve/desne roke

---

## Arhitektura pipeline-a

```
VIDEO (MP4)
    │
    ▼
[1] Kalibracija (calibration.py)
    │   Naloži K, D, H iz calibration.json
    │   Undistort vsak frame
    │   Pretvori pixel → mm via homografija
    │
    ▼
[2] Detekcija luknjic (hole_tracker.py)
    │   Zazna LED drop/comeback → nastavi baseline
    │   Določi aktivni sektor (levo/desno polje)
    │   Hysteresis detekcija: zapolnjena / prazna luknjica
    │   Sektor guard: ignorira luknjice ko je roka v kadru
    │
    ▼
[3] Detekcija roke (hand_detector.py)
    │   MediaPipe Hands — do 2 roki, 21 landmarkov/frame
    │   Aktivacija: roka vstopi v storage bbox
    │   Zaklepanje po handedness labelu
    │
    ▼
[4] Glajenje (kalman_filter.py)
    │   Kalmanov filter (state: x, y, vx, vy)
    │   Interpolacija čez izpadle frame
    │   En tracker na landmark (zapestje, palec, kazalec)
    │
    ▼
[5] Kinematika (kinematics.py)
    │   Centralna diferenca za velocity in acceleration
    │   Sliding window glajenje
    │   Vse enote v metrih in m/s
    │
    ▼
[6] Zaznava faz (phase_detector.py)
    │   Temelji na HoleTracker fill_order
    │   PLACING: filled_count raste 0→9
    │   RETURNING: filled_count pada 9→0
    │   PegEvent na vsak nov zatič: čas, hitrost, pot
    │
    ▼
[7] Vizualizacija (visualizer.py + graphs.py)
        Anotiran video z landmarki in luknjicami
        Kinematični grafi (velocity, acceleration, path length, FFT)
        Board figura v mm koordinatah s trajektorijo
        Tabela povzetkov (čas, hitrost, tremor peak)
```

---

## Struktura repozitorija

```
MatejH/
├── Dockerfile
├── README.md
├── ARCHITECTURE.md
│
├── config/
│   ├── config.py               # Centralna konfiguracija (poti, parametri)
│   └── calibration.json        # K, D, H matrike za vse 3 kamere
│
└── src/
    ├── main.py                 # Vstopna točka — interaktivni in batch način
    │
    ├── detection/
    │   └── hand_detector.py    # MediaPipe detekcija, aktivacija, tracking
    │
    ├── processing/
    │   ├── calibration.py      # Kalibracija, undistort, pixel→mm
    │   ├── kalman_filter.py    # Kalmanov filter za glajenje trajektorij
    │   └── kinematics.py       # Izračun d/v/a iz mm koordinat
    │
    ├── analysis/
    │   ├── hole_tracker.py     # Detekcija zapolnjenih luknjic (LED based)
    │   ├── phase_detector.py   # Zaznava faz iz HoleTracker
    │   └── graphs.py           # Kinematični grafi + board figura
    │
    └── utils/
        ├── visualizer.py       # Anotiran video output
        └── logger.py           # Structured logging
```

---

## Zagon

### Predpogoji

- Docker
- SSH dostop do strežnika (port 3322)

### Interaktivni način (en pacient)

```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py
```

### Batch način (več videov naenkrat)

```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py /pot/video1.mp4 /pot/video2.mp4 ...
```

### Batch način — vsi posnetki enega pacienta

```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py \
  /media/FastDataMama/data_rv_26/Data/patient_XXX/patient_XXXcamP_1_*.mp4
```

### Download rezultatov (Windows CMD)

```cmd
scp -P 3322 -r matejh@192.168.32.141:/media/FastDataMama/MatejH/results/patient_XXX "C:\Users\matej\Documents\01-Fakulteta\Magisterij\01_Letnik\2. semester\01-RV\04-Izziv\Results_testing\"
```

---

## Izhodni podatki

Za vsak video se generira mapa `results/<patient_id>/<video_stem>/`:

| Datoteka | Opis |
|---|---|
| `*_analyzed.mp4` | Video z anotiranimi landmarki in luknjicami |
| `*_graphs.png` | Kinematični grafi + tabela povzetkov |
| `*_board.png` | Board figura z vrstnim redom zatičev in trajektorijo |
| `*_results.csv` | Per-zatič statistike (čas, hitrost, pospešek, pot) |
| `*.log` | Strukturiran log procesiranja |

---

## Tehnologije

| Komponenta | Knjižnica | Namen |
|---|---|---|
| Detekcija roke | `mediapipe 0.10` | 21 landmarkov/frame, CPU |
| Glajenje | Kalmanov filter (`numpy`) | Interpolacija, odstranitev šuma |
| Kalibracija | `opencv` | Homografija pixel→mm, undistort |
| Kinematika | `numpy` | Centralna diferenca, sliding window |
| Vizualizacija | `matplotlib`, `opencv` | Grafi, anotiran video |

---

## Omejitve

- Sistem je kalibriran za kamero `camP_1` (sredinska kamera, top-down pogled)
- Deluje za eno roko na enkrat — aktivna roka se zaklene ob prvem vstopu v storage bbox
- FFT analiza kaže dominantno frekvenco gibanja — za klinično interpretacijo tremora bi potrebovali ločeno analizo mirujočih faz
- Zaznava zatičev temelji na LED luknjicah — občutljivo na spremembe osvetlitve

---

Akademsko leto 2025/26 — Robotski vid, Fakulteta za elektrotehniko UL