# 9HPT Kinematična Analiza Gibanja Roke

Avtomatska računalniška analiza kinematičnih parametrov iz video posnetkov **Nine-Hole Peg Testa (9HPT)** — testa ročne spretnosti, ki se uporablja v diagnostiki multiple skleroze in možganske kapi.

---

## Opis projekta

Iz video posnetka (top-down kamera, fiksen setup) sistem avtomatsko določi:

- **pot** gibanja roke/prstov skozi čas
- **hitrost** (prvi odvod pozicije)
- **pospešek** (drugi odvod pozicije)
- ločeno sledenje **palca in kazalca** (landmark 4 in 8)
- zaznavo **prijema in odlaganja** zatiča

Brez fizičnih markerjev, brez posebne opreme — samo kamera in Python.

---

## Arhitektura pipeline-a

```
VIDEO
  │
  ▼
[1] mediapipe_detector.py   — detekcija 21 landmarkov roke / frame
  │
  ▼
[2] kalman_filter.py        — glajenje šumnih trajektorij
  │
  ▼
[3] calibration.py          — homografija: piksel → mm (referenca: luknjice plošče)
  │
  ▼
[4] kinematics.py           — numerični odvodi → d / v / a skozi čas
  │
  ▼
[5] phase_detector.py       — zaznava faz: prijem / prenos / odlaganje
  │
  ▼
REZULTATI: grafi (.png), podatki (.csv)
```

---

## Struktura repozitorija

```
9hpt-kinematic-analysis/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── src/
│   ├── mediapipe_detector.py   # Detekcija roke z MediaPipe Hands
│   ├── kalman_filter.py        # Kalmanov filter za glajenje
│   ├── calibration.py          # Kalibracija pixel → mm (homografija)
│   ├── kinematics.py           # Izračun poti, hitrosti, pospeška
│   ├── phase_detector.py       # Zaznava faz prijema/odlaganja
│   └── main.py                 # Glavna skripta — zažene celoten pipeline
│
├── data/
│   ├── videos/                 # Vhodni video posnetki (.mp4)
│   └── results/                # Izhodni grafi in CSV datoteke
│
└── notebooks/
    └── analysis.ipynb          # Jupyter notebook za analizo rezultatov
```

---

## Namestitev in zagon

### Predpogoji

- [Docker](https://www.docker.com/)
- [VS Code](https://code.visualstudio.com/) + Remote - Containers razširitev

### 1. Kloniranje repozitorija

```bash
git clone https://github.com/<tvoje-ime>/9hpt-kinematic-analysis.git
cd 9hpt-kinematic-analysis
```

### 2. Gradnja Docker slike

```bash
docker build -t 9hpt_analysis .
```

### 3. Zagon analize

```bash
docker run --rm \
  -v $(pwd)/data:/workdir/data \
  9hpt_analysis \
  python src/main.py --video data/videos/test.mp4
```

### 4. Zagon v VS Code (Dev Container)

Odpri mapo v VS Code → `Reopen in Container` → terminal je že v okolju.

---

## Dockerfile

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    mediapipe \
    scipy \
    matplotlib \
    pandas

WORKDIR /workdir
COPY src/ src/
COPY data/ data/

CMD ["python", "src/main.py"]
```

---

## Konfiguracija SSH dostopa (strežnik LST)

```
Host zigab_w1
  HostName 192.168.32.141
  User <tvoje-uporabniško-ime>
  Port 3322
```

Podatki na strežniku so v `/media/FastDataMama/imep`.

---

## Tehnologije

| Komponenta | Knjižnica | Namen |
|---|---|---|
| Detekcija roke | `mediapipe` | 21 landmarkov/frame, brez GPU |
| Glajenje | `scipy` (Savitzky-Golay) / Kalman | Odstranitev šuma iz trajektorij |
| Kalibracija | `opencv` (homografija) | Pretvorba piksel → mm |
| Kinematika | `numpy` (`np.gradient`) | Numerični odvodi d/v/a |
| Vizualizacija | `matplotlib` | Grafi, animacije |
| Podatki | `pandas` | Izvoz v CSV |

---

## Rezultati

Po zagonu pipeline-a najdeš v `data/results/`:

- `trajectory.png` — vizualizacija poti roke
- `velocity.png` — hitrost skozi čas
- `acceleration.png` — pospešek skozi čas
- `kinematics.csv` — surovi numerični podatki (čas, x, y, v, a)

---

Akademsko leto 2025/26
