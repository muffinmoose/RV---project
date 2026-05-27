# 9HPT Kinematična Analiza

Avtomatska kinematična analiza gibanja roke pri **Nine-Hole Peg Test (9HPT)** iz video posnetkov brez markerjev.

---

## Predpogoji

- Docker
- Video posnetki 9HPT (kamera od zgoraj, format MP4)
- `config/calibration.json` — kalibracijski parametri kamere

---

## Zagon

### 1. En video (interaktivni način)
```bash
docker run -it \
 -v /pot/do/projekta:/workdir \
 -v /pot/do/podatkov:/pot/do/podatkov \
 rv-project \
 python3 src/main.py
```

### 2. Batch način (več videov naenkrat)
```bash
docker run -it \
 -v /pot/do/projekta:/workdir \
 -v /pot/do/podatkov:/pot/do/podatkov \
 rv-project \
 python3 src/main.py /pot/video1.mp4 /pot/video2.mp4
```

### 3. Vsi posnetki enega pacienta
```bash
docker run -it \
 -v /pot/do/projekta:/workdir \
 -v /pot/do/podatkov:/pot/do/podatkov \
 rv-project \
 python3 src/main.py /pot/do/patient_XXX/patient_XXXcamP_1_*.mp4
```

### 4. Kohortna analiza (po batch procesiranju)
```bash
docker run --rm \
 -v /pot/do/projekta:/workdir \
 rv-project \
 python3 testing/analyze.py
```

Generira grafe in `all_patients.csv` v `testing/analysis/`.

---

## Izhodni podatki

Za vsak video → `testing/results/<patient_id>/<video_stem>/`:

| Datoteka | Opis |
|---|---|
| `*_analyzed.mp4` | Anotiran video z landmarki in fazami |
| `*_graphs.png` | Kinematični grafi + tabela povzetkov |
| `*_board.png` | Board figura s trajektorijo v mm |
| `*_results.csv` | Per-zatič statistike (čas, hitrost, pot) |

Kohortna analiza → `testing/analysis/`:

| Datoteka | Opis |
|---|---|
| `all_patients.csv` | Agregiran povzetek vseh posnetkov |
| `plot_01_pin_time.png` | Povprečen čas na zatič po pacientih |
| `plot_02_distribution.png` | Porazdelitev časov |
| `plot_03_velocity_time.png` | Hitrost vs čas |
| `plot_04_fatigue.png` | Trend utrujenosti |
| `plot_07_learning.png` | Krivulja učenja med sejami |
| `plot_08_within_patient.png` | Variabilnost znotraj pacienta |
| `plot_09_path_vs_time.png` | Korelacija pot vs čas |

---

## Reprodukcija rezultatov

Za reprodukcijo rezultatov iz poročila:

1. Zagoni batch procesiranje za vse paciente (003–072, kamera camP_1)
2. Zagoni kohortno analizo: `python3 testing/analyze.py`
3. Grafi se shranijo v `testing/analysis/`

Minimalni prag za veljavni posnetek: **6 od 9 zatičev** zaznanih.

---

Akademsko leto 2025/26 — Robotski vid, FE UL
