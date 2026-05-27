# 9HPT Kinematična Analiza — Navodila za zagon

## Korak 1 — Postavi se v pravo mapo

```bash
cd /media/FastDataMama/MatejH
```

---

## Korak 2 — Zgradi Docker image (samo prvič, traja ~5 min)

```bash
docker build -t rv-project .
```

---

## Korak 3 — Zaženi analizo

### Način A: Interaktivni (najlažje)

```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py
```

Program te bo sam vodil:
1. Vpiši številko pacienta (npr. `064`)
2. Izberi video iz ponujenega seznama
3. Počakaj — rezultati se shranijo v `/media/FastDataMama/MatejH/testing/results/`

---

### Način B: Direktno poda pot

```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py --input /media/FastDataMama/data_rv_26/patient_064
```

Za en sam video:
```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py --input /media/FastDataMama/data_rv_26/patient_064/patient_064camP_1_20230413_13_13_07.mp4
```

---

### Način C: Analiza vseh pacientov skupaj

```bash
docker run --rm \
  -v /media/FastDataMama/MatejH:/workdir \
  rv-project \
  python3 testing/analyze.py
```

---

## Korak 4 — Rezultati

**Za vsak video** se shranijo v:
```
/media/FastDataMama/MatejH/testing/results/<patient_id>/<video_stem>/
```

| Datoteka | Opis |
|---|---|
| `*_analyzed.mp4` | Video z označenimi landmarki in fazami |
| `*_graphs.png` | Kinematični grafi |
| `*_board.png` | Trajektorija v mm |
| `*_results.csv` | Statistike po zatičih |

**Po analizi vseh pacientov skupaj** se shranijo v:
```
/media/FastDataMama/MatejH/testing/analysis/
```

| Datoteka | Opis |
|---|---|
| `all_patients.csv` | Povzetek vseh posnetkov |
| `plot_01_pin_time.png` | Povprečen čas na zatič |
| `plot_02_distribution.png` | Porazdelitev časov |
| `plot_03_velocity_time.png` | Hitrost vs čas |
| `plot_04_fatigue.png` | Trend utrujenosti |
| `plot_07_learning.png` | Krivulja učenja med sejami |
| `plot_08_within_patient.png` | Variabilnost znotraj pacienta |
| `plot_09_path_vs_time.png` | Korelacija pot vs čas |

---

Akademsko leto 2025/26 — Robotski vid, FE UL
