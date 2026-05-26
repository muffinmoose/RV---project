# 9HPT Kinematična Analiza

Avtomatska kinematična analiza gibanja roke pri **Nine-Hole Pin Test (9HPT)** iz video posnetkov.

---

## Zagon

### Predpogoji
- Docker
- SSH dostop do strežnika

### Interaktivni način
```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py
```

### Batch način (več videov)
```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py /pot/video1.mp4 /pot/video2.mp4
```

### Profesor način
```bash
docker run -it \
  -v /media/FastDataMama/MatejH:/workdir \
  -v /media/FastDataMama/data_rv_26:/media/FastDataMama/data_rv_26 \
  rv-project \
  python3 src/main.py --input /pot/do/patient_dir --output /pot/do/outputa
```

### Kohortna analiza (po batch procesiranju)
```bash
docker run --rm \
  -v /media/FastDataMama/MatejH:/workdir \
  rv-project \
  python3 testing/analyze.py
```

---

## Izhodni podatki

Za vsak video → `results/<patient_id>/<video_stem>/`:

| Datoteka | Opis |
|---|---|
| `*_analyzed.mp4` | Anotiran video |
| `*_graphs.png` | Kinematični grafi + tabela povzetkov |
| `*_board.png` | Board figura s trajektorijo |
| `*_results.csv` | Per-zatič statistike |

### Download rezultatov (Windows CMD)
```cmd
scp -P 3322 -r matejh@192.168.32.141:/media/FastDataMama/MatejH/results/patient_XXX "C:\Users\matej\...\Results_testing\"
```

---

Akademsko leto 2025/26 — Robotski vid, FE UL
