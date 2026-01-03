# Biomass2Prediction (Kaggle) — plan projektu

> Cel: zbudować projekt ML (image → regression) w stylu produkcyjnym: modularna logika w `src/`, eksperymenty w notebookach / Kaggle, pełna reprodukowalność, czytelne wyniki i dokumentacja do portfolio.

## 0) Założenia workflow
- **Lokalnie (PyCharm):** piszemy większość logiki (dataset, model, trening, walidacja, inferencja, metryki, utils).
- **Chmura notebookowa (Kaggle/Colab/Modal itp.):** uruchamiamy eksperymenty i trening (GPU/CPU), ale importujemy kod z repo (`pip install -e .` lub `pythonpath`).
- **Reprodukowalność:** configi (YAML), seed, logi (CSV/JSON), zapisy checkpointów.
- **Czytelność portfolio:** README + szybkie odtworzenie wyniku (`make train`, `make infer`, `make submit`).

## 1) Milestones (kolejność prac)
### M1 — Setup repo + struktura
- [ ] Utwórz repozytorium Git (GitHub).
- [ ] Dodaj `.gitignore`, `README.md`, licencję (np. MIT), `pyproject.toml`.
- [ ] Ustal strukturę katalogów (poniżej).
- [ ] Stwórz minimalny pipeline: „wczytaj dane → DataLoader → model → forward”.

### M2 — Baseline end-to-end
- [ ] Dataset i augmentacje (train/valid split).
- [ ] Model baseline (np. EfficientNet/ConvNeXt/ResNet) + head regresyjny.
- [ ] Trening z logowaniem (loss/metric), zapisy checkpointów.
- [ ] Inferencja i generacja pliku submission.
- [ ] Jeden notebook „Train baseline” + jeden „Inference & submission”.

### M3 — Eksperymenty i ulepszenia
- [ ] K-fold / GroupKFold (jeśli ma sens wg metadanych).
- [ ] Lepsze augmentacje (RandAugment, MixUp/CutMix jeśli działa).
- [ ] Harmonogram LR (Cosine, OneCycle).
- [ ] TTA / ensembling (lekko, pod portfolio).
- [ ] Analiza błędów (pred vs true, outliers, wykresy).

### M4 — Produkcyjna jakość kodu
- [ ] `configs/*.yaml` + loader configów.
- [ ] `src/train.py`, `src/infer.py` jako CLI (argparse/typer).
- [ ] Testy jednostkowe dla krytycznych elementów (dataset, metryki).
- [ ] Pre-commit (black/ruff/isort).
- [ ] Krótkie „Model Card” i „Repro steps” w README.

### M5 — Finalizacja pod portfolio
- [ ] Czytelne wyniki (tabela eksperymentów, najlepszy config).
- [ ] Wykresy treningu, przykłady predykcji.
- [ ] Tag release v1.0 + krótkie podsumowanie „co zrobiłem i dlaczego”.

## 2) Struktura projektu (proponowana)
```
biomass2prediction/
  README.md
  LICENSE
  .gitignore
  pyproject.toml
  Makefile
  configs/
    baseline.yaml
    train.yaml
    infer.yaml
  notebooks/
    01_eda.ipynb
    02_train_baseline.ipynb
    03_infer_submit.ipynb
  src/
    biomass2pred/
      __init__.py
      config.py
      seed.py
      data/
        __init__.py
        dataset.py
        transforms.py
        datamodule.py
        splits.py
      models/
        __init__.py
        backbone.py
        head.py
        factory.py
      train/
        __init__.py
        losses.py
        metrics.py
        trainer.py
        callbacks.py
      infer/
        __init__.py
        predict.py
        tta.py
      utils/
        __init__.py
        io.py
        logging.py
        paths.py
  scripts/
    download_kaggle_data.sh
    kaggle_train.sh
    kaggle_infer.sh
  outputs/
    .gitkeep
    runs/
    checkpoints/
    submissions/
  tests/
    test_dataset.py
    test_metrics.py
  docs/
    report.md
```
**Zasada:** wszystko co „ważne” jest w `src/biomass2pred/`. Notebooki to tylko „driver” + wizualizacje.

## 3) Konwencje danych i artefaktów
- Dane trzymamy **poza gitem** (`data/` w `.gitignore`), a ścieżki ustawiamy przez:
  - zmienną środowiskową `DATA_DIR`, lub
  - `configs/*.yaml`.
- Artefakty:
  - `outputs/runs/<run_id>/metrics.csv`
  - `outputs/checkpoints/<run_id>/best.ckpt`
  - `outputs/submissions/<run_id>/submission.csv`

## 4) Minimalny standard trenowania
- Seed: jeden helper `seed_everything(seed)`.
- Split: `splits.py` generuje i zapisuje indexy train/valid.
- Logi: CSV (łatwe do wrzucenia w README) + opcjonalnie W&B.
- Checkpoint: zawsze zapisuj „best” i „last”.
- Metryki: jedna główna metryka z Kaggle + pomocnicze (MAE/RMSE).

## 5) Integracja z notebookową chmurą
W notebooku:
1) sklonuj repo (`git clone ...`)
2) `pip install -e .`
3) odpal `python -m biomass2pred.train ...` albo importuj `Trainer`.
To pozwala, żeby **kod był ten sam** lokalnie i w chmurze.

## 6) Checklist dla GitHub (portfolio)
- README: opis problemu, podejście, wyniki, jak uruchomić.
- „Quickstart” (3 komendy) + sekcja „Project structure”.
- Link do Kaggle notebook (jeśli używasz) + krótki raport w `docs/report.md`.
- Zrzuty wykresów / tabelka wyników (w README).

---
Następny krok: stworzenie repo + szkieletu folderów + pierwszych plików (`pyproject.toml`, `src/...`, `configs/baseline.yaml`).
