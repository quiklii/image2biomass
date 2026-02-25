# CSIRO - Image2Biomass Prediction

Predict above-ground biomass from images using PyTorch + timm backbones (e.g. efficientnet).

The project itself is still under development. Current version includes:
Clean, extendable baseline for biomass regression built on:
- **timm model zoo** (Backbone is defined in config)
- a **single regression head** (with 3 outputs)
- a **Trainer** class with staged fine-tuning (freeze -> head-only -> partial unfreeze)

> Note: This Kaggle competition is finished. This repository focuses on the training pipeline and experiment setup.
> Inference/submission generation is planned as a future improvement.

---
## Why this repo?

This project is structured as a solid **base for future improvements**:
- swap backbone models (timm model zoo)
- add stronger augmentations, losses, and schedulers
- extend to K-Fold, and later add left/right splitting + fusion - without rewriting the whole pipeline

All training/experiment parameters (e.g., backbone name, image size, epochs, learning rate, freezing/unfreezing stages)
are defined in a config file, which makes experiments reproducible and easy to compare.
---
## Quickstart

### 1) Create venv
```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```
### 2) Install (from `pyproject.toml`)

Install the project in editable mode:
```bash
pip install -e .
```
PyTorch is defined as an optional dependency because the correct build depends on your hardware (CPU vs. CUDA).

**CPU-only**:
```bash
pip install -e .[torch]
```
**GPU(CUDA)**:

For CUDA, do not rely on `.[torch]`; install PyTorch first from the official selector [here](https://pytorch.org/get-started/locally/)

### 3) Prepare data
1. Download the dataset from the Kaggle competition page [CSIRO - Image2Biomass Prediction Competition](https://www.kaggle.com/competitions/csiro-biomass/data) and unzip.
2. Create a `data/` directory in the project root (if it doesn’t exist) and place the downloaded files there.
3. Run the `data.ipynb` notebook **once** to preprocess/transform the data.

After running the notebook, it will generate `*_split.csv` files inside `data/` (you shall use them in your config).

### 4) Configure experiment

Edit the config file (example):
```
configs/baseline.yaml
```
Typical parameters include:
- `model_name`(e.g `efficientnet_b0`)
- `img_size`, `batch_size`
- `epochs`, `lr`
- staged fine-tuning settings (freeze -> unfreeze)

### 5) Run training
```bash
python src/biomass2pred/train/run.py configs/baseline.yaml
```

### 6) Evaluate

Validation metrics are printed during training and saved in logs/checkpoints (depending on config).