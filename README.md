# Coffee Species Detection (Object Detection Project)

## Overview

Detect and classify Coffee plant species using iNaturalist datasets. Build and compare multiple object detection models capable of distinguishing between coffee species and non-target vegetation.

The project focuses on:
- *Coffea arabica*
- *Coffea canephora*

A negative sampling strategy is also used to improve model robustness. Grad-CAM explainability is applied to both models to visualise learned features.

---

## Dataset

- Source: iNaturalist
- Region: Indonesia (place_id=6966)
- Split: 70% train / 15% val / 15% test (stratified)
- Classes:
  - Coffea arabica — 300 images (taxon 64342)
  - Coffea canephora — 300 images (taxon 64345)
  - Negative (non-coffee vegetation) — 60 images (taxon 47693)

### Download iNaturalist data

Go to https://www.inaturalist.org/observations/export and download three CSV exports:

**Coffea arabica** → save as `observations-712548_csv.zip`
```
taxon_id=64342, quality_grade=research, has[]=photos
```

**Coffea canephora** → save as `observations-713339_csv.zip`
```
taxon_id=64345, quality_grade=research, has[]=photos
```

**Negative samples** → save as `observations-716817_csv.zip`
```
taxon_id=47693, without_taxon_id=64342,64345
place_id=6966, quality_grade=research
d1=2020-01-01, d2=2024-12-31
```

Place all three zip files in `data/raw/` before running the pipeline.

---

## Models

- **YOLOv8n** — lightweight anchor-free detector (test accuracy: 81%, mAP@50: 0.880)
- **RetinaNet** (ResNet50 backbone) — anchor-based detector with focal loss

---

## Tech Stack

- Python 3.11.13
- PyTorch 2.2.2 / Ultralytics 8.4.40 — YOLOv8
- TensorFlow 2.16.2 / keras-cv 0.9.0 — RetinaNet
- Pandas, NumPy, OpenCV
- scikit-learn — evaluation metrics
- Matplotlib — visualisation

---

## Project Structure

```
coffee-cherry-detection/
├── data/
│   ├── coco/               # COCO format annotations (train/val/test)
│   ├── raw/                # iNaturalist CSVs (place zip files here)
│   └── yolo/               # YOLO format images and labels
├── data/retinanet/         # RetinaNet weights
├── src/
│   ├── data/               # download, dataset, conversion scripts
│   ├── training/           # train_yolo.py, train_retinanet.py
│   ├── evaluation/         # evaluation + gradcam scripts
│   └── utils/              # config.py
├── outputs/                # confusion matrices, gradcam outputs
├── runs/detect/coffee/     # YOLOv8 training outputs (auto-generated)
├── notebooks/
│   └── coffee_species_detection.ipynb
├── report.md
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd coffee-cherry-detection
pip install -r requirements.txt
```

---

## Run Notebook

```bash
# Activate virtual environment
source venv/bin/activate

# Install Jupyter inside venv (first time only)
pip install jupyter ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# Launch from project root
jupyter notebook notebooks/coffee_species_detection.ipynb
```

When the notebook opens select **Kernel → Change Kernel → Python (venv)**.

The notebook automatically resolves the project root — no manual path configuration needed.

---

## Run Pipeline

```bash
# 1. Download images from iNaturalist CSVs
python -m src.data.download_data

# 2. Build YOLO train/val/test split
#    dataset.yaml auto-created on first run
python -m src.data.create_yolo_dataset

# 3. Convert to COCO format for RetinaNet
python -m src.data.yolo_to_coco

# 4. Train YOLOv8n (~2 hours on CPU)
python -m src.training.train_yolo

# 5. Train RetinaNet (~40 minutes on CPU)
python -m src.training.train_retinanet

# 6. Evaluate both models
python -m src.evaluation.evaluation_yolo
python -m src.evaluation.evaluation_retinanet

# 7. Generate Grad-CAM outputs
python -m src.evaluation.gradcam_yolo
python -m src.evaluation.gradcam_retinanet
```

---

## Results

| Metric | YOLOv8n | RetinaNet (α=0.75) |
|--------|---------|---------------------|
| mAP@50 | 0.880 | — |
| Val Accuracy | 87% | ~50% |
| **Test Accuracy** | **81%** | — |
| Class balance | ✅ Balanced | ❌ Unstable |
| Inference (CPU) | ~34ms | ~4s |

See `report.md` for full results and Grad-CAM analysis.

---

## Known Issues

- 7 canephora images corrupted during download — skipped automatically
- `dataset.yaml` auto-created on first run — not overwritten on re-runs
- RetinaNet Phase 2 fine-tuning destabilises on small datasets — best weights always restored from Phase 1
- Notebook must be launched from project root — see Run Notebook section above