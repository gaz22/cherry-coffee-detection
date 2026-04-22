# Coffee Species Detection (Object Detection Project)

## Overview

Detect and classify Coffee plant species using iNaturalist datasets. Build and compare multiple object detection models capable of distinguishing between coffee species and non-target vegetation.

The project focuses on:
- Coffea arabica
- Coffea canephora

A negative sampling strategy is also used to improve model robustness.

---

## Dataset

- Source: iNaturalist
- Region: West Java, Indonesia
- Classes:
  - Coffea arabica
  - Coffea canephora
  - Negative (non-coffee vegetation)

## Models

The project implements and compares multiple object detection architectures:

- YOLOv8 (baseline model)
- RetinaNet (planned / in progress)

---

## Tech Stack

- Python
- PyTorch / Ultralytics YOLOv8
- TensorFlow / KerasCV (RetinaNet)
- Pandas, NumPy
- OpenCV

---

## Project Structure

```
data/
notebooks/
reports/
src/
outputs/
```

## Usage

```bash
pip install -r requirements.txt
python -m src.data.download_data
python -m src.data.create_yolo_dataset
python src/training/train_yolo.py
```
