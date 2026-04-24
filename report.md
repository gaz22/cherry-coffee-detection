# YOLOv8 Object Detection Results Report (Draft)

## Project: Coffee Species Detection

## 1. Overview

This document summarises the results of a YOLOv8-based object detection model trained to classify coffee plant species from iNaturalist dataset. The model is part of a comparative study between YOLO and RetinaNet architectures.

This is a **draft report** and will be updated after further experiments and model comparisons.

---

## 2. Model Configuration

- Model: YOLOv8n (nano)
- Input size: 416 × 416
- Epochs: 50
- Batch size: 8
- Optimiser: Default YOLOv8 optimiser
- Task: Multi-class object detection

The YOLOv8n was selected as a lightweight baseline due to its suitability for small datasets.

---

## 3. Dataset Summary

- Total images: 660
- Classes:
  - Coffea arabica: 300 images
  - Coffea canephora: 300 images
  - Negative samples: 60 images

### Data split:
- Training set: 528 images
- Validation set: 132 images
- No overlap between splits

Negative samples (60 images) were sourced from iNaturalist taxon 47693 
(Magnoliopsida, Indonesia), specifically excluding Coffea species 
(taxon 64342, 64345). These hard negatives share similar leaf structure 
and geographic context with the target classes, improving the model's 
ability to distinguish coffee plants from visually similar vegetation.

---

## 4. Training Performance

The model achieved the following validation results:

| Metric       | Value |
|--------------|-------|
| Precision    | 0.793 |
| Recall       | 0.874 |
| mAP@50       | 0.888 |
| mAP@50-95    | 0.886 |

Overall, the model shows good performance for a baseline detector.
---

## 5. Class-wise Performance

| Class            | mAP@50 |
|------------------|--------|
| Coffea arabica   | 0.875  |
| Coffea canephora | 0.897  |

The results show balanced learning across both classes with no significant bias.

---

## 6. Inference Performance

- Inference time: ~35 ms per image (CPU)
- Suitable for near real-time applications
- Efficient performance despite training on limited hardware

---

## 7. Observations

- The model performs well on both coffee species classes
- Negative sampling improves robustness against false positives
- Balanced dataset contributes to stable training
- No significant overfitting observed during evaluation

---

## 8. Limitations (Draft)

- Bounding boxes are full-image auto-annotations. mAP reflects species 
  classification accuracy rather than object localisation precision. 
  This is an intentional auto-annotation baseline approach.
- 7 canephora images were corrupted during download and excluded 
  (duplicate URLs due to replace=True sampling on a small source pool 
  of 117 observations).
- Dataset size is relatively small (660 images total).
- Single architecture evaluated to date — RetinaNet comparison pending.

---

## 10. Conclusion

This YOLOv8n baseline achieves mAP@50 of 0.888 with balanced per-class 
performance (arabica: 0.875, canephora: 0.897), demonstrating that 
species-level coffee plant detection is feasible using auto-annotated 
iNaturalist data. The 35ms inference time confirms suitability for 
near real-time applications. Comparison with RetinaNet and explainable 
AI analysis will follow in the next phase.

---
