# Object Detection Results Report (YOLOv8n vs RetinaNet)

## Project: Coffee Species Detection

## 1. Overview

This document summarises the results of two object detection architectures — YOLOv8n and RetinaNet — trained to classify coffee plant species from iNaturalist dataset. This is part of a comparative study between the two architectures.

---

## 2. Model Configuration

### YOLOv8n
- Model: YOLOv8n (nano)
- Input size: 416 × 416
- Epochs: 50
- Batch size: 8
- Optimiser: Default YOLOv8 optimiser
- Task: Multi-class object detection

The YOLOv8n was selected as a lightweight baseline due to its suitability for small datasets.

### RetinaNet
- Backbone: ResNet50 (ImageNet pretrained)
- Input size: 416 × 416
- Batch size: 8
- Box loss: Smooth L1
- Classification loss: Focal Loss (alpha=0.75, gamma=2.0, reduction="sum")
- Optimiser: Adam + CosineDecay + gradient clipping (clipnorm=1.0)
- Training: Two-phase (frozen backbone → partial unfreeze)

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

## 4. YOLOv8n — Training Performance

The model achieved the following validation results:

| Metric       | Value |
|--------------|-------|
| Precision    | 0.793 |
| Recall       | 0.874 |
| mAP@50       | 0.888 |
| mAP@50-95    | 0.886 |

Overall, the model shows good performance for a baseline detector.

---

## 5. YOLOv8n — Class-wise Performance

| Class            | mAP@50 |
|------------------|--------|
| Coffea arabica   | 0.875  |
| Coffea canephora | 0.897  |

The results show balanced learning across both classes with no significant bias.

---

## 6. YOLOv8n — Inference Performance

- Inference time: ~35 ms per image (CPU)
- Suitable for near real-time applications
- Efficient performance despite training on limited hardware

---

## 7. RetinaNet — Training Performance

Two-phase training was applied. Phase 1 (frozen backbone) converged steadily to a best val_loss of 0.1488 at epoch 10. Phase 2 (partial backbone unfreeze) immediately destabilised — val_loss spiked from 0.149 to 2.679 in the first epoch, triggering EarlyStopping. Best weights from Phase 1 were restored.

| Phase | Best Val Loss | Outcome |
|-------|--------------|---------|
| Phase 1 — Frozen backbone | 0.1488 | Converged |
| Phase 2 — Partial unfreeze | — | Overfit epoch 1, rolled back |

This confirms the dataset (474 training samples) is too small to benefit from backbone fine-tuning.

---

## 8. RetinaNet — Evaluation Results (alpha=0.75)

| Metric | Arabica | Canephora |
|--------|---------|-----------|
| Precision | 0.67 | 0.51 |
| Recall | 0.10 | 0.95 |
| F1-score | 0.17 | 0.66 |
| **Accuracy** | | **0.52** |

**Confusion matrix:**
```
              Predicted
              Arabica  Canephora
True Arabica  [  6        54  ]
True Canephora[  3        56  ]
```

A prior training run with alpha=0.50 (equal class weight) resulted in complete class collapse — the model predicted canephora for all 119 images (arabica recall=0.00, accuracy=0.50). Increasing alpha to 0.75 to upweight arabica partially recovered predictions, introducing 9 arabica predictions (prediction distribution: {arabica: 9, canephora: 110}). However, canephora bias remained dominant with arabica recall at only 0.10.

The average prediction confidence was similar across true classes (arabica images: 0.625, canephora images: 0.627), suggesting the model assigns nearly identical scores to both species — consistent with the high visual similarity between *Coffea arabica* and *Coffea canephora* in whole-plant images.

---

## 9. Observations

### YOLOv8n
- Balanced learning across both coffee species with no class collapse
- Negative sampling improves robustness against false positives
- Balanced dataset contributes to stable training
- No significant overfitting observed during evaluation

### RetinaNet
- Persistent canephora bias across all configurations
- alpha=0.50 caused complete collapse — no arabica predictions
- alpha=0.75 partially recovered arabica but recall remained poor (0.10)
- Phase 2 fine-tuning destabilised training on this dataset size
- Similar confidence scores for both classes confirms the visual similarity challenge

---

## 10. Model Comparison

| Metric | YOLOv8n | RetinaNet (α=0.75) |
|--------|---------|---------------------|
| Arabica Recall | — | 0.10 |
| Canephora Recall | — | 0.95 |
| mAP@50 / Accuracy | 0.888 | 0.52 |
| Inference speed (CPU) | ~35ms | ~4s |
| Class balance | ✅ Balanced | ❌ Canephora bias |
| Training stability | ✅ Stable | ⚠️ Phase 2 unstable |

YOLOv8n significantly outperformed RetinaNet under equivalent dataset conditions. YOLOv8's anchor-free design tolerates full-image annotations more gracefully, while RetinaNet's anchor-based mechanism requires precise per-object bounding boxes to function as designed.

---

## 11. Limitations

- Bounding boxes are full-image auto-annotations. mAP reflects species 
  classification accuracy rather than object localisation precision. 
  This is an intentional auto-annotation baseline approach.
- 7 canephora images were corrupted during download and excluded 
  (duplicate URLs due to replace=True sampling on a small source pool 
  of 117 observations).
- Dataset size is relatively small (660 images total).
- RetinaNet evaluation is limited to image-level classification, while box-level mAP was not computed given full-image annotation constraints.

---

## 12. Conclusion

YOLOv8n achieved strong balanced performance (mAP@50=0.888, arabica: 0.875, canephora: 0.897) with fast inference (~35ms/image CPU), demonstrating that species-level coffee plant detection is feasible using auto-annotated iNaturalist data. RetinaNet exhibited persistent class bias under all configurations — alpha=0.50 caused complete collapse, while alpha=0.75 achieved only 52% accuracy with poor arabica recall (0.10). These results confirm YOLOv8n as the more appropriate architecture for small datasets with full-image annotations, while RetinaNet's strengths would be better realised with tighter per-object bounding box annotations.

---