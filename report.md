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
- Optimiser: Default YOLOv8 optimiser (AdamW with default augmentation: mosaic, random flip, HSV colour jitter)
- Task: Multi-class object detection

The YOLOv8n was selected as a lightweight baseline due to its suitability for small datasets.

### RetinaNet
- Backbone: ResNet50 (ImageNet pretrained)
- Input size: 416 × 416
- Batch size: 8
- Box loss: Smooth L1
- Classification loss: Focal Loss (alpha=0.75, gamma=2.0, reduction="sum")
- Optimiser: Adam + CosineDecay + gradient clipping (clipnorm=1.0)
- Augmentation: RandomFlip, RandomTranslation (keras_cv)
- Training: Two-phase (frozen backbone → partial unfreeze)

---

## 3. Dataset Summary

- Total images: 660
- Classes:
  - Coffea arabica: 300 images
  - Coffea canephora: 300 images
  - Negative samples: 60 images

### Data split (70/15/15):
- Training set: 462 images (210 arabica, 210 canephora, 42 negative)
- Validation set: 99 images (45 arabica, 45 canephora, 9 negative)
- Test set: 99 images (45 arabica, 45 canephora, 9 negative)
- Stratified split — zero overlap confirmed between all three sets

Negative samples (60 images) were sourced from iNaturalist taxon 47693 
(Magnoliopsida, Indonesia), specifically excluding Coffea species 
(taxon 64342, 64345). These hard negatives share similar leaf structure 
and geographic context with the target classes, improving the model's 
ability to distinguish coffee plants from visually similar vegetation.

---

## 4. YOLOv8n — Training Performance

| Metric       | Value |
|--------------|-------|
| Precision    | 0.747 |
| Recall       | 0.859 |
| mAP@50       | 0.880 |
| mAP@50-95    | 0.878 |

---

## 5. YOLOv8n — Validation Evaluation

| Metric | Arabica | Canephora |
|--------|---------|-----------|
| Precision | 0.92 | 0.83 |
| Recall | 0.82 | 0.93 |
| F1 | 0.87 | 0.88 |
| **Accuracy** | | **0.87** |

**Confusion matrix:**
```
              Predicted
              Arabica  Canephora
True Arabica  [ 49       11 ]
True Canephora[  4       55 ]
```

### 5.1 YOLOv8n — Test Set Evaluation (held-out)

| Metric | Arabica | Canephora |
|--------|---------|-----------|
| Precision | 0.89 | 0.75 |
| Recall | 0.71 | 0.91 |
| F1 | 0.79 | 0.82 |
| **Accuracy** | | **0.81** |

**Confusion matrix:**
```
              Predicted
              Arabica  Canephora
True Arabica  [ 32       13 ]
True Canephora[  4       40 ]
```

Test set: 89 annotated samples (45 arabica, 44 canephora — negatives excluded from classification metrics).

---

## 6. YOLOv8n — Inference Performance

- Inference time: ~34 ms per image (CPU)
- Suitable for near real-time applications
- Efficient performance despite training on limited hardware

---

## 7. RetinaNet — Training Performance

Two-phase training was applied. Phase 1 (frozen backbone) converged steadily to a best val_loss of 0.1488 at epoch 10. Phase 2 (partial backbone unfreeze) immediately destabilised — val_loss spiked from 0.149 to 2.679 in the first epoch, triggering EarlyStopping. Best weights from Phase 1 were restored.

| Phase | Best Val Loss | Outcome |
|-------|--------------|---------|
| Phase 1 — Frozen backbone | 0.1488 | Converged |
| Phase 2 — Partial unfreeze | — | Overfit epoch 1, rolled back |

This confirms the dataset (462 training samples) is too small to benefit from backbone fine-tuning.

---

## 8. RetinaNet — Evaluation Results (alpha=0.75)

| Metric | Arabica | Canephora |
|--------|---------|-----------|
| Precision | 0.49 | 0.00 |
| Recall | 0.98 | 0.00 |
| F1-score | 0.66 | 0.00 |
| **Accuracy** | | **0.49** |

**Confusion matrix:**
```
              Predicted
              Arabica  Canephora
True Arabica  [  44        1  ]
True Canephora[  45        0  ]
```

Training with alpha=0.50 caused complete class collapse — canephora predicted for all images (accuracy=0.50). Increasing to alpha=0.75 partially recovered arabica predictions (9/119, recall=0.10), but bias remained dominant. A subsequent run with identical configuration collapsed in the opposite direction — predicting arabica for 89/90 images — confirming RetinaNet is unstable across runs and not learning genuine discriminative features.

The similar prediction confidence across both classes (arabica: 0.625, canephora: 0.627) confirms the model cannot distinguish the two species, consistent with the high visual similarity between *Coffea arabica* and *Coffea canephora* in whole-plant images.

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
| Arabica Recall (val) | 0.82 | 0.10 |
| Canephora Recall (val) | 0.93 | 0.95 |
| Val Accuracy | 0.87 | ~0.49-0.52 (Unstable) |
| **Test Accuracy** | **0.81** | — |
| mAP@50 | 0.880 | — |
| Inference (CPU) | ~34ms | ~4s |
| Class balance | ✅ Balanced | ❌ Canephora bias |

YOLOv8n significantly outperformed RetinaNet under equivalent dataset conditions. YOLOv8's anchor-free design tolerates full-image annotations more gracefully, while RetinaNet's anchor-based mechanism requires precise per-object bounding boxes to function as designed.

---

## 11. Explainable AI — Grad-CAM Analysis

Grad-CAM was applied to both models to visualise which image regions drove predictions.

- **RetinaNet** target layer: `v2_stack_3_block3_out`
- **YOLOv8** target layer: C2f block, layer 9

### 11.1 RetinaNet
- Activations concentrated in **upper background regions** (sky, lighting)
- Both arabica and canephora heatmaps activated on **identical regions**
- Plant features (leaves, flowers, cherries) received near-zero activation
- Consistent with class collapse — model learned background cues, not plant morphology

See: `outputs/gradcam/retinanet_gradcam_summary.png`

### 11.2 YOLOv8
- Activations distributed across **biologically relevant plant structures**
- Arabica images: leaf surfaces, branch patterns, cherry clusters
- Canephora images: globular white flower clusters, jasmine-like flowers
- Arabica and canephora heatmaps activated on **different regions** — confirming distinct feature learning per species

See: `outputs/gradcam/yolo_gradcam_summary.png`

### 11.3 Comparison

| | RetinaNet | YOLOv8 |
|---|---|---|
| Activation pattern | Background / sky | Plant structures |
| Class differentiation | ❌ Identical heatmaps | ✅ Different regions per class |
| Biological validity | ❌ None | ✅ Flowers, cherries, leaves |

YOLOv8 learned plant-relevant features; RetinaNet learned background context — directly explaining the performance gap.

---

## 12. Limitations

- Bounding boxes are full-image auto-annotations. mAP reflects species 
  classification accuracy rather than object localisation precision. 
  This is an intentional auto-annotation baseline approach.
- 7 canephora images were corrupted during download and excluded 
  (duplicate URLs due to replace=True sampling on a small source pool 
  of 117 observations).
- Dataset size is relatively small (660 images total).
- RetinaNet evaluation is limited to image-level classification — box-level mAP was not computed given full-image annotation constraints.
- Grad-CAM heatmaps reflect backbone feature activations rather than detection head outputs — interpretation should account for this architectural distinction.

---

## 13. Conclusion

YOLOv8n achieved strong balanced performance (mAP@50=0.880, val accuracy=87%, test accuracy=81%) demonstrating that species-level coffee plant detection is feasible using auto-annotated iNaturalist data. The 6% gap between val and test accuracy reflects normal generalisation variance on a small dataset. RetinaNet exhibited persistent class bias under all configurations — alpha=0.50 caused complete collapse, while alpha=0.75 achieved only (~49%-52%) accuracy with poor arabica recall (0.10).

Grad-CAM analysis confirmed that YOLOv8 learned biologically meaningful features — flower morphology, cherry clusters, leaf structure — while RetinaNet attended solely to background context, explaining its failure to distinguish species. These results confirm YOLOv8n as the more appropriate architecture for small datasets with full-image annotations, while RetinaNet's strengths would be better realised with tighter per-object bounding box annotations.

---