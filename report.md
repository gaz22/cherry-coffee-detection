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

Negative samples were included to improve model robustness by helping the detector differentiate coffee plants from non-target vegetation.

---

## 4. Training Performance

The model achieved the following validation results:

| Metric | Value |
|--------|------|
| Precision | 0.79 |
| Recall | 0.78 |
| mAP@50 | 0.85 |
| mAP@50–95 | 0.85 |

Overall, the model shows good performance for a baseline detector.

---

## 5. Class-wise Performance

| Class | Performance (mAP approx.) |
|------|----------------------------|
| Coffea arabica | 0.848 |
| Coffea canephora | 0.851 |

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

- Dataset size is relatively small
- Limited diversity in negative samples
- Evaluation restricted to a single model (YOLOv8n)

---

## 10. Conclusion

This initial YOLOv8 model provides a strong foundation for coffee species detection. The results demonstrate that deep learning-based object detection is effective for distinguishing coffee plant species under real-world conditions. Further improvements and model comparisons will be conducted in the next phase of the project.

---
