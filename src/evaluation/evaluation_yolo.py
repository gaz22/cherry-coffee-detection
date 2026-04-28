import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from ultralytics import YOLO

COCO_JSON    = "data/coco/val.json"
IMAGE_DIR    = "data/yolo/images/val"
YOLO_WEIGHTS = "runs/detect/train-3/weights/best.pt"
OUTPUT_DIR   = "outputs"

CONFIDENCE_THRESHOLD = 0.25
CLASS_NAMES = ["arabica", "canephora"]


def load_val_data():
    print("Loading val dataset...")

    with open(COCO_JSON) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    dataset = []
    arabica_count   = 0
    canephora_count = 0

    for image_id, img in images.items():
        path = os.path.join(IMAGE_DIR, img["file_name"])
        if not os.path.exists(path):
            continue
        anns = annotations.get(image_id, [])
        if not anns:
            continue
        cls = int(anns[0]["category_id"]) - 1
        dataset.append({"image_path": path, "class": cls})
        if cls == 0:
            arabica_count += 1
        else:
            canephora_count += 1

    print(f"Loaded: {len(dataset)} samples")
    print(f"Arabica: {arabica_count} | Canephora: {canephora_count}")
    return dataset


def evaluate(model, dataset):
    print(f"\nRunning evaluation (threshold: {CONFIDENCE_THRESHOLD})...")

    y_true, y_pred, y_conf = [], [], []

    for item in dataset:
        results = model.predict(
            item["image_path"],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        true_label = item["class"]
        boxes = results[0].boxes

        if len(boxes) > 0:
            confs   = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            best    = np.argmax(confs)
            pred_label = classes[best]
            pred_conf  = float(confs[best])
        else:
            pred_label = 0
            pred_conf  = 0.0

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_conf.append(pred_conf)

    print(f"Prediction distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    return np.array(y_true), np.array(y_pred), np.array(y_conf)


def report(y_true, y_pred):
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    ))


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    ).plot(cmap="Blues", ax=ax)
    plt.title("YOLOv8n — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix_yolo.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def plot_confidence_distribution(y_true, y_pred, y_conf):
    correct   = y_conf[y_true == y_pred]
    incorrect = y_conf[y_true != y_pred]

    plt.figure(figsize=(8, 4))
    plt.hist(correct,   bins=20, alpha=0.6, color="green", label="Correct")
    plt.hist(incorrect, bins=20, alpha=0.6, color="red",   label="Wrong")
    plt.axvline(
        CONFIDENCE_THRESHOLD, color="black", linestyle="--",
        label=f"Threshold ({CONFIDENCE_THRESHOLD})"
    )
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.title("YOLOv8n — Prediction Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confidence_distribution_yolo.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"Saved: {path}")


def main():
    print("Starting YOLOv8 Evaluation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(YOLO_WEIGHTS):
        print(f" Weights not found: {YOLO_WEIGHTS}")
        print("   Check runs/detect/ for your latest training folder.")
        return

    dataset = load_val_data()
    model   = YOLO(YOLO_WEIGHTS)

    y_true, y_pred, y_conf = evaluate(model, dataset)

    report(y_true, y_pred)
    plot_confusion(y_true, y_pred)
    plot_confidence_distribution(y_true, y_pred, y_conf)

    print("\n YOLOv8 evaluation complete.")
    print(f"   Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()