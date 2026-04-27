import tensorflow as tf
import keras_cv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

COCO_JSON  = "data/coco/val.json"
IMAGE_DIR  = "data/yolo/images/val"
MODEL_PATH = "outputs/retinanet_best.weights.h5"

CONFIDENCE_THRESHOLD = 0.15
IMG_SIZE = 416


def load_coco():
    print("Loading val dataset...")

    with open(COCO_JSON) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    dataset = []
    arabica_boxes   = 0
    canephora_boxes = 0

    for image_id, img in images.items():
        path = os.path.join(IMAGE_DIR, img["file_name"])
        if not os.path.exists(path):
            continue

        boxes, classes = [], []
        for ann in annotations.get(image_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            cls = int(ann["category_id"]) - 1
            classes.append(cls)
            if cls == 0:
                arabica_boxes += 1
            else:
                canephora_boxes += 1

        if len(boxes) == 0:
            continue

        dataset.append({
            "image_path": path,
            "boxes":   np.array(boxes,   dtype=np.float32),
            "classes": np.array(classes, dtype=np.int32)
        })

    print(f"Loaded: {len(dataset)} samples")
    print(f"Arabica boxes: {arabica_boxes} | Canephora boxes: {canephora_boxes}")
    return dataset


def load_model():
    print("Loading RetinaNet model...")

    backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
    model = keras_cv.models.RetinaNet(
        backbone=backbone,
        num_classes=2,
        bounding_box_format="xyxy"
    )
    model.load_weights(MODEL_PATH)
    print("Weights loaded:", MODEL_PATH)
    return model


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def preprocess_to_dataset(image_tensor):
    return tf.data.Dataset.from_tensors(image_tensor).batch(1)


def debug_predictions(model, dataset):
    print("\n=== DEBUG: First 3 samples ===")
    for item in dataset[:3]:
        image = preprocess_image(item["image_path"])
        ds    = preprocess_to_dataset(image)
        preds = model.predict(ds, verbose=0)

        pred_classes = preds["classes"][0]
        pred_scores  = preds["confidence"][0]
        mask = pred_scores > CONFIDENCE_THRESHOLD

        print(f"\nTrue class:                   {item['classes']}")
        print(f"Pred classes above threshold: {pred_classes[mask]}")
        print(f"Pred scores above threshold:  {pred_scores[mask]}")
        print(f"Unique pred classes (all):    {np.unique(pred_classes)}")
        print(f"Max score:                    {pred_scores.max():.3f}")
        print(f"Min score:                    {pred_scores.min():.3f}")


def evaluate(model, dataset):
    print(f"\nRunning evaluation (threshold: {CONFIDENCE_THRESHOLD})...")

    y_true, y_pred = [], []
    score_log = {"arabica": [], "canephora": []}

    for item in dataset:
        image = preprocess_image(item["image_path"])
        ds    = preprocess_to_dataset(image)
        preds = model.predict(ds, verbose=0)

        pred_classes = preds["classes"][0]
        pred_scores  = preds["confidence"][0]

        mask = pred_scores > CONFIDENCE_THRESHOLD
        pred_classes_filtered = pred_classes[mask]
        pred_scores_filtered  = pred_scores[mask]

        true_classes = item["classes"]
        true_label   = 1 if 1 in true_classes else 0

        # use highest confidence prediction
        if len(pred_scores_filtered) > 0:
            best_idx   = np.argmax(pred_scores_filtered)
            pred_label = int(pred_classes_filtered[best_idx])
        else:
            pred_label = 0

        if true_label == 0:
            score_log["arabica"].extend(pred_scores_filtered.tolist())
        else:
            score_log["canephora"].extend(pred_scores_filtered.tolist())

        y_true.append(true_label)
        y_pred.append(pred_label)

    if score_log["arabica"]:
        print(f"  Avg score on arabica images:   {np.mean(score_log['arabica']):.3f}")
    if score_log["canephora"]:
        print(f"  Avg score on canephora images: {np.mean(score_log['canephora']):.3f}")

    print(f"\nPrediction distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")

    return np.array(y_true), np.array(y_pred)


def report(y_true, y_pred):
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["arabica", "canephora"],
        zero_division=0
    ))


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== CONFUSION MATRIX ===")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["arabica", "canephora"]
    ).plot(cmap="Blues", ax=ax)
    plt.title("RetinaNet — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/retinanet/confusion_matrix_retinanet.png", dpi=150)
    plt.show()
    print("Saved: outputs/retinanet/confusion_matrix_retinanet.png")


def plot_score_distribution(model, dataset):
    print("Plotting score distribution...")

    arabica_scores, canephora_scores = [], []

    for item in dataset:
        image = preprocess_image(item["image_path"])
        ds    = preprocess_to_dataset(image)
        preds = model.predict(ds, verbose=0)
        scores = preds["confidence"][0]

        if 1 in item["classes"]:
            canephora_scores.extend(scores.tolist())
        else:
            arabica_scores.extend(scores.tolist())

    plt.figure(figsize=(8, 4))
    plt.hist(arabica_scores,   bins=30, alpha=0.6, label="Arabica",   color="green")
    plt.hist(canephora_scores, bins=30, alpha=0.6, label="Canephora", color="red")
    plt.axvline(
        CONFIDENCE_THRESHOLD, color="black", linestyle="--",
        label=f"Threshold ({CONFIDENCE_THRESHOLD})"
    )
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.title("RetinaNet — Score Distribution by True Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/retinanet/retinanet/score_distribution_retinanet.png", dpi=150)
    plt.show()
    print("Saved: outputs/retinanet/score_distribution_retinanet.png")


def main():
    print("Starting RetinaNet Evaluation...")
    os.makedirs("outputs", exist_ok=True)

    dataset = load_coco()
    model   = load_model()

    # debug: shows what model is actually predicting
    debug_predictions(model, dataset)

    y_true, y_pred = evaluate(model, dataset)

    report(y_true, y_pred)
    plot_confusion(y_true, y_pred)
    plot_score_distribution(model, dataset)

    print("\n Evaluation complete.")


if __name__ == "__main__":
    main()