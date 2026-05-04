import os
import numpy as np
import tensorflow as tf
import keras_cv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

COCO_JSON  = "data/coco/val.json"
IMAGE_DIR  = "data/yolo/images/val"
MODEL_PATH = "outputs/retinanet_best.weights.h5"
OUTPUT_DIR = "outputs/gradcam"
IMG_SIZE   = 416

# last conv layer in ResNet50
TARGET_LAYER = "v2_stack_3_block3_out"

CLASS_NAMES = ["arabica", "canephora"]


# load model
def load_model():
    print("Loading RetinaNet...")
    backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
    model = keras_cv.models.RetinaNet(
        backbone=backbone,
        num_classes=2,
        bounding_box_format="xyxy"
    )
    model.load_weights(MODEL_PATH)
    return model, backbone

# Build GradCAM model
# Outputs: [feature_maps, raw_logits]
def build_gradcam_model(backbone):
    # Get the target conv layer output
    target_layer = backbone.get_layer(TARGET_LAYER)

    gradcam_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output]
    )
    return gradcam_model

# preprocess image
def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


# compute Grad-CAM heatmap
def compute_gradcam(gradcam_model, image_tensor, class_idx):
    img_batch = tf.expand_dims(image_tensor, axis=0)

    with tf.GradientTape() as tape:
        # watch the feature maps
        conv_outputs, backbone_outputs = gradcam_model(img_batch, training=False)
        tape.watch(conv_outputs)

        # global average pooled features as proxy for class score
        # (RetinaNet head is complex — backbone features are the signal)
        pooled = tf.reduce_mean(backbone_outputs, axis=[1, 2])
        class_score = pooled[:, class_idx]

    # gradients of class score w.r.t. feature maps
    grads = tape.gradient(class_score, conv_outputs)

    # global average pool the gradients
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])

    # weight feature maps by gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# overlay heatmap on image
def overlay_heatmap(heatmap, image_np, alpha=0.4):
    # resize heatmap to image size
    heatmap_resized = np.array(
        tf.image.resize(heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE))
    ).squeeze()

    # apply colormap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]

    # overlay
    overlaid = heatmap_colored * alpha + image_np * (1 - alpha)
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid, heatmap_resized


# load val samples
def load_samples():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    arabica_samples   = []
    canephora_samples = []

    for image_id, img in images.items():
        path = os.path.join(IMAGE_DIR, img["file_name"])
        if not os.path.exists(path):
            continue
        anns = annotations.get(image_id, [])
        if not anns:
            continue
        cls = int(anns[0]["category_id"]) - 1
        item = {"image_path": path, "class": cls}
        if cls == 0 and len(arabica_samples) < 4:
            arabica_samples.append(item)
        elif cls == 1 and len(canephora_samples) < 4:
            canephora_samples.append(item)
        if len(arabica_samples) == 4 and len(canephora_samples) == 4:
            break

    return arabica_samples + canephora_samples

# plot GradCAM grid
def plot_gradcam_grid(samples, gradcam_model, title, filename):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(title, fontsize=14)

    for row, item in enumerate(samples):
        image = preprocess_image(item["image_path"])
        image_np = image.numpy()
        true_cls = item["class"]

        # compute GradCAM for both classes
        heatmap_arabica   = compute_gradcam(gradcam_model, image, class_idx=0)
        heatmap_canephora = compute_gradcam(gradcam_model, image, class_idx=1)

        overlay_a, _ = overlay_heatmap(heatmap_arabica,   image_np)
        overlay_c, _ = overlay_heatmap(heatmap_canephora, image_np)

        # original image
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(
            f"Original\nTrue: {CLASS_NAMES[true_cls]}", fontsize=9
        )
        axes[row, 0].axis("off")

        # GradCAM — arabica
        axes[row, 1].imshow(overlay_a)
        axes[row, 1].set_title("GradCAM: Arabica", fontsize=9)
        axes[row, 1].axis("off")

        # GradCAM — canephora
        axes[row, 2].imshow(overlay_c)
        axes[row, 2].set_title("GradCAM: Canephora", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")

def main():
    print("RetinaNet GradCAM Analysis...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, backbone = load_model()
    gradcam_model   = build_gradcam_model(backbone)

    print(f"Target layer: {TARGET_LAYER}")
    print(f"Output shape: {gradcam_model.output[0].shape}")

    samples = load_samples()
    arabica_samples   = [s for s in samples if s["class"] == 0]
    canephora_samples = [s for s in samples if s["class"] == 1]

    print(f"\nGenerating GradCAM for {len(arabica_samples)} arabica samples...")
    plot_gradcam_grid(
        arabica_samples,
        gradcam_model,
        title="RetinaNet GradCAM — Arabica Samples",
        filename="retinanet_gradcam_arabica.png"
    )

    print(f"\nGenerating GradCAM for {len(canephora_samples)} canephora samples...")
    plot_gradcam_grid(
        canephora_samples,
        gradcam_model,
        title="RetinaNet GradCAM — Canephora Samples",
        filename="retinanet_gradcam_canephora.png"
    )

    # Combined summary — 2 arabica + 2 canephora
    print("\nGenerating combined summary...")
    summary_samples = arabica_samples[:2] + canephora_samples[:2]
    plot_gradcam_grid(
        summary_samples,
        gradcam_model,
        title="RetinaNet GradCAM — Summary (Arabica & Canephora)",
        filename="retinanet_gradcam_summary.png"
    )

    print("\n RetinaNet GradCAM complete.")
    print(f"   Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()