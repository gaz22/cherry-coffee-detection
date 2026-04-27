import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch
from ultralytics import YOLO

COCO_JSON  = "data/coco/val.json"
IMAGE_DIR  = "data/yolo/images/val"
OUTPUT_DIR = "outputs/gradcam"
IMG_SIZE   = 416

CLASS_NAMES = ["arabica", "canephora"]

# latest YOLO run weights
YOLO_WEIGHTS = "runs/detect/train-3/weights/best.pt"

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


# manual GradCAM via hooks
class GradCAMYOLO:
    def __init__(self, model):
        self.model     = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # target the last backbone layer in YOLOv8n (layer 9 — C2f)
        target = self.model.model.model[9]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)

    def compute(self, image_path, class_idx):
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        img_tensor = torch.from_numpy(
            img_resized.transpose(2, 0, 1)
        ).float().unsqueeze(0) / 255.0

        img_tensor.requires_grad = True

        self.model.model.train()
        output = self.model.model(img_tensor)

        # scores shape: [1, num_classes, 3549]
        # class_idx 0=arabica, 1=canephora
        class_scores = output["scores"][0, class_idx, :]
        score = class_scores.max()

        self.model.model.zero_grad()
        score.backward()

        # [C, H, W]
        grads = self.gradients[0]   
        acts  = self.activations[0]

        weights = grads.mean(dim=[1, 2])
        heatmap = (weights[:, None, None] * acts).sum(dim=0)
        heatmap = torch.relu(heatmap).numpy()

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap, img_resized


# overlay heatmap
def overlay_heatmap(heatmap, image_np, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    image_float = image_np.astype(float) / 255.0
    overlaid = heatmap_colored * alpha + image_float * (1 - alpha)
    overlaid = np.clip(overlaid, 0, 1)
    return overlaid


# plot grid
def plot_gradcam_grid(samples, gradcam, title, filename):
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    fig.suptitle(title, fontsize=14)

    for row, item in enumerate(samples):
        true_cls = item["class"]

        heatmap_a, img_np = gradcam.compute(item["image_path"], class_idx=0)
        heatmap_c, _      = gradcam.compute(item["image_path"], class_idx=1)

        overlay_a = overlay_heatmap(heatmap_a, img_np)
        overlay_c = overlay_heatmap(heatmap_c, img_np)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(
            f"Original\nTrue: {CLASS_NAMES[true_cls]}", fontsize=9
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_a)
        axes[row, 1].set_title("GradCAM: Arabica", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_c)
        axes[row, 2].set_title("GradCAM: Canephora", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


def main():
    print("YOLOv8 GradCAM Analysis...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(YOLO_WEIGHTS):
        print(f" Weights not found: {YOLO_WEIGHTS}")
        print("   Check runs/detect/ for your latest training run folder.")
        return

    model  = YOLO(YOLO_WEIGHTS)
    gradcam = GradCAMYOLO(model)

    samples = load_samples()
    arabica_samples   = [s for s in samples if s["class"] == 0]
    canephora_samples = [s for s in samples if s["class"] == 1]

    print(f"Generating GradCAM for {len(arabica_samples)} arabica samples...")
    plot_gradcam_grid(
        arabica_samples,
        gradcam,
        title="YOLOv8 GradCAM — Arabica Samples",
        filename="yolo_gradcam_arabica.png"
    )

    print(f"Generating GradCAM for {len(canephora_samples)} canephora samples...")
    plot_gradcam_grid(
        canephora_samples,
        gradcam,
        title="YOLOv8 GradCAM — Canephora Samples",
        filename="yolo_gradcam_canephora.png"
    )

    # Summary — 2 of each
    print("Generating combined summary...")
    summary = arabica_samples[:2] + canephora_samples[:2]
    plot_gradcam_grid(
        summary,
        gradcam,
        title="YOLOv8 GradCAM — Summary (Arabica & Canephora)",
        filename="yolo_gradcam_summary.png"
    )

    print("\n YOLOv8 GradCAM complete.")
    print(f"   Outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()