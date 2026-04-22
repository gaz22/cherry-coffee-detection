import os
import json
import cv2

YOLO_LABEL_DIR = "data/yolo/labels/train"
IMAGE_DIR = "data/yolo/images/train"
OUTPUT_JSON = "data/coco/train.json"


def yolo_to_coco_bbox(x, y, width, height, image_width, image_height):
    x_min = (x - width / 2) * image_width
    y_min = (y - height / 2) * image_height
    box_width = width * image_width
    box_height = height * image_height
    return [x_min, y_min, box_width, box_height]  # FIXED


def main():
    print("Convert YOLO → COCO format...")

    images = []
    annotations = []

    categories = [
        {"id": 1, "name": "arabica"},
        {"id": 2, "name": "canephora"}
    ]

    annotation_id = 0
    image_id = 0

    for file in os.listdir(YOLO_LABEL_DIR):
        if not file.endswith(".txt"):
            continue

        image_name = file.replace(".txt", ".jpg")
        image_path = os.path.join(IMAGE_DIR, image_name)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)

        if image is None:  # FIXED
            print("Skipping broken image:", image_path)
            continue

        height, width = image.shape[:2]

        images.append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        with open(os.path.join(YOLO_LABEL_DIR, file), "r") as f:
            for line in f.readlines():
                parts = line.strip().split()

                # allow empty label (negative sample)
                if len(parts) == 0:
                    continue

                if len(parts) != 5:
                    continue

                cls, x, y, bw, bh = map(float, parts)

                bbox = yolo_to_coco_bbox(x, y, bw, bh, width, height)

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,  # FIXED
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })

                annotation_id += 1

        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs("data/coco", exist_ok=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco, f)

    print("COCO dataset created successfully")
    print("Images:", len(images))
    print("Annotations:", len(annotations))


if __name__ == "__main__":
    main()