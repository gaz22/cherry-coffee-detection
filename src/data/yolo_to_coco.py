import os
import json
import cv2

SPLITS = [
    ("data/yolo/labels/train", "data/yolo/images/train", "data/coco/train.json"),
    ("data/yolo/labels/val",   "data/yolo/images/val",   "data/coco/val.json"),
]


def yolo_to_coco_bbox(x, y, width, height, image_width, image_height):
    x_min = (x - width / 2) * image_width
    y_min = (y - height / 2) * image_height
    box_width = width * image_width
    box_height = height * image_height
    return [x_min, y_min, box_width, box_height]


def main():
    print("Convert YOLO → COCO format...")

    os.makedirs("data/coco", exist_ok=True)

    categories = [
        {"id": 1, "name": "arabica"},
        {"id": 2, "name": "canephora"}
    ]

    for label_dir, image_dir, output_json in SPLITS:

        images = []
        annotations = []
        annotation_id = 0
        image_id = 0

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue

            image_name = file.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)

            if image is None:
                print("Skipping broken image:", image_path)
                continue

            height, width = image.shape[:2]

            images.append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height
            })

            with open(os.path.join(label_dir, file), "r") as f:
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
                        "category_id": int(cls) + 1,
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

        with open(output_json, "w") as f:
            json.dump(coco, f)

        print(f"\n{output_json}")
        print(f"  Images:      {len(images)}")
        print(f"  Annotations: {len(annotations)}")

    print("\nCOCO dataset created successfully")


if __name__ == "__main__":
    main()