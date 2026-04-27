import os
import json
import tensorflow as tf
import keras_cv

from src.utils.config import BATCH_SIZE

COCO_TRAIN_JSON = "data/coco/train.json"
COCO_VAL_JSON   = "data/coco/val.json"
TRAIN_IMAGE_DIR = "data/yolo/images/train"
VAL_IMAGE_DIR   = "data/yolo/images/val"
IMG_SIZE = 416


# load COCO splits
def load_coco_split(json_path, image_dir):
    with open(json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    annotations = {}
    for ann in coco["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    dataset = []
    for image_id, img in images.items():
        path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(path):
            continue
        boxes, classes = [], []
        for ann in annotations.get(image_id, []):
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            classes.append(int(ann["category_id"]) - 1)
        if not boxes:
            continue
        dataset.append({
            "image_path": path,
            "boxes": boxes,
            "classes": classes
        })

    print(f"  {json_path}: {len(dataset)} samples")
    return dataset


def load_coco(smoke=False):
    print("Loading COCO dataset...")
    train_data = load_coco_split(COCO_TRAIN_JSON, TRAIN_IMAGE_DIR)
    val_data   = load_coco_split(COCO_VAL_JSON,   VAL_IMAGE_DIR)

    if smoke:
        train_data = train_data[:16]
        val_data   = val_data[:8]
        print("  SMOKE MODE: using 16 train / 8 val samples")

    return train_data, val_data


# dataset pipeline
def build_dataset(dataset, training=True):

    def gen():
        for item in dataset:
            yield item

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "image_path": tf.TensorSpec((), tf.string),
            "boxes":      tf.TensorSpec((None, 4), tf.float32),
            "classes":    tf.TensorSpec((None,), tf.int32),
        }
    )

    def preprocess(x):
        image = tf.io.read_file(x["image_path"])
        image = tf.image.decode_jpeg(image, channels=3)

        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)

        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.cast(image, tf.float32) / 255.0

        scale_x = IMG_SIZE / w
        scale_y = IMG_SIZE / h

        boxes = x["boxes"]
        boxes = tf.stack([
            boxes[:, 0] * scale_x,
            boxes[:, 1] * scale_y,
            boxes[:, 2] * scale_x,
            boxes[:, 3] * scale_y,
        ], axis=-1)

        return {
            "images": image,
            "bounding_boxes": {
                "boxes":   boxes,
                "classes": x["classes"]
            }
        }

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmenter = tf.keras.Sequential([
            keras_cv.layers.RandomFlip(
                "horizontal", bounding_box_format="xyxy"
            ),
            keras_cv.layers.RandomTranslation(
                0.05, 0.05, bounding_box_format="xyxy"
            ),
        ])

        ds = ds.shuffle(512).repeat()
        ds = ds.ragged_batch(BATCH_SIZE)
        ds = ds.map(
            lambda x: augmenter(x, training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        ds = ds.ragged_batch(BATCH_SIZE)

    ds = ds.map(
        lambda x: {
            "images": x["images"],
            "bounding_boxes": keras_cv.bounding_box.to_dense(
                x["bounding_boxes"]
            )
        },
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.map(
        lambda x: (x["images"], x["bounding_boxes"]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.prefetch(tf.data.AUTOTUNE)