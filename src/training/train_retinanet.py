import os
import tensorflow as tf
import keras_cv

from src.data.dataset_retinanet import load_coco, build_dataset
from src.utils.config import BATCH_SIZE

IMG_SIZE = 416
NUM_CLASSES = 2
OUTPUT_PATH = "outputs/retinanet.weights.h5"

SMOKE = False


def focal_loss():
    return keras_cv.losses.FocalLoss(
        # upweights arabica (class 0)
        alpha=0.75,
        gamma=2.0,
        from_logits=True,
        reduction="sum"
    )


def build_model():
    backbone = keras_cv.models.ResNet50Backbone.from_preset("resnet50_imagenet")
    backbone.trainable = False

    model = keras_cv.models.RetinaNet(
        backbone=backbone,
        num_classes=NUM_CLASSES,
        bounding_box_format="xyxy",
        images_input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model.compile(
        box_loss="smoothl1",
        classification_loss=focal_loss(),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=3e-4,
                # steps_per_epoch * epochs
                decay_steps=59 * 10   
            ),
            global_clipnorm=1.0
        )
    )

    return model, backbone


def main():
    print("Training RetinaNet...")

    train_data, val_data = load_coco(smoke=SMOKE)

    train_ds = build_dataset(train_data, training=True)
    val_ds   = build_dataset(val_data,   training=False)

    steps_per_epoch = len(train_data) // BATCH_SIZE
    val_steps       = len(val_data)   // BATCH_SIZE

    print(f"Steps/epoch: {steps_per_epoch} | Val steps: {val_steps}")

    model, backbone = build_model()

    os.makedirs("outputs", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "outputs/retinanet/retinanet_best.weights.h5",
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 1 - frozen backbone
    print("\n=== Phase 1: Frozen backbone ===")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    # 2 - fine tune top layers
    print("\n=== Phase 2: Fine-tuning ===")

    # partial unfreeze — last 20 layers only
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    for layer in backbone.layers[-20:]:
        layer.trainable = True

    model.compile(
        box_loss="smoothl1",
        classification_loss=focal_loss(),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-4,
                decay_steps=steps_per_epoch * 10
            ),
            global_clipnorm=1.0
        )
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    model.save_weights(OUTPUT_PATH)
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()