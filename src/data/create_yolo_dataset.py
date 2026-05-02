import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import RAW_DATA_DIR

YOLO_DIR = "data/yolo"
IMAGE_DIR = os.path.join(YOLO_DIR, "images")
LABEL_DIR = os.path.join(YOLO_DIR, "labels")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# class mapping
CLASS_MAP = {
    "arabica": 0,
    "canephora":1
}

# create yolo train/val folder structure
def create_dir():
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(IMAGE_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABEL_DIR, split), exist_ok=True)
def create_label_file(label_path, class_id=None):
    with open(label_path, "w") as f:
        if class_id is not None:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            # else: negative sample

def main():
    csv_path = os.path.join(RAW_DATA_DIR,  "processed_dataset.csv")
    df = pd.read_csv(csv_path)

    print("Total dataset:", len(df))
    print(df["dataset_type"].value_counts())

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state = 42,
        shuffle = True,
        stratify=df["dataset_type"]
    )

    # Second split
    # val vs test (equal halves of the remaining)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=42,
        shuffle=True,
        stratify=temp_df["dataset_type"]
    )

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"\nTrain class distribution:")
    print(train_df["dataset_type"].value_counts())
    print(f"\nVal class distribution:")
    print(val_df["dataset_type"].value_counts())
    print(f"\nTest class distribution:")
    print(test_df["dataset_type"].value_counts())

    create_dir()

    # process splits
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\nProcessing {split_name}...")

        for _, row in split_df.iterrows():
            src_image = row["image_path"]
            if not os.path.exists(src_image):
                continue
            filename = os.path.basename(src_image)
            dst_image = os.path.join(IMAGE_DIR, split_name, filename)
            dst_label = os.path.join(
                LABEL_DIR,
                split_name,
                filename.replace(".jpg", ".txt")
            )

            shutil.copy(src_image, dst_image)

            class_id = CLASS_MAP.get(row["dataset_type"], None)
            create_label_file(dst_label, class_id)

    # Verify no overlap
    train_files = set(os.listdir(os.path.join(IMAGE_DIR, "train")))
    val_files   = set(os.listdir(os.path.join(IMAGE_DIR, "val")))
    test_files  = set(os.listdir(os.path.join(IMAGE_DIR, "test")))

    train_val_overlap  = train_files & val_files
    train_test_overlap = train_files & test_files
    val_test_overlap   = val_files   & test_files

    print(f"\n=== Data Leakage Check ===")
    print(f"Train/Val overlap:  {len(train_val_overlap)} " if not train_val_overlap else f"Train/Val overlap:  {len(train_val_overlap)} ❌")
    print(f"Train/Test overlap: {len(train_test_overlap)} " if not train_test_overlap else f"Train/Test overlap: {len(train_test_overlap)} ❌")
    print(f"Val/Test overlap:   {len(val_test_overlap)}  " if not val_test_overlap else f"Val/Test overlap:   {len(val_test_overlap)} ❌")

    print("\n YOLO dataset created successfully!")


if __name__ == "__main__":
    main()