import os
import sys
import pandas as pd
from collections import Counter
from PIL import Image

# paths
RAW_CSV = "data/raw/processed_dataset.csv"

# yolo
YOLO_DIR = "data/yolo"
TRAIN_IMAGE = os.path.join(YOLO_DIR, "images/train")
VAL_IMAGE = os.path.join(YOLO_DIR, "images/val")
TRAIN_LABEL = os.path.join(YOLO_DIR, "labels/train")
VAL_LABEL = os.path.join(YOLO_DIR, "labels/val")


# util
def get_files(folder, ext=None):
    if not os.path.exists(folder):
        return []

    files = os.listdir(folder)
    if ext:
        files = [f for f in files if f.endswith(ext)]
    return sorted(files)


# raw dataset check
def check_raw_dataset():
    print("\n=== RAW DATASET CHECK ===")

    df = pd.read_csv(RAW_CSV)

    print("\n=== BASIC INFO ===")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("\n=== CLASS BALANCE ===")
    if "dataset_type" in df.columns:
        print(df["dataset_type"].value_counts())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATES ===")
    print("Duplicate images:", df["image_path"].duplicated().sum())

    print("\n=== SAMPLE DATA ===")
    print(df.head(5))


# YOLO dataset check
def check_yolo_dataset():
    print("\n=== YOLO DATASET CHECK ===")

    def check_overlap():
        train_files = set(get_files(TRAIN_IMAGE))
        val_files = set(get_files(VAL_IMAGE))

        overlap = train_files.intersection(val_files)

        print("\n=== SPLIT CHECK ===")
        print("Train images:", len(train_files))
        print("Val images:", len(val_files))
        print("Overlap:", len(overlap))

    def check_missing_labels():
        print("\n=== LABEL CHECK ===")

        missing_train = []
        missing_val = []

        for img in get_files(TRAIN_IMAGE):
            label = img.replace(".jpg", ".txt")
            if not os.path.exists(os.path.join(TRAIN_LABEL, label)):
                missing_train.append(img)

        for img in get_files(VAL_IMAGE):
            label = img.replace(".jpg", ".txt")
            if not os.path.exists(os.path.join(VAL_LABEL, label)):
                missing_val.append(img)

        print("Missing train labels:", len(missing_train))
        print("Missing val labels:", len(missing_val))

    def check_label_format(folder):
        print(f"\n=== LABEL FORMAT CHECK ({folder}) ===")

        labels = get_files(folder, ".txt")

        invalid = 0
        class_counter = Counter()

        for label_file in labels:
            path = os.path.join(folder, label_file)

            with open(path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()

                if len(parts) == 0:
                    continue

                if len(parts) != 5:
                    invalid += 1
                    continue

                class_counter[parts[0]] += 1

        print("Invalid label lines:", invalid)
        print("Class distribution:", dict(class_counter))

    def check_corrupted_images():
        print("\n=== IMAGE INTEGRITY CHECK ===")

        broken = []

        for folder in [TRAIN_IMAGE, VAL_IMAGE]:
            for img_file in os.listdir(folder):
                path = os.path.join(folder, img_file)

                try:
                    Image.open(path).verify()
                except Exception:
                    broken.append(path)

        print("Broken images:", len(broken))

    # all checks
    check_overlap()
    check_missing_labels()
    check_label_format(TRAIN_LABEL)
    check_label_format(VAL_LABEL)
    check_corrupted_images()



def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("python -m src.debug.check_dataset [raw|yolo]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "raw":
        check_raw_dataset()

    elif mode == "yolo":
        check_yolo_dataset()

    else:
        print("Invalid mode. Use: raw | yolo")
        sys.exit(1)


if __name__ == "__main__":
    main()