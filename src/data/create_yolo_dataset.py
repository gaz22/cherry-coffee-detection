import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import RAW_DATA_DIR

YOLO_DIR = "data/yolo"
IMAGE_DIR = os.path.join(YOLO_DIR, "images")
LABEL_DIR = os.path.join(YOLO_DIR, "labels")

TRAIN_RATION = 0.8

# class mapping
CLASS_MAP = {
    "arabica": 0,
    "canephora":1
}

# create yolo train/val folder structure
def create_dir():
    if os.path.exists(YOLO_DIR):
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val"]:
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

    # train / val split
    train_df, val_df = train_test_split(
        df,
        test_size = 1 - TRAIN_RATION,
        random_state = 42,
        shuffle = True,
        stratify=df["dataset_type"]
    )

    create_dir()

    # process splits
    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        print(f"\nProcessing {split_name}...")

        for _, row in split_df.iterrows():

            src_image = row["image_path"]
            filename = os.path.basename(src_image)

            dst_image = os.path.join(IMAGE_DIR, split_name, filename)
            dst_label = os.path.join(
                LABEL_DIR,
                split_name,
                filename.replace(".jpg", ".txt")
            )

            # copy image
            shutil.copy(src_image, dst_image)

            # assign class ID
            dataset_type = row["dataset_type"]

            class_id = CLASS_MAP.get(dataset_type, None)

            # create label file
            create_label_file(dst_label, class_id)
    
    print("\nYOLO dataset created successfully!")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}") 

if __name__ == "__main__":
    main()