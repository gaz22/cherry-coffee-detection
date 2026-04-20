import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

RAW_DATA_DIR = "data/raw/processed_dataset.csv"

def load_data():
    df = pd.read_csv(RAW_DATA_DIR)
    print("\n=== BASIC INFO ===")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())
    return df


# class distribution
def check_class_balance(df):
    print("\n=== CLASS BALANCE ===")
    print(df["dataset_type"].value_counts())


# check taxon
def check_taxon_mapping(df):
    print("\n=== TAXON CHECK ===")

    for cls in df["dataset_type"].unique():
        print(f"\n{cls}:")
        print(df[df["dataset_type"] == cls]["taxon_id"].unique())


# check missing values
def check_missing(df):
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())


# check duplicate images
def check_duplicates(df):
    print("\n=== DUPLICATES ===")
    print("Duplicate image paths:", df["image_path"].duplicated().sum())


# check broken images
def check_broken_images(df):
    print("\n=== IMAGE INTEGRITY ===")

    broken = []

    for path in df["image_path"]:
        try:
            Image.open(path).verify()
        except Exception:
            broken.append(path)

    print("Broken images:", len(broken))

    return broken


# random visual check
def visualize_samples(df, n=9):
    print("\n=== VISUAL CHECK ===")

    sample = df.sample(n)

    plt.figure(figsize=(10, 10))

    for i, (_, row) in enumerate(sample.iterrows()):
        try:
            img = Image.open(row["image_path"])

            plt.subplot(3, 3, i + 1)
            plt.imshow(img)
            plt.title(row["dataset_type"])
            plt.axis("off")

        except Exception:
            print("Failed to load:", row["image_path"])

    plt.tight_layout()
    plt.show()


def main():
    df = load_data()

    check_class_balance(df)
    check_taxon_mapping(df)
    check_missing(df)
    check_duplicates(df)

    broken = check_broken_images(df)

    visualize_samples(df, n=9)

    print("\n=== SUMMARY ===")
    print("Total images:", len(df))
    print("Broken images:", len(broken))


if __name__ == "__main__":
    main()