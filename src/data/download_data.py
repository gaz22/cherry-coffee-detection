import os
import shutil
import zipfile
import pandas as pd
import requests

from src.utils.config import (
    ARABICA_ZIP,
    CANEPHORA_ZIP,
    PLANT_ZIP,
    RAW_DATA_DIR,
    IMAGE_DIR
)
# clean image folder
if os.path.exists(IMAGE_DIR):
    shutil.rmtree(IMAGE_DIR)
os.makedirs(IMAGE_DIR, exist_ok=True)

# extract zip file -> raw data directory
def extract_zip(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

# load csv
def load_csv(file_name):
    path = os.path.join(RAW_DATA_DIR, file_name)
    return pd.read_csv(path)

# download images
def download_images(arabica_df, canephora_df, plant_df,
                    target_per_class=300, negative_ratio=0.1):

    print("Arabica raw:", len(arabica_df))
    print("Canephora raw:", len(canephora_df))
    print("Plant raw:", len(plant_df))

    # balancing target class
    arabica_sample = arabica_df.sample(
        n=target_per_class,
        replace=len(arabica_df) < target_per_class,
        random_state=42
    )
    canephora_sample = canephora_df.sample(
        n=target_per_class,
        replace=True,
        random_state=42
    )

    # negative sampling
    neg_size = int(negative_ratio * (2 * target_per_class))

    plant_sample = plant_df.sample(
        n=min(len(plant_df), neg_size),
        random_state=42
    )

    arabica_sample["dataset_type"] = "arabica"
    canephora_sample["dataset_type"] = "canephora"
    plant_sample["dataset_type"] = "negative"
    
    # combine dataset
    combined = pd.concat(
        [arabica_sample, canephora_sample, plant_sample],
        ignore_index=True
    )

    print("Final dataset:", len(combined))

    # download images
    records = []

    for i, row in combined.iterrows():
        url = row.get("image_url")

        if pd.isna(url):
            continue

        # higher resolution
        url = url.replace("square", "large")

        try:
            image_data = requests.get(url, timeout=10).content
            
            image_path = os.path.join(IMAGE_DIR, f"{i}.jpg")

            with open(image_path, "wb") as f:
                f.write(image_data)

            # store metadata for training pipeline
            records.append({
                "image_path": image_path,
                "dataset_type": row["dataset_type"],
                "taxon_id": row.get("taxon_id"),
                "scientific_name": row.get("scientific_name"),
                "species_guess": row.get("species_guess"),
                "common_name": row.get("common_name")
            })
        
        except Exception as e:
            print(f"Failed: {url} - {e}")

    return pd.DataFrame(records)

def main():
    print("Extracting dataset...")
    extract_zip(ARABICA_ZIP)
    extract_zip(CANEPHORA_ZIP)
    extract_zip(PLANT_ZIP)

    # load data
    print("Loading CSVs...")
    arabica_df = load_csv("observations-712548.csv")
    canephora_df = load_csv("observations-713339.csv")
    plant_df = load_csv("observations-709374.csv")

    print("Download images...")
    processed_df = download_images(
        arabica_df,
        canephora_df,
        plant_df,
        target_per_class=300
    )

    output_csv = os.path.join(RAW_DATA_DIR, "processed_dataset.csv")
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    processed_df.to_csv(output_csv, index=False)
    print("Saving NEW dataset ONLY:", len(processed_df))
    print("Saved processed dataset:", output_csv)
    print("Total saved:", len(processed_df))

if __name__ == "__main__":
    main()