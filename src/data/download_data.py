import os
import zipfile
import pandas as pd
import requests

from src.utils.config import (
    COFFEE_ZIP, PLANT_ZIP,
    RAW_DATA_DIR, IMAGE_DIR
)

os.makedirs(IMAGE_DIR, exist_ok=True)

# extract zip file -> raw data directory
def extract_zip(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_DIR)

# load all CSVs from raw directory and combine to one DF
def load_csv_from_dir():
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    dataframes = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file))
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

# download images
def download_images(df, limit=500):
    df = df.copy()

    # combine metadata fields -> one text field
    # improve labeling robustness
    df["text_blob"] = (
        df["scientific_name"].fillna("") + " " +
        df["species_guess"].fillna("") + " " +
        df["common_name"].fillna("")
    ).str.lower()

    # separate coffee non-coffee using semantic matching
    coffee_df = df[df["text_blob"].str.contains("coffea|coffee", na=False)]
    plant_df = df[~df["text_blob"].str.contains("coffea|coffee", na=False)]

    print("Coffee rows:", len(coffee_df))
    print("Plant rows:", len(plant_df))

    # balancing dataset
    coffee_sample = coffee_df.sample(n=min(len(coffee_df), limit // 2), random_state=42)
    plant_sample = plant_df.sample(n=min(len(plant_df), limit // 2), random_state=42)

    combined = pd.concat([coffee_sample, plant_sample], ignore_index=True)

    print("Final combined:", len(combined))
    print(combined["text_blob"].head(10))
    
    records = []

    for i, row in combined.iterrows():
        url = row.get("image_url")

        if pd.isna(url):
            continue

        # higher resolution
        url = url.replace("square", "large")

        # assign label using semantic text detection
        is_positive = "coffea" in str(row["text_blob"])
        dataset_type = "positive" if is_positive else "negative"
        
        taxon_id = row.get("taxon_id")

        try:
            image_data = requests.get(url, timeout=10).content
            
            image_path = os.path.join(IMAGE_DIR, f"{i}.jpg")

            with open(image_path, "wb") as f:
                f.write(image_data)

            # store metadata for training pipeline
            records.append({
                "image_path": image_path,
                "dataset_type": dataset_type,
                "taxon_id": taxon_id,
                "scientific_name": row.get("scientific_name"),
                "species_guess": row.get("species_guess"),
                "common_name": row.get("common_name")
            })
        
        except Exception as e:
            print(f"Failed: {url} - {e}")

    return pd.DataFrame(records)

def main():
    print("Extracting dataset...")
    extract_zip(COFFEE_ZIP)
    extract_zip(PLANT_ZIP)

    # load data
    print("Loading CSVs...")
    df = load_csv_from_dir()
    print(f"Total records: {len(df)}")

    print("Download images...")
    processed_df = download_images(df, limit=500)

    output_csv = os.path.join(RAW_DATA_DIR, "processed_dataset.csv")
    
    if os.path.exists(output_csv):
        os.remove(output_csv)
    
    processed_df.to_csv(output_csv, index=False)
    print("Saving NEW dataset ONLY:", len(processed_df))

    print("Saved processed dataset:", output_csv)
    print("Completed")

    print("Completed")

if __name__ == "__main__":
    main()