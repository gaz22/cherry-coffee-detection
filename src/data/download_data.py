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
def download_images(df, limit=100):

    records = []

    for i, row in df.head(limit).iterrows():
        url = row.get("image_url")

        if pd.isna(url):
            continue

        # higher resolution
        url = url.replace("square", "large")

        # assign dataset role
        taxon = str(row.get("taxon_id"))

        # coffee
        if taxon == "64342":
            dataset_type = "positive"
        else: 
            # plant / background
            dataset_type = "negative"

        try:
            image_data = requests.get(url, timeout=10).content
            
            image_path = os.path.join(IMAGE_DIR, f"{i}.jpg")

            with open(image_path, "wb") as f:
                f.write(image_data)

            records.append({
                "image_path": image_path,
                "dataset_type": dataset_type,
                "taxon_id": taxon,
                "scientific_name": row.get("scientific_name")
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
    processed_df = download_images(df, limit=100)

    output_csv = os.path.join(RAW_DATA_DIR, "processed_dataset.csv")
    processed_df.to_csv(output_csv, index=False)

    print("Saved processed dataset:", output_csv)
    print("Completed")

    print("Completed")

if __name__ == "__main__":
    main()