import os
import zipfile
import pandas as pd
import requests

from src.utils.config import COFFEE_ZIP, OUTPUT_DIR, PLANT_ZIP, RAW_DATA_DIR

OUTPUT_DIR = os.path.join(RAW_DATA_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    for i, row in df.head(limit).iterrows():
        url = row.get("image_url")

        if pd.isna(url):
            continue

        # higher resolution
        url = url.replace("square", "large")

        try:
            image_data = requests.get(url, timeout=10).content
            with open(f"{OUTPUT_DIR}/{i}.jpg", "wb") as f:
                f.write(image_data)
        
        except Exception as e:
            print(f"Failed: {url} - {e}")

def main():
    print("Extracting dataset...")
    extract_zip(COFFEE_ZIP)
    extract_zip(PLANT_ZIP)

    print("Loading CSVs...")
    df = load_csv_from_dir()
    print(f"Total records: {len(df)}")

    print("Download images...")
    download_images(df, limit=100)

    print("Completed")

if __name__ == "__main__":
    main()