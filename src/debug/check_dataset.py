import os
import pandas as pd

RAW_DATA_DIR = "data/raw/"

def main():
    # load csv
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]

    print("CSV count:", len(csv_files))

    dfs = []
    for f in csv_files:
        path = os.path.join(RAW_DATA_DIR, f)
        df = pd.read_csv("data/raw/processed_dataset.csv")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # health check
    print("\n=== DATASET INFO ===")
    print("Total rows:", len(df))
    print("Columns:", df.columns.tolist())

    print("\n=== COFFEE COUNT ===")
    print(df["scientific_name"].str.contains("coffea", case=False, na=False).sum())
    
    # check key columns
    print("\n=== COLUMN CHECK ===")
    required = ["scientific_name", "species_guess", "common_name"]

    for col in required:
        print(f"{col}:", col in df.columns)

    # signal check 
    print("\n=== COFFEE SIGNAL CHECK ===")

    if "scientific_name" in df.columns:
        n = df["scientific_name"].str.contains("coffea", case=False, na=False).sum()
        print("scientific_name (coffea):", n)

    if "species_guess" in df.columns:
        n = df["species_guess"].str.contains("coffee", case=False, na=False).sum()
        print("species_guess (coffee):", n)

    if "common_name" in df.columns:
        n = df["common_name"].str.contains("coffee", case=False, na=False).sum()
        print("common_name (coffee):", n)

    # sanity sample (only 3 rows)
    print("\n=== SAMPLE ROWS ===")
    print(df[["scientific_name"]].dropna().head(3))


if __name__ == "__main__":
    main()