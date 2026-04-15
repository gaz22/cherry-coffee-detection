import os

# Paths
DATA_DIR = "data/"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = "outputs/"

IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")

# Dataset files
COFFEE_ZIP = os.path.join(RAW_DATA_DIR, "observations-709370.csv.zip")
PLANT_ZIP = os.path.join(RAW_DATA_DIR, "observations-709374.csv.zip")

# Role label
POSITIVE_DATASET = "coffee"
NEGATIVE_DATASET = "plant_background"

# Class
CLASSES = ["coffee_cherry"]

# Training params
BATCH_SIZE = 8
EPOCHS = 20
IMAGE_SIZE = 416