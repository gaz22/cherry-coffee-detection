import os

# Paths
DATA_DIR = "data/"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
IMAGE_DIR = os.path.join(PROCESSED_DIR, "images")

# Dataset files
ARABICA_ZIP = os.path.join(RAW_DATA_DIR, "observations-712548.csv.zip")
CANEPHORA_ZIP = os.path.join(RAW_DATA_DIR, "observations-713339.csv.zip")
PLANT_ZIP = os.path.join(RAW_DATA_DIR, "observations-709374.csv.zip")

# Classes
CLASSES = ["arabica", "canephora"]

# Training params
BATCH_SIZE = 8
EPOCHS = 20
IMAGE_SIZE = 416