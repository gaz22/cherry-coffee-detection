# Paths
DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
OUTPUT_DIR = "outputs/"

# Dataset files
COFFEE_ZIP = RAW_DATA_DIR + "observations-709370.csv.zip"
PLANT_ZIP = RAW_DATA_DIR + "observations-709374.csv.zip"

# Classes
CLASSES = ["ripe", "unripe"]

# Training params
BATCH_SIZE = 8
EPOCHS = 20
IMAGE_SIZE = 416