"""
Central project configuration.

This file keeps the main paths and experiment parameters in one place so the
project can run on different machines without hard-coding a local Windows path.
"""

from pathlib import Path

# Project root = parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Main folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"

# Image resizing size
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

# Updated after inspecting the dataset
CLASS_NAMES = []

# Default KNN values
DEFAULT_K_VALUES = [1, 3, 5, 7, 9, 11, 15]

# Reproducibility
RANDOM_SEED = 42


def make_all_dirs():
    """Create all required project folders if they do not exist."""
    dir_list = [
        DATA_DIR,
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        PROCESSED_IMAGES_DIR,
        FEATURES_DIR,
        SPLITS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        EXPERIMENTS_DIR,
        EXPERIMENT_RESULTS_DIR,
    ]
    for temp_dir in dir_list:
        temp_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("Setting up project folders...")
    make_all_dirs()
    print("Done.")
