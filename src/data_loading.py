# File: C:\Users\Arafat\Desktop\IMA Project\src\data_loading.py

"""
In this file I want to load the dataset info.
Basically I want to scan the image folders, read the labels file,
and create a pandas DataFrame with filename, label, and split.
"""

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .utils import set_random_seed


def scan_image_folder_to_df(image_root_dir: str) -> pd.DataFrame:
    """
    In this function I want to look through the dataset folders and
    create a DataFrame that has one row per image.
    Here I assume that each class is a subfolder with images inside,
    like 'raw/blood_cells/RBC/image1.png' etc.

    If my dataset is different (like labels in a CSV), we can adapt this later.
    """
    file_list = []
    label_list = []

    for temp_root, temp_dirs, temp_files in os.walk(image_root_dir):
        for temp_file in temp_files:
            if temp_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                full_path = os.path.join(temp_root, temp_file)
                # class label = name of the immediate parent folder
                label_name = os.path.basename(os.path.dirname(full_path))
                file_list.append(full_path)
                label_list.append(label_name)

    df = pd.DataFrame({"filepath": file_list, "label": label_list})
    print(f"[data_loading] Found {len(df)} images in {image_root_dir}")
    print("[data_loading] Label value counts:")
    print(df["label"].value_counts())
    return df


def create_train_val_test_split(
    df_all: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    In this function I want to take the full DataFrame with all images,
    and split it into train, validation, and test sets in a stratified way.
    I will add a new column called 'split' with values 'train', 'val', 'test'.
    """
    set_random_seed(random_seed)

    df_temp = df_all.copy()

    # First, split off the test set
    df_train_val, df_test = train_test_split(
        df_temp,
        test_size=test_size,
        random_state=random_seed,
        stratify=df_temp["label"],
    )

    # Now split train and val from df_train_val
    val_ratio = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df_train_val["label"],
    )

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    df_final = pd.concat([df_train, df_val, df_test], ignore_index=True)
    print("[data_loading] Split sizes:")
    print(df_final["split"].value_counts())
    return df_final


def save_splits_to_csv(df_splits: pd.DataFrame, csv_name: str = "split_info.csv") -> str:
    """
    In this function I want to save the DataFrame with the split info
    into the 'processed/splits' folder so I can reuse it later.
    """
    if not os.path.exists(config.SPLITS_DIR):
        os.makedirs(config.SPLITS_DIR, exist_ok=True)

    csv_path = os.path.join(config.SPLITS_DIR, csv_name)
    df_splits.to_csv(csv_path, index=False)
    print(f"[data_loading] Split info saved to {csv_path}")
    return csv_path


def load_splits_csv(csv_path: str = None) -> pd.DataFrame:
    """
    In this function I want to load the split CSV into a DataFrame.
    If no path given, I will use the default 'processed/splits/split_info.csv'.
    """
    if csv_path is None:
        csv_path = os.path.join(config.SPLITS_DIR, "split_info.csv")

    print(f"[data_loading] Loading split CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    print("[data_loading] Loaded DataFrame shape:", df.shape)
    return df


def get_split_dataframes(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    In this function I want to take the big DataFrame with 'split' column
    and just return three smaller DataFrames: train, val, and test.
    """
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    print("[data_loading] Train size:", len(df_train))
    print("[data_loading] Val size:", len(df_val))
    print("[data_loading] Test size:", len(df_test))

    return df_train, df_val, df_test


if __name__ == "__main__":
    # Example quick test I need to update RAW_DATA_DIR to actual image root
    from . import config as cfg

    cfg.make_all_dirs()
    temp_df = scan_image_folder_to_df(cfg.RAW_DATA_DIR)
    temp_df = create_train_val_test_split(temp_df)
    save_splits_to_csv(temp_df)
