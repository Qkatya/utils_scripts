from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
import argparse


def load_dataframe(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_dataframe(df, path):
    with open(path, 'wb') as f:
        pickle.dump(df, f)

def transfer_attrs(src_path, dst_path, save_path=None):
    # Load both DataFrames
    df_src = load_dataframe(src_path)
    df_dst = load_dataframe(dst_path)

    # Check and transfer missing attrs
    for key, value in df_src.attrs.items():
        if key not in df_dst.attrs:
            df_dst.attrs[key] = value
            print(f"Added attribute '{key}': {value}")
        else:
            print(f"Attribute '{key}' already exists in destination.")

    # Save the updated df_dst if requested
    if save_path:
        save_dataframe(df_dst, save_path)
        print(f"Updated DataFrame saved to: {save_path}")
    else:
        print("Dry run: nothing saved.")



if __name__ == "__main__":
    path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
    src_filename = "LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl"
    lst = [
        "LIP_GIP_general_clean_250415_v2_with_side",
        "SILENT_GIP_general_clean_250415_v2_with_side",
        "WHISPER_GIP_general_clean_250415_v2_with_side"]
    src = path / src_filename
    for dst_filename in tqdm(lst):
        dst = path / dst_filename
        dst2 = dst.with_name(dst.stem + '_attrs.pkl')
        transfer_attrs(src, dst, dst2)
a=1