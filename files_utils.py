import numpy as np


import shutil
from pathlib import Path


def safe_save(data: np.ndarray, save_path: Path, save_size: str = None):
    if data.size == 0:
        print(f"[WARNING] Empty array, nothing saved to {save_path}")
        return
    save_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_save_path = save_path.with_suffix(".tmp.npy")
    if save_size is not None:
        if save_size == "float16":
            data = data.astype(np.float16)
        elif save_size == "float32":
            data = data.astype(np.float32)
        elif save_size == "float64":
            data = data.astype(np.float64)
        elif save_size == "uint8":
            data = data.astype(np.uint8)
    np.save(tmp_save_path, data)
    shutil.move(tmp_save_path, save_path.with_suffix(".npy"))
