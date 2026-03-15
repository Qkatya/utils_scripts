import numpy as np
from pathlib import Path

blendshape_base_path = Path('/mnt/ML/Development/katya.ivantsiv/blendshapes/2025/02/10/AbbeyAntenna-200139/101_0_4a8afc71-ecb9-465e-9a56-737253b18a8d_loud')


npy_path = blendshape_base_path / "blendshape_coeffs.npy"
npz_path = blendshape_base_path / "landmarks_and_blendshapes.npz"

# blendshape_label = np.load(blendshape_base_path / "blendshape_coeffs.npy") # in the shape of (camera_frames (in 30 fps), 52)

blendshape_label = None

if npy_path.exists():
    blendshape_label = np.load(npy_path)  # shape: (frames, 52)
elif npz_path.exists():
    blendshapes_and_landmarks = np.load(npz_path)
    blendshape_label = blendshapes_and_landmarks['blendshapes']  # shape: (frames, 52)
else:
    raise FileNotFoundError("Neither blendshape_coeffs.npy nor landmarks_and_blendshapes.npz found.")

blendshapes_idx=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

blendshape_label = blendshape_label[:, blendshapes_idx] #take only the shapes we want & can
mask = np.ones(blendshape_label.shape[0], dtype=bool)
mask[5::6] = False  # TODO: Interpolate to 25/50 Hz?

blendshape_label = blendshape_label[mask] #take out every 6th frame to turn it into 25 Hz from the 30 FPS it is currently in. TODO?: Can be done more precise if needed with interpolation if needed.
out_of_bounds_or_nan = np.any((blendshape_label < 0) | (blendshape_label > 1) | np.isnan(blendshape_label)) 
assert out_of_bounds_or_nan == False, f'There seems to be blendshape values which are out of bounds, {self.blendshape_base_path / row.run_path / "blendshape_coeffs.npy"}' # replace it in a future version - TODO: handle the err instead of raise

if blendshapes_normalize_path != '':
    # blendshape_label = (blendshape_label - self.blendshape_normalization_factors["mean"].values) / self.blendshape_normalization_factors["std"].values

    blendshape_label = (blendshape_label - self.blendshape_normalization_factors["Mean"].values[np.array(self.blendshapes_idx)]) / self.blendshape_normalization_factors["Std"].values[np.array(self.blendshapes_idx)]
    blendshape_label = np.clip(blendshape_label, -3, 3)
    # blendshape_label = np.clip(blendshape_label, -1, 1)


if blendshapes_diff:
    blendshape_label = np.diff(blendshape_label, axis=0, prepend=blendshape_label[0][None,:])
    blendshape_label[0] = blendshape_label[1]