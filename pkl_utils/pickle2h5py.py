import pandas as pd
from pathlib import Path
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import sys
import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric

base_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
save_folder = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")

features_path = '/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features'
# features_path = '/mnt/ML/Development/dolev.orgad/low_fps_Q_Features/v2_420x800_optical_flow_quarter_simd/burst_50fps/features'
hubert_soft_path = '/mnt/ML/Production/ML_Processed_Data/Audio_Features/hubert_soft/features/'
hubert_asr_path = '/mnt/ML/Production/ML_Processed_Data/Audio_Features/hubert_asr/features/'

pkls=['expressive_blendshapes_no_glasses_110326_test.pkl', 'expressive_blendshapes_no_glasses_110326_train_x5.pkl']
      #['LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl',
#'WHISPER_GIP_general_clean_250415_v2_with_side_attrs_valid_20250709_190842.pkl','LIP_GIP_general_clean_250415_v2_with_side_attrs.pkl'']

for pkl in pkls:
    p = base_path / pkl
    # p = base_path / "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes.pkl"

    print(f"processing {p.stem}")
    df = pd.read_pickle(p)
    sizes = df.frame_num.to_list()
    df.drop(columns=["frame_num"], inplace=True)
    df["tar_id"] = df["tar_id"].astype(str).to_list()
    df.reset_index(drop=True, inplace=True)

    with pd.HDFStore(save_folder / f'{p.stem}.h5', mode='w', swmr=True) as store:
        store.put(f'{p.stem}', df, format='table', complevel=9, complib='blosc')
        store.flush(fsync=True)  # Ensure all data is written to disk
    
    print("==========================================")
    
    with h5py.File(save_folder / f'{p.stem}.h5', 'a') as h5file:
        # Store the list of integers
        h5file.create_dataset('sizes', data=sizes)
        h5file.attrs['features_path']=features_path
        h5file.attrs['hubert_soft_path']=hubert_soft_path
        h5file.attrs['hubert_asr_path']=hubert_asr_path
        
        print(h5file.keys())
        print(h5file.attrs['features_path'])

    print("==========================================")
    print(f"saved  {p.stem}")