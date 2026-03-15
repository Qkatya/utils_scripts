import h5py
import pandas as pd

def read_run_paths(file_path):
    with h5py.File(file_path, 'r') as f:
        def find_run_path_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype.names and 'run_path' in obj.dtype.names:
                    print(f"[{name}] run_path values:")
                    print(obj['run_path'][:])
                elif 'run_path' in name:
                    print(f"[{name}] values:")
                    print(obj[:])

        f.visititems(find_run_path_datasets)

if __name__ == "__main__":
       
    file_path = "/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes.h5"
    
    # See what keys (i.e., datasets or tables) are in the file
    with pd.HDFStore(file_path, 'r') as store:
        print("Available keys (datasets):", store.keys())

    # Load one of the datasets
    df = pd.read_hdf(file_path, key='/6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes')  # Replace with the correct key
    run_paths = df["run_path"].tolist()

    target = "2025/03/27/LongGrating-122452/143_0_e0fa21cd-1432-4345-bd17-0abc288dfa69_loud"

    if target in run_paths:
        print("Found!")
    else:
        print("Not found.")
        
        
        
    print(df.head())

    # Optionally, view all run_path values if it exists
    if 'run_path' in df.columns:
        print(df['run_path'].unique())
    
    read_run_paths(path)
