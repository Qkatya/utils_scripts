"""
Script to process split files:
1. Add 'side' column from the database
2. Convert to h5 format with metadata
"""

import pandas as pd
import psycopg2
from tqdm import tqdm
from pathlib import Path
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import sys
import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric
from datetime import datetime


# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Input configuration
base_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')
files = [
    "blendshapes_KatyaIvantsiv_2026_01_11.h5"
]
# base_dir = Path('/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250708_split_1')
# files = [
#     "train_kfold_all_24p1M",
#     "train_kfold_19p8M",
# ]

# Output configuration
output_dir = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits')

# Feature paths for h5 metadata
features_path = '/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features'
hubert_soft_path = '/mnt/ML/Production/ML_Processed_Data/Audio_Features/hubert_soft/features/'
hubert_asr_path = '/mnt/ML/Production/ML_Processed_Data/Audio_Features/hubert_asr/features/'

# Options
output_prefix = ""  # Prefix to add to output filenames (e.g., "V3_" -> "V3_train_with_side_20251216.h5")
save_pkl = True  # Set to True to also save intermediate pkl with side column
batch_size = 100000  # Batch size for DB queries

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_df_with_side(df_with_tar_id, batch_size=100000):
    """Add 'side' column to dataframe by querying the database."""
    assert isinstance(df_with_tar_id.tar_id.iloc[0], str), "tar_id must be string"
    
    # Get unique tar_ids to query (handles both unique and non-unique cases)
    unique_tar_ids = df_with_tar_id.tar_id.unique()
    print(f"  Total rows: {len(df_with_tar_id):,}, Unique tar_ids: {len(unique_tar_ids):,}")
    
    # DB setup using psycopg2 directly
    conn = psycopg2.connect(
        host="q-data-db-replica.q.ai",
        port=5432,
        database="q-data-db-prod",
        user="bits_viewer",
        password="qviewer1"
    )

    results = []
    
    try:
        for i in tqdm(range(0, len(unique_tar_ids), batch_size), desc="Fetching side from DB"):
            batch_ids = unique_tar_ids[i:i + batch_size]
            placeholders = ",".join([f"'{x}'" for x in batch_ids])
            query = """
            select id as tar_id, side
            from q_frames_files
            WHERE id IN ({})
            """.format(placeholders)
            
            df_batch = pd.read_sql(query, conn)
            df_batch["tar_id"] = df_batch["tar_id"].astype(str)
            results.append(df_batch)
    finally:
        conn.close()
    
    # Combine all batches
    df_all_results = pd.concat(results, ignore_index=True)
    
    print(f"  DB returned {len(df_all_results):,} rows with side info")
    print(f"  Sample tar_ids from input: {df_with_tar_id.tar_id.iloc[:3].tolist()}")
    print(f"  Sample tar_ids from DB: {df_all_results.tar_id.iloc[:3].tolist() if len(df_all_results) > 0 else 'EMPTY'}")
    
    # Merge back into original df
    df_with_tar_id = df_with_tar_id.merge(df_all_results, how='left', on='tar_id')
    
    print(f"  Columns after merge: {df_with_tar_id.columns.tolist()}")

    return df_with_tar_id


def convert_to_h5(df, output_path, features_path, hubert_soft_path, hubert_asr_path):
    """Convert dataframe to h5 format with metadata."""
    sizes = df.frame_num.to_list()
    df = df.drop(columns=["frame_num"])
    df["tar_id"] = df["tar_id"].astype(str).to_list()
    
    # Convert date columns to string for HDF5 serialization
    if "recording_date" in df.columns:
        df["recording_date"] = df["recording_date"].astype(str)
    
    df.reset_index(drop=True, inplace=True)
    
    # Get the key name from the output path stem
    key_name = output_path.stem
    
    with pd.HDFStore(output_path, mode='w', swmr=True) as store:
        store.put(key_name, df, format='table', complevel=9, complib='blosc')
        store.flush(fsync=True)
    
    with h5py.File(output_path, 'a') as h5file:
        h5file.create_dataset('sizes', data=sizes)
        h5file.attrs['features_path'] = features_path
        h5file.attrs['hubert_soft_path'] = hubert_soft_path
        h5file.attrs['hubert_asr_path'] = hubert_asr_path
        
        print(f"  Keys: {list(h5file.keys())}")
        print(f"  features_path: {h5file.attrs['features_path']}")


def load_dataframe(input_path: Path) -> pd.DataFrame:
    """Load dataframe from either .pkl or .h5 file."""
    suffix = input_path.suffix.lower()
    
    if suffix == '.pkl':
        print("Loading pickle file...")
        df = pd.read_pickle(input_path)
    elif suffix == '.h5':
        print("Loading h5 file...")
        # Try to read using HDFStore first
        with pd.HDFStore(input_path, mode='r') as store:
            keys = store.keys()
            if keys:
                # Use the first key found (usually matches the filename stem)
                key = keys[0]
                print(f"  Using key: {key}")
                df = store[key]
            else:
                raise ValueError(f"No dataframe keys found in h5 file: {input_path}")
        
        # Check if 'sizes' dataset exists and add frame_num column if needed
        with h5py.File(input_path, 'r') as h5file:
            if 'sizes' in h5file.keys() and 'frame_num' not in df.columns:
                sizes = h5file['sizes'][:]
                df['frame_num'] = sizes
                print(f"  Added frame_num column from 'sizes' dataset")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .pkl or .h5")
    
    return df


def process_split_file(input_path: Path, output_dir: Path):
    """
    Process a single split file:
    1. Load the pickle or h5 file
    2. Add 'side' from database
    3. Optionally save the pkl with side
    4. Convert to h5 format
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")
    
    # Load dataframe (supports both pkl and h5)
    df = load_dataframe(input_path)
    print(f"Loaded {len(df):,} rows")
    
    # Ensure tar_id is string
    df["tar_id"] = df["tar_id"].astype(str)
    
    # Add side from database (skip if already exists)
    if 'side' in df.columns:
        print(f"'side' column already exists. Skipping DB fetch.")
        print(f"  Side value counts: {df['side'].value_counts().to_dict()}")
        print(f"  Null count: {df['side'].isna().sum():,}")
        df_with_side = df
    else:
        print("Adding 'side' from database...")
        df_with_side = get_df_with_side(df, batch_size=batch_size)
    
    if 'side' in df_with_side.columns:
        print(f"Added side column. Null count: {df_with_side['side'].isna().sum():,}")
    else:
        print("WARNING: 'side' column was not added! Check tar_id matching.")
    
    # Generate output name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    base_name = input_path.stem
    output_name = f"{output_prefix}{base_name}_with_side_{timestamp}"
    
    # Optionally save pkl with side
    if save_pkl:
        pkl_output_path = output_dir / f"{output_name}.pkl"
        df_with_side.to_pickle(pkl_output_path)
        print(f"Saved pkl: {pkl_output_path}")
    
    # Convert to h5
    h5_output_path = output_dir / f"{output_name}.h5"
    print(f"Converting to h5: {h5_output_path}")
    convert_to_h5(
        df_with_side.copy(),
        h5_output_path,
        features_path,
        hubert_soft_path,
        hubert_asr_path
    )
    print(f"Saved h5: {h5_output_path}")
    
    return h5_output_path


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {files}")
    
    for file_name in files:
        # Check for the file with its original extension first
        input_path = base_dir / file_name
        
        # If file_name doesn't have an extension or file doesn't exist, try both .h5 and .pkl
        if not input_path.exists() or input_path.suffix.lower() not in ['.pkl', '.h5']:
            h5_path = (base_dir / file_name).with_suffix(".h5")
            pkl_path = (base_dir / file_name).with_suffix(".pkl")
            
            if h5_path.exists():
                input_path = h5_path
            elif pkl_path.exists():
                input_path = pkl_path
            else:
                print(f"Warning: File not found: {file_name} (tried .h5 and .pkl)")
                continue
        
        if not input_path.exists():
            print(f"Warning: File not found: {input_path}")
            continue
            
        process_split_file(input_path, output_dir)
    
    print(f"\n{'='*60}")
    print("All files processed successfully!")
    print(f"{'='*60}")
