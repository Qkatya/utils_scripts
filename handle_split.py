from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pickle
import argparse
from sqlalchemy import create_engine
from joblib import Parallel, delayed, cpu_count
from datetime import datetime
from collections import Counter
from landmarks_utils import plot_lmks3d

ADD_ATTRS = False #True
ADD_SIDE = False #True
GET_STATUS = True
SAVE_SNIPPET = True

def check_tarid_in_df(df, tar_id_to_check):
    if tar_id_to_check in df['tar_id'].values:
        print("tar_id is in the DataFrame.")
    else:
        print("tar_id is NOT in the DataFrame.")
    
    
def load_dataframe(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_dataframe(df, path):
    with open(path, 'wb') as f:
        pickle.dump(df, f)


def unique_runpaths(df):
    run_paths = list(dict.fromkeys(df['run_path'].tolist()))
    return run_paths


def transfer_attrs(new_df):
    attrs_src_file_path = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits') / "LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl"
    attrs_src_file = load_dataframe(attrs_src_file_path)
    new_df.attrs = attrs_src_file.attrs
    return new_df

def get_df_with_side(df_with_tar_id, batch_size=100000):
    assert df_with_tar_id.tar_id.value_counts().max()==1
    assert isinstance(df_with_tar_id.tar_id.iloc[0], str)
    
    # DB setup
    engine = create_engine("postgresql://bits_viewer:qviewer1@q-data-db-replica.q.ai:5432/q-data-db-prod")

    results = []
    
    tar_ids = df_with_tar_id.tar_id
    
    for i in tqdm(range(0, len(tar_ids), batch_size)):
        batch_ids = tar_ids[i:i + batch_size]
        placeholders = ",".join([f"'{x}'" for x in batch_ids])
        #dwh.q_frames_details
        query = """
        select id as tar_id, side
        from q_frames_files
        WHERE id IN ({})
        """.format(placeholders)
        
        # df_batch = pd.read_sql(query, engine,params batch_ids)
        df_batch = pd.read_sql(query, engine)
        df_batch["tar_id"] = df_batch["tar_id"].astype(str)
        
        results.append(df_batch)
    
    df_all_results = pd.concat(results, ignore_index=True)
    
    # Merge back into original df
    df_with_tar_id = df_with_tar_id.merge(df_all_results, how='left', on='tar_id')

    return df_with_tar_id

def query_batch(batch_ids):
    engine = create_engine("postgresql://bits_viewer:qviewer1@q-data-db-replica.q.ai:5432/q-data-db-prod")
    placeholders = ",".join([f"'{x}'" for x in batch_ids])
    
    query = f"""
    SELECT id AS tar_id, side
    FROM q_frames_files
    WHERE id IN ({placeholders})
    """
    
    df_batch = pd.read_sql(query, engine)
    df_batch["tar_id"] = df_batch["tar_id"].astype(str)
    return df_batch

def get_side_from_db_parallel(df_with_tar_id, n_jobs=10, batch_size=100000):
    assert df_with_tar_id.tar_id.value_counts().max() == 1
    assert isinstance(df_with_tar_id.tar_id.iloc[0], str)

    tar_ids = df_with_tar_id.tar_id.tolist()
    
    # Create batches
    batches = [tar_ids[i:i + batch_size] for i in range(0, len(tar_ids), batch_size)]
    
    # Run queries in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(query_batch)(batch) for batch in tqdm(batches)
    )
    
    # Combine results
    df_all_results = pd.concat(results, ignore_index=True)
    
    # Merge back into original df
    df_with_tar_id = df_with_tar_id.merge(df_all_results, how='left', on='tar_id')
    
    return df_with_tar_id

def validate_data(data):
    if data.ndim != 3:
        return 'wrong_dim'
    if data.shape[1] != 478:
        return 'wrong_num_lmks'
    if np.isnan(data).any():
        return 'has_nan'
    if not np.all((data >= -15) & (data <= 15)):
        return 'lmks_vals_out_of_range'
    return 'valid'
    
def validate_row(index, run_path):
    vast_path = Path("/mnt/ML/Development/katya.ivantsiv/landmarks")
    npy_path = vast_path / Path(run_path) / "landmarks.npy"

    try:
        if not npy_path.exists():
            return index, 'missing_file'
        data = np.load(npy_path)
        data_status = validate_data(data)
        return index, data_status
    
    except Exception:
        return index, 'load_error'
    
def get_stats_df(df, file_path, njobs=10, save_snippet=False):
    run_paths = list(dict.fromkeys(df['run_path'].tolist())) # list(set(df['run_path'].tolist()))
    print(f"Loaded DataFrame with {len(df)} rows, {len(run_paths)} unique run paths")
    
    print(f"Available CPUs:, {njobs}")
    results = Parallel(n_jobs=njobs)(delayed(validate_row)(i, row) for i, row in tqdm(enumerate(run_paths)))
    
    # === ORGANIZE RESULTS ===
    # status_map = {i: status for i, status in results}
    # df['validation_status'] = df.index.map(status_map)

    # df_valid = df[df['validation_status'] == 'valid'].drop(columns=['validation_status'])
    # df_invalid = df[df['validation_status'] != 'valid'].rename(columns={'validation_status': 'fail_reason'})

    
    status_map = {run_paths[i]: status for i, status in results}

    # Assign status and failed frame count to DataFrame
    df['validation_status'] = df['run_path'].map(lambda p: status_map.get(p, ('valid', None)))

    # Separate valid and invalid DataFrames
    df_valid = df[df['validation_status'] == 'valid'].drop(columns=['validation_status'])
    df_invalid = df[df['validation_status'] != 'valid']
    
    
    # === PRINT SUMMARY ===
    counts = Counter(status_map.values())
    summary_lines = [
        "Validation Summary:",
        file_path.name,
        f"❗ TOTAL: {len(run_paths)} unique run paths, df size {len(df)}",
        f"✅ Success: {counts.get('valid', 0)}",
    ]
    for reason in ['missing_file', 'load_error', 'wrong_dim', 'has_nan', 'lmks_vals_out_of_range']:
        summary_lines.append(f"❌ {reason}: {counts.get(reason, 0)}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

        # === SAVE RESULTS ===   
    if save_snippet:
        parent_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = parent_path / "snipets" / Path(file_path.name).name
        valid_path = f"{save_path.with_suffix('')}_valid_{timestamp}.pkl"
        invalid_path = f"{save_path.with_suffix('')}_invalid_{timestamp}.pkl"
        log_path = f"{save_path.with_suffix('')}_validation_summary_{timestamp}.txt"
    
    
        df_valid.to_pickle(valid_path)
        df_invalid.to_pickle(invalid_path)
    
        with open(log_path, "w") as f:
            f.write(summary_text)

        print(f"\nSaved:\n  - Valid:   {valid_path}\n  - Invalid: {invalid_path}\n  - Log:     {log_path}")
    
    # relative_path = df_valid.sample(n=3)['run_path']
    # for i in range(len(relative_path)):
    #     canonical_lmks_path = landmarks_home_path/ relative_path.iloc[i]/ 'landmarks.npy'
    #     canonical_lmks = np.load(canonical_lmks_path)
    #     plot_lmks3d(canonical_lmks, relative_path.iloc[i]+"_"+timestamp)
    
    return summary_text
        

if __name__ == "__main__":
    path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
    lst = ["loud_and_whisper_and_lip_20250713_064722.pkl"] 

    df = lst[0]
    df_path = path / df
    df = load_dataframe(df_path)
    
    summary_text_list = []
    for df in lst:
        df_path = path / df
        df = load_dataframe(df_path)
        # df = df.head(100) # DEL!! - for debug
        if ADD_ATTRS:
            transfer_attrs(df)
        
        if ADD_SIDE:
            get_df_with_side(df)
            
        if GET_STATUS:
            summary_text = get_stats_df(df, df_path, -1, save_snippet = SAVE_SNIPPET)
            summary_text_list.append(summary_text)
    for summary_text in summary_text_list:
        print(summary_text)    
a=1