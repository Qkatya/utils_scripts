from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import trimesh
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from joblib import Parallel, delayed, cpu_count
import argparse
from files_utils import safe_save
from handle_split import validate_data
from collections import Counter
from datetime import datetime
import plotly.io as pio
pio.renderers.default = 'browser'
import time

# Number of seconds in 24 hours
ten_hours_seconds = 24 * 60 * 60
now = time.time()

def is_recent(path, threshold_secs):
    return path.exists() and (now - path.stat().st_mtime) <= threshold_secs


def estimate_affine_3d(src_points, dst_points):
    assert src_points.shape == dst_points.shape
    N = src_points.shape[0]

    # Add a column of ones to convert to homogeneous coordinates
    src_h = np.hstack([src_points, np.ones((N, 1))])  # (N, 4)

    # Solve: src_h @ A = dst => A = np.linalg.lstsq(src_h, dst)
    A, res, _, _ = np.linalg.lstsq(src_h, dst_points, rcond=None)  # A: (4, 3)

    # Convert to full 4x4 matrix
    T = np.eye(4)
    T[:3, :] = A.T
    return T

def save_canonical_lmks(index, relative_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices):
    video_path = videos_path / relative_path / 'video_full.mp4'
    landmarks_path = landmarks_home_path/ relative_path/ 'landmarks.npy'
    trans_mtx_path = lmks_transformation_mtx/ relative_path/ 'trans_mtx.npy'

    if is_recent(landmarks_path, ten_hours_seconds) and is_recent(trans_mtx_path, ten_hours_seconds):
        return index, "valid", -2
 
    # set up mp
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=False,
                                        output_facial_transformation_matrixes=False,
                                        running_mode=VisionTaskRunningMode.VIDEO,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # load_video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    canonical_lmks_lst = []
    trans_mtx_lst = []
    failed = 0
    timestamp_ms = 0
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    
    # calc mp on video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # get the mediapipe model on it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms=int(timestamp_ms))
        timestamp_ms += 1000 / fps
        try:
            landmarks_mp = detection_result.face_landmarks[0]
        except:
            failed+=1
            continue
        
        landmarks_lst = [(lm.x, lm.y, lm.z) for lm in landmarks_mp]
        # landmarks = np.stack(landmarks_lst)
        try:
            landmarks = np.stack(landmarks_lst)
        except Exception as e:
            print(f"[ERROR] Failed to stack landmarks for {relative_path}: {e}")
            return index, "no_lmks", -1 
        
        # calc affine transformation between the landmarks and the canonical face
        trans = estimate_affine_3d(landmarks[:-10,:], vertices)
        landmarks_h = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
        
        # apply transformation matrix on the landmarks
        canonical_lmks_h = trans @ landmarks_h.T
        canonical_lmks = canonical_lmks_h[:3, :] / canonical_lmks_h[3,:]
        # plt.scatter(canonical_lmks[0,:], canonical_lmks[1,:])
        # plt.scatter(vertices.T[0,:], vertices.T[1,:])
        
        canonical_lmks_lst.append(canonical_lmks.T)
        trans_mtx_lst.append(trans.T)
        frame_idx += 1
        

    
    if not canonical_lmks_lst:
        print(f"[WARNING] 2 No valid landmarks detected for {relative_path}, skipping.")
        return index, "no_lmks", -1       
        
    canonical_lmks = np.stack(canonical_lmks_lst)
    
    data_status = validate_data(canonical_lmks)
    
    # if data_status == "lmks_vals_out_of_range":
    #     plot_lmks3d(canonical_lmks, "")
    
    if data_status == 'valid' and failed < 0.05*frame_count:
        # SAVE
        # save the canonical landmarks in vast: landmarks_label.shape (299, 478, 3)-(T,L,3)
        landmarks_path.parent.mkdir(parents=True, exist_ok=True)
        safe_save(canonical_lmks.astype(np.float32), landmarks_path)

        # save list of all transformation matrixes 
        trans_mtx_path.parent.mkdir(parents=True, exist_ok=True)
        safe_save(np.array(trans_mtx_lst, dtype=object).astype(np.float32), trans_mtx_path)
    
    return index, data_status, failed

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Process canonical landmarks for video frames.")
    parser.add_argument("job_idx", type=int, help="Index of the current job (0-based).")
    parser.add_argument("total_jobs", type=int, help="Total number of jobs in the job array.")
    args = parser.parse_args()
    
    # args.job_idx=0
    # args.total_jobs=6
    
    # Load df
    print("Loading DataFrame from pickle...")
    parent_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
    
    filename = "train_kfold_18p4M_whisper.pkl"
    # Set paths
    videos_path = Path("/mnt/A3000/Recordings/v2_data")
    # landmarks_home_path = Path("/mnt/ML/Development/katya.ivantsiv/landmarks0") # DEL!! - for debug
    landmarks_home_path = Path("/mnt/ML/Development/katya.ivantsiv/landmarks")
    lmks_transformation_mtx = Path('/mnt/ML/TrainResults/katya.ivantsiv/lmks_canonical_trans')    
    
    # load canonical face
    mesh = trimesh.load('canonical_face_model.obj', process=False)
    vertices = mesh.vertices
    
    # for j, filename in enumerate(filenames):            
    full_path = parent_path / filename
    df = pd.read_pickle(full_path)
    # df = df.head(10) # DEL!! - for debug
    
    run_paths = list(dict.fromkeys(df['run_path'].tolist()))
    print(f"Loaded {filename} DataFrame with {len(df)} rows, {len(run_paths)} unique run paths")
    
    sampled_runpaths = run_paths[args.job_idx :: args.total_jobs]
    # sampled_runpaths = run_paths#[0 :: 2] # DEL!! - for debug
    # print(f"job idx: {args.job_idx}")
    # print(f"total jobs: {args.total_jobs}")
    # print("---all paths---")
    # print('\n'.join(run_paths))
    # print("")
    # print("----------sampled path--------")
    # print('\n'.join(sampled_runpaths))
    
    
    job_args = [
    (i, run_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices)
    for i, run_path in enumerate(sampled_runpaths)]
    
    print("Available CPUs:", cpu_count())

    results = Parallel(n_jobs=-1)(delayed(save_canonical_lmks)(*job_arg) for job_arg in tqdm(job_args, desc="Calculating Canonical Lmks"))
    
    # === ORGANIZE RESULTS ===
    status_map = {sampled_runpaths[i]: (status, num_failed) for i, status, num_failed in results}

    # Assign status and failed frame count to DataFrame
    df['validation_status'] = df['run_path'].map(lambda p: status_map.get(p, ('missing_in_map', None))[0])
    df['num_failed_frames'] = df['run_path'].map(lambda p: status_map.get(p, ('', 0))[1])


    # Separate valid and invalid DataFrames
    df_valid = df[df['validation_status'] == 'valid'].drop(columns=['validation_status', 'num_failed_frames'])
    df_invalid = df[df['validation_status'] != 'valid']

    # === PRINT SUMMARY ===
    from collections import Counter
    counts = Counter(df.drop_duplicates(subset='run_path')['validation_status'])
    counts_existed = Counter(df.drop_duplicates(subset='run_path')['num_failed_frames'])
    summary_lines = [
        "Validation Summary:",
        f"✅ TOTAL: {len(sampled_runpaths)} samples of {len(run_paths)} unique run paths, df size {len(df)}",
        f"✅ Success: {counts.get('valid', 0)}",
        f"✅ Out of Success, Existed: {counts_existed.get(-2, 0)}",
    ]
    for reason in ['wrong_dim', 'wrong_num_lmks', 'has_nan', 'lmks_vals_out_of_range', 'no_lmks']:
        summary_lines.append(f"❌ {reason}: {counts.get(reason, 0)}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # === SAVE RESULTS ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = parent_path / "snipets" / Path(filename).name
    valid_path = f"{save_path.with_suffix('')}_valid_{timestamp}_{args.job_idx}.pkl"
    invalid_path = f"{save_path.with_suffix('')}_invalid_{timestamp}_{args.job_idx}.pkl"
    log_path = f"{save_path.with_suffix('')}_validation_summary_{timestamp}_{args.job_idx}.txt"

    df_valid.to_pickle(valid_path)
    df_invalid.to_pickle(invalid_path)

    with open(log_path, "w") as f:
        f.write(summary_text)

    print(f"\nSaved:\n  - Valid:   {valid_path}\n  - Invalid: {invalid_path}\n  - Log:     {log_path}")
    
    # print('\n'.join(sampled_runpaths))
    
    print("-------Successfull finish----------")
    # relative_path = df_valid.sample(n=3)['run_path']
    # for i in range(len(relative_path)):
    #     canonical_lmks_path = landmarks_home_path/ relative_path.iloc[i]/ 'landmarks.npy'
    #     canonical_lmks = np.load(canonical_lmks_path)
    #     plot_lmks3d(canonical_lmks, relative_path.iloc[i]+"_"+timestamp)
        
