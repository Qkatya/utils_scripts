import os
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import mediapipe as mp
import cv2
import trimesh
from pathlib import Path
import cv2
import numpy as np
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method
import multiprocessing as mup
import argparse
import contextlib
import shutil
from easydict import EasyDict as edict

def safe_save(data: np.ndarray, save_path: Path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_save_path = save_path.with_suffix(".tmp.npy")
    np.save(tmp_save_path, data)
    shutil.move(tmp_save_path, save_path.with_suffix(".npy"))
    
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


def save_canonical_lmks(index, relative_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices, tj):
    # print(f"Starting row {index}")

    video_path = videos_path / relative_path / 'video_full.mp4'
    landmarks_path = landmarks_home_path/ relative_path/ 'landmarks.npy'
    trans_mtx_path = lmks_transformation_mtx/ relative_path/ 'trans_mtx.npy'
    
    if landmarks_path.exists():# and trans_mtx_path.exists():
        # return 
        try:
            a = np.load(landmarks_path)
            # return 0
        except:
            os.remove(landmarks_path)
            print(f"####################################### deprecated file, removed {landmarks_path} ################################################")
            # return 1
        arr = np.load(landmarks_path)
        if arr.dtype == np.float64:
            os.remove(landmarks_path)
            arr_32 = arr.astype(np.float32)
            np.save(landmarks_path, arr_32)
        return
    # else:
    #     return 0

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

    # video_frames = []
    canonical_lmks_lst = []
    trans_mtx_lst = []
    failed = 0
    timestamp_ms = 0
    frame_idx = 0

    # calc mp on video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # video_frames.append(frame)

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
        landmarks = np.stack(landmarks_lst)
        
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
        

    canonical_lmks = np.stack(canonical_lmks_lst)
    
    # SAVE
    # save the canonical landmarks in vast: landmarks_label.shape (299, 478, 2)-(T,L,2)
    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    # safe_save(canonical_lmks, landmarks_path)
    np.save(landmarks_path, canonical_lmks.astype(np.float32))

    # save list of all transformation matrixes 
    trans_mtx_path.parent.mkdir(parents=True, exist_ok=True)
    # safe_save(np.array(trans_mtx_lst, dtype=object), trans_mtx_path)
    np.save(trans_mtx_path, np.array(trans_mtx_lst, dtype=object).astype(np.float32))
    
    print(f"####################################### Saved row {index} out of {tj} ################################################")


def check_saved_jobs(df, landmarks_home_path):
    count_done_jobs = 0
    failed_jobs = []
    existing_jobs = []
    for i in tqdm(range(args.job_idx, len(df), args.total_jobs)):
        row = df.iloc[i]
        relative_path = Path(row['run_path'])
        landmarks_path = landmarks_home_path/ relative_path/ 'landmarks.npy'
        if landmarks_path.exists():
            if landmarks_path.stat().st_size > 1000:
                count_done_jobs += 1
                existing_jobs.append(row)
                continue
        failed_jobs.append(row)
    return pd.DataFrame(failed_jobs), pd.DataFrame(existing_jobs), count_done_jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process canonical landmarks for video frames.")

    # Positional arguments
    parser.add_argument("input_file", type=str, help="Path to the input .pkl or list file.")
    parser.add_argument("job_idx", type=int, help="Index of the current job (0-based).")
    parser.add_argument("total_jobs", type=int, help="Total number of jobs in the job array.")

    # Optional arguments
    parser.add_argument("--canon-lmks-save-dir", type=str,default="/mnt/ML/Development/katya.ivantsiv/landmarks",help="Directory to save canonicalized landmarks.")
    parser.add_argument("--trns-mtx-save-dir", type=str,default="/mnt/ML/TrainResults/katya.ivantsiv/lmks_canonical_trans",help="Directory to save transformation matrices.")
    parser.add_argument("--videos-path", type=str,default="/mnt/A3000/Recordings/v2_data",help="Directory where the videos are stored.")

    args = parser.parse_args()

    job_name = 'create_canonical_lmks'
    
    # from multiprocessing import set_start_method
    # set_start_method("spawn", force=True)
    
    
    # Set paths0
    videos_path = Path(args.videos_path)
    landmarks_home_path = Path(args.canon_lmks_save_dir)
    lmks_transformation_mtx = Path(args.trns_mtx_save_dir)

    # load canonical face
    mesh = trimesh.load('/home/katya.ivantsiv/utils_scripts/canonical_face_model.obj', process=False)
    vertices = mesh.vertices

    # joblib
    # Load df
    print("Loading DataFrame from pickle...")
    df = pd.read_pickle(args.input_file)
    print(f"Loaded DataFrame with {len(df)} rows.")
    
    run_paths = list(set(df['run_path'].tolist()))

    print(f"Processing {len(run_paths)//args.total_jobs} rows", flush=True)
    
    # print("Checking saved files...")
    # jobs_to_run_df, existing_jobs, count_done_jobs = check_saved_jobs(df, landmarks_home_path)
    # print(f"Done {count_done_jobs} jbs, {len(jobs_to_run_df)} jobs left to do.")
    
    stats = edict(success=0, failed=0, exists=0)
    j = 0
    new_df = pd.DataFrame()
    for i in tqdm(range(args.job_idx, len(run_paths), args.total_jobs)):
    # for i in tqdm(range(len(existing_jobs))):
        relative_path = run_paths[i]
        f = save_canonical_lmks(i, relative_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices, len(df)/args.total_jobs)
        if f == 1:
            rows = df[df['run_path'] == relative_path]
            new_df = pd.concat([new_df, rows])
            stats.exists += 1
        else:
            stats.failed += 1
        if j%100 == 0:
            print(f'{job_name} stats={{"exists": {stats.exists}, "not exists": {stats.failed}}}')
        j+=1
    
    
    new_df.attrs = df.attrs
    output_path = Path('/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/existing_files')
    output_path.mkdir(parents=True, exist_ok=True) 
    new_df.to_pickle(output_path / f'job_{args.job_idx}.pkl')
    
    print(f"Finished {job_name}. Tar stats: {stats}", flush=True)
    
    # for i in tqdm(range(len(jobs_to_run_df))):
    #     row = jobs_to_run_df.iloc[i]
    #     relative_path = Path(row['run_path'])
    #     save_canonical_lmks(i, relative_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices, len(df)/args.total_jobs)
        

# import plotly.io as pio
# pio.renderers.default = 'browser'
# import numpy as np
# import plotly.graph_objects as go
# import plotly.io as pio

# pio.renderers.default = 'browser'  # Open plots in the default web browser

# import plotly.graph_objects as go
# import numpy as np


# # Create animation frames
# frames = [
#     go.Frame(
#         data=[
#             go.Scatter(
#                 x=canonical_lmks[frame_idx, :, 0],
#                 y=canonical_lmks[frame_idx, :, 1],
#                 mode='markers',
#                 marker=dict(size=4, color='blue')
#             )
#         ],
#         name=str(frame_idx)
#     )
#     for frame_idx in range(canonical_lmks.shape[0])
# ]

# # Base figure with first frame
# fig = go.Figure(
#     data=[
#         go.Scatter(
#             x=canonical_lmks[0, :, 0],
#             y=canonical_lmks[0, :, 1],
#             mode='markers',
#             marker=dict(size=4, color='blue')
#         )
#     ],
#     layout=go.Layout(
#         title='Canonical Landmarks (2D)',
#         xaxis=dict(title='X'),
#         yaxis=dict(title='Y', scaleanchor='x', scaleratio=1),
#         sliders=[{
#             'steps': [
#                 {
#                     'method': 'animate',
#                     'args': [[str(i)], {'mode': 'immediate', 'frame': {'duration': 0}, 'transition': {'duration': 0}}],
#                     'label': str(i)
#                 } for i in range(canonical_lmks.shape[0])
#             ],
#             'transition': {'duration': 0},
#             'x': 0.1, 'y': -0.1,
#             'currentvalue': {'prefix': 'Frame: '}
#         }],
#         updatemenus=[{
#             'type': 'buttons',
#             'showactive': False,
#             'y': 1.1,
#             'x': 1.05,
#             'xanchor': 'right',
#             'yanchor': 'top',
#             'buttons': [
#                 {
#                     'label': 'Play',
#                     'method': 'animate',
#                     'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
#                 },
#                 {
#                     'label': 'Pause',
#                     'method': 'animate',
#                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
#                 }
#             ]
#         }]
#     ),
#     frames=frames
# )

# fig.show()