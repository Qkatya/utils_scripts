from pathlib import Path
import pandas as pd
import numpy as np
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
from joblib import Parallel, delayed, cpu_count
import shutil
import argparse

def safe_save(data: np.ndarray, save_path: Path):
    if data.size == 0:
        print(f"[WARNING] Empty array, nothing saved to {save_path}")
        return
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


def save_canonical_lmks(index, relative_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices):
    # print(f"Starting row {index}")

    # relative_path = Path(row['run_path'])
    video_path = videos_path / relative_path / 'video_full.mp4'
    landmarks_path = landmarks_home_path/ relative_path/ 'landmarks.npy'
    trans_mtx_path = lmks_transformation_mtx/ relative_path/ 'trans_mtx.npy'

    if landmarks_path.exists() and trans_mtx_path.exists():
        return 
 
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
            return  # or raise, or return None, depending on what makes sense in your function
        
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
        

    # canonical_lmks = np.stack(canonical_lmks_lst)
    if not canonical_lmks_lst:
        print(f"[WARNING] 2 No valid landmarks detected for {relative_path}, skipping.")
        return
    
    if failed > 0:
        print(f'fained {failed} failed frames in video {relative_path}')
        
    # SAVE
    # save the canonical landmarks in vast: landmarks_label.shape (299, 478, 2)-(T,L,2)
    landmarks_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(canonical_lmks.astype(np.float32), landmarks_path)
    # np.save(landmarks_path, canonical_lmks.astype(np.float32))

    # save list of all transformation matrixes 
    trans_mtx_path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(np.array(trans_mtx_lst, dtype=object).astype(np.float32), trans_mtx_path)
    # np.save(trans_mtx_path, np.array(trans_mtx_lst, dtype=object).astype(np.float32))
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Process canonical landmarks for video frames.")
    # parser.add_argument("--df_name", type=str, help="Path to the input .pkl or list file.")
    # args = parser.parse_args()
    # filename = Path(args.df_name)
    
    # Load df
    print("Loading DataFrame from pickle...")
    path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits")
    # filename = "6M_20250220_loud_whisperwer_lower_0p1_with_attrs_with_blendshapes_cleaned2_with_side.pkl"
    # filename = "LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl"
    # filenames = ["LIP_GIP_general_clean_250415_v2_with_side_attrs.pkl",
    #             "WHISPER_GIP_general_clean_250415_v2_with_side_attrs.pkl",
    #             "SILENT_GIP_general_clean_250415_v2_with_side_attrs.pkl"]
    
    filename = "train_kfold_18p4M_whisper.pkl"
    # Set paths
    videos_path = Path("/mnt/A3000/Recordings/v2_data")
    landmarks_home_path = Path("/mnt/ML/Development/katya.ivantsiv/landmarks")
    lmks_transformation_mtx = Path('/mnt/ML/TrainResults/katya.ivantsiv/lmks_canonical_trans')    
    
    # load canonical face
    mesh = trimesh.load('canonical_face_model.obj', process=False)
    vertices = mesh.vertices
    
    # for j, filename in enumerate(filenames):            
    full_path = path / filename
    df = pd.read_pickle(full_path)
    
    run_paths = list(set(df['run_path'].tolist()))
    print(f"Loaded {filename} DataFrame with {len(df)} rows, {len(run_paths)} unique run paths")
    
    job_args = [
    (i, run_path, videos_path, landmarks_home_path, lmks_transformation_mtx, vertices)
    for i, run_path in enumerate(run_paths)]
    
    print("Available CPUs:", cpu_count())

    Parallel(n_jobs=-1)(delayed(save_canonical_lmks)(*job_arg) for job_arg in tqdm(job_args, desc="Calculating Canonical Lmks"))

        

# #load-example
# loaded = np.load(landmarks_path, allow_pickle=True)
# trans_mtx = np.load(trans_mtx_path, allow_pickle=True)


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