import os
import sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from pathlib import Path
# notebook_path = Path(os.path.abspath(''))
from fairseq import checkpoint_utils, utils
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import groupby
from tqdm import tqdm
import time
import editdistance
import re
from pydub import AudioSegment
import whisper
import torchaudio
# sys.path.append('/home/oren.amsalem/cloned/urhythmic')
# from urhythmic.vocoder import HifiganGenerator
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import cv2
from IPython.display import display
from PIL import Image
import plotly.graph_objects as go

fairseq_path = "/home/katya.ivantsiv/q-fairseq-landmarks"
sys.path.insert(0, fairseq_path) 
sys.path.insert(0, f"{fairseq_path}/examples/data2vec")

# Here you place the path to the model you want to use for inference
model_path = '/mnt/ML/TrainResults/katya.ivantsiv/blendshapes/landmarks_6m/0/checkpoints/checkpoint_best.pt'

# Here you place the path to the dataframe you want to infere
df_path = '/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/LOUD_GIP_general_clean_250415_v2_with_blendshapes_cleaned2_with_attrs_with_side.pkl'



def normalize_scale(landmarks, reference_distance=1.0):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    current_distance = np.linalg.norm(left_eye - right_eye)
    scale = reference_distance / current_distance
    return landmarks * scale, scale

def get_face_rotation_matrix(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    # 1. Compute eye direction
    eye_dir = right_eye - left_eye
    eye_dir /= np.linalg.norm(eye_dir)
    
    # 2. Compute angle to rotate to x-axis
    angle = np.arctan2(eye_dir[1], eye_dir[0])
    
    # 3. Create 2D rotation matrix (counter-rotate)
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s],
                    [s,  c]])
    
    return R

def center_landmarks(landmarks):
    return landmarks - landmarks[1], landmarks[1] # Make nose tip the (0,0)


def align_orientation(landmarks):
    R = get_face_rotation_matrix(landmarks)
    center = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - center
    canonical =  (R @ landmarks_centered.T).T
    
    return canonical, R, center

def landmarks_naive_normalization(landmarks):
    landmarks_scaled, scale = normalize_scale(landmarks)
    landmarks_aligned, R, center = align_orientation(landmarks_scaled)
    landmarks_centered, nose_tip = center_landmarks(landmarks_aligned)
    return landmarks_centered, scale, R, center, nose_tip

def unnormalization(landmarks_original, canonical_landmarks):
    landmarks_centered, scale, R, center, nose_tip = landmarks_naive_normalization(landmarks_original)
    canonical_landmarks += nose_tip
    rotated_landmarks = (R.T @ canonical_landmarks.T).T + center
    reconstructed_landmarks = rotated_landmarks / scale
    return reconstructed_landmarks

def mirror_landmarks(landmarks):
    landmarks[:,0]*=-1
    return landmarks


device = 'cuda:0'
use_fp16 = True
use_cuda = True

class MyDefaultObj:
    def __init__(self):
        self.user_dir = f"{fairseq_path}/examples/data2vec"
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

utils.import_user_module(MyDefaultObj())


models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix='',strict=False,
)

for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.to(device)
        
df = pd.read_pickle(df_path)

row_ind = 0
row = df.iloc[row_ind]

st = 0
en = None

print('start loading model')

data = np.load('/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features/' + row.run_path + '/' + str(row.tar_id) + '.npy')
stack = np.array(data[st:en])
sample = torch.from_numpy(stack).to(device)
sample = sample.half()
sample = sample.float().unsqueeze(0).to(device)  # batch_size = 1
sample = sample.to(device)
padding_mask = torch.Tensor([False] * sample.shape[1]).unsqueeze(0).to(device)
sample = sample.half()

with torch.inference_mode():
    net_output = model(**{"source": sample, "padding_mask": padding_mask})
    lndmks = net_output["encoder_lndmks"]

print('finish loading model')

lmks_data_path = Path('/mnt/ML/Development/katya.ivantsiv/blendshapes')
all_data_path = Path('/mnt/A3000/Recordings/v2_data')
video_path = all_data_path / Path(row.run_path) / Path('video_full.mp4')
lmks_path = lmks_data_path / Path(row.run_path) / Path('landmarks_and_blendshapes.npz')

# --- Load data ---
with np.load(lmks_path) as data:
    landmarks_all_original = data['landmarks'] 

frame_idx = 0

padded_flattened_normalized_landmark_labels = lndmks[0, :, frame_idx]

n_landmarks = 220
unpadded = padded_flattened_normalized_landmark_labels.cpu()[:n_landmarks * 2]
landmarks_netout = unpadded.reshape(n_landmarks, 2)
if row.side == 'left':
    landmarks_netout = mirror_landmarks(landmarks_netout)

landmarks_netout_image_scale = unnormalization(landmarks_all_original[frame_idx,:,:], landmarks_netout.cpu().numpy())

# Open video and read first frame
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

if ret:
    # Draw landmarks
    frame_with_landmarks = frame.copy()
    for x_pt, y_pt in landmarks_netout_image_scale:
        cv2.circle(frame_with_landmarks, (int(x_pt), int(y_pt)), 1, (0, 255, 0), -1)
    for x_pt, y_pt in landmarks_all_original[frame_idx,:,:]:
        cv2.circle(frame_with_landmarks, (int(x_pt), int(y_pt)), 1, (0, 0, 255), -1)

    # Convert and display
    frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(frame_rgb))
else:
    print("Failed to read the video.")
    

