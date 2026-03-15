import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

video_path = '/mnt/A3000/Recordings/v2_data/2025/03/19/NuggetCitadel-113505/88_0_6ba7722d-2f2a-4ad6-9c3e-971cc869dcd1_loud/video_full.mp4'
landmarks_path = '/mnt/ML/Development/katya.ivantsiv/landmarks/2025/03/19/NuggetCitadel-113505/88_0_6ba7722d-2f2a-4ad6-9c3e-971cc869dcd1_loud/landmarks.npy'

xx = 300
yy = 420
s = 20
lmks_idxs = [473, 474, 475, 476, 477]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]


landmarks = np.load(landmarks_path)

cap = cv2.VideoCapture(str(video_path))
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = frame[225:611, 527:864]
    frames.append(frame)
cap.release()

frames = np.array(frames)

# Drop every 6th frame
mask = np.arange(frames.shape[0]) % 6 != 5
sampled_frames = frames[mask]
sampled_landmarks = landmarks[mask]
landmarks_netout = np.load('vis_iris/landmarks_loc_netout.npy')

sampled_frames = sampled_frames[:120,:,:]
sampled_landmarks = sampled_landmarks[:120,:,:]*s
landmarks_netout = landmarks_netout[:120,:,:]*s

# fig = go.Figure()
# fig.add_trace(go.Scatter(y=sampled_landmarks[:,-1,0], mode='lines'))
# fig.add_trace(go.Scatter(y=landmarks_netout[:,0,0], mode='lines'))
# fig.show()

for i in tqdm(range(sampled_frames.shape[0])):
    frame = sampled_frames[i].copy()

    # print(landmarks[i, :, 0]+ xx)
    
    x_landmarks_all_original=sampled_landmarks[i, :, 0] + xx
    y_landmarks_all_original=-sampled_landmarks[i, :, 1] + yy
    
    x_landmarks_right_eye_original=sampled_landmarks[i, LEFT_EYE, 0] + xx
    y_landmarks_right_eye_original=-sampled_landmarks[i, LEFT_EYE, 1] + yy

    x_landmarks_iris_original=sampled_landmarks[i, lmks_idxs, 0] + xx
    y_landmarks_iris_original=-sampled_landmarks[i, lmks_idxs, 1] + yy
    
    x_landmarks_loc_netout=landmarks_netout[i, :, 0] + xx
    y_landmarks_loc_netout=-landmarks_netout[i, :, 1] + yy
    
    # Plot
    fig, ax = plt.subplots(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
    ax.imshow(frame)
    ax.scatter(x_landmarks_all_original, y_landmarks_all_original, c='b', s=10) 
    ax.scatter(x_landmarks_iris_original, y_landmarks_iris_original, c='g', s=10)  
    ax.scatter(x_landmarks_loc_netout, y_landmarks_loc_netout, c='r', s=10) 
    
    xxx = 500
    yyy = 50

    # ax.scatter(x_landmarks_all_original+xxx, y_landmarks_all_original+yyy, c='b', s=10)  
    ax.scatter(x_landmarks_right_eye_original+xxx, y_landmarks_right_eye_original+yyy, c='b', s=10)  
    ax.scatter(x_landmarks_iris_original+xxx, y_landmarks_iris_original+yyy, c='g', s=10)  
    ax.scatter(x_landmarks_loc_netout+xxx, y_landmarks_loc_netout+yyy, c='r', s=10) 

    # Remove axes, ticks, border
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save
    fig.savefig(f"vis_iris/frames/frame{i}.png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

frames_dir = Path("vis_iris/frames")
frame_files = [str(p) for p in frames_dir.glob("frame*.png")]
# frame_files = natsort.natsorted(frame_files)  # sort by frame index

# Read the first frame to get dimensions
frame_sample = cv2.imread(frame_files[0])
h, w, _ = frame_sample.shape

# Define the video writer
out_path = "vis_iris/overlay_video6.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

# Write each frame
for file_path in frame_files:
    frame = cv2.imread(file_path)
    writer.write(frame)

writer.release()
print(f"Video saved to: {out_path}")


a=1



# ################
# landmarks_all_original = np.load('vis_iris/landmarks_all_original.npy')
# landmarks_loc_netout = np.load('vis_iris/landmarks_loc_netout.npy')
# sampled_frames_crop = np.load('vis_iris/sampled_frames_crop.npy')
# sampled_frames = np.load('vis_iris/sampled_frames.npy')


# mask = np.arange(landmarks_all_original.shape[0]) % 6 != 5
# landmarks_all_original = landmarks_all_original[mask]
    
# frames = sampled_frames

# xx = 300
# yy = 420
# s = 20
# lmks_idxs = [473, 474, 475, 476, 477]
# RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
# LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# # Trim to first 50 frames
# nn = len(landmarks_all_original) - 4
# frames = frames[0:nn,:,:,:]
# landmarks_all_original = landmarks_all_original[0:nn,:,:]*s
# landmarks_loc_netout = landmarks_loc_netout[0:nn,:,:]*s

# # Optional: Create a video writer to save the result
# h, w = frames[0].shape[:2]

# for i in range(frames.shape[0]):
#     frame = frames[i].copy()

#     # print(landmarks[i, :, 0]+ xx)
    
#     x_landmarks_all_original=landmarks_all_original[i, :, 0] + xx
#     y_landmarks_all_original=-landmarks_all_original[i, :, 1] + yy
    
#     x_landmarks_right_eye_original=landmarks_all_original[i, LEFT_EYE, 0] + xx
#     y_landmarks_right_eye_original=-landmarks_all_original[i, LEFT_EYE, 1] + yy
#     # x_landmarks_right_eye_original=landmarks_all_original[i, RIGHT_EYE, 0] + xx
#     # y_landmarks_right_eye_original=-landmarks_all_original[i, RIGHT_EYE, 1] + yy

#     x_landmarks_iris_original=landmarks_all_original[i, lmks_idxs, 0] + xx
#     y_landmarks_iris_original=-landmarks_all_original[i, lmks_idxs, 1] + yy
    
#     x_landmarks_loc_netout=landmarks_loc_netout[i, :, 0] + xx
#     y_landmarks_loc_netout=-landmarks_loc_netout[i, :, 1] + yy
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
#     ax.imshow(frame)
#     ax.scatter(x_landmarks_all_original, y_landmarks_all_original, c='b', s=10) 
#     ax.scatter(x_landmarks_iris_original, y_landmarks_iris_original, c='g', s=10)  
#     ax.scatter(x_landmarks_loc_netout, y_landmarks_loc_netout, c='r', s=10) 
    
#     xxx = 500
#     yyy = 50

#     # ax.scatter(x_landmarks_all_original+xxx, y_landmarks_all_original+yyy, c='b', s=10)  
#     ax.scatter(x_landmarks_right_eye_original+xxx, y_landmarks_right_eye_original+yyy, c='b', s=10)  
#     ax.scatter(x_landmarks_iris_original+xxx, y_landmarks_iris_original+yyy, c='g', s=10)  
#     ax.scatter(x_landmarks_loc_netout+xxx, y_landmarks_loc_netout+yyy, c='r', s=10) 

#     # Remove axes, ticks, border
#     ax.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

#     # Save
#     fig.savefig(f"vis_iris/frames/frame{i}.png", dpi=100, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)

# frames_dir = Path("vis_iris/frames")
# frame_files = [str(p) for p in frames_dir.glob("frame*.png")]
# # frame_files = natsort.natsorted(frame_files)  # sort by frame index

# # Read the first frame to get dimensions
# frame_sample = cv2.imread(frame_files[0])
# h, w, _ = frame_sample.shape

# # Define the video writer
# out_path = "vis_iris/overlay_video3.mp4"
# writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

# # Write each frame
# for file_path in frame_files:
#     frame = cv2.imread(file_path)
#     writer.write(frame)

# writer.release()
# print(f"Video saved to: {out_path}")