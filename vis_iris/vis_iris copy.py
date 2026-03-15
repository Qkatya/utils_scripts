import numpy as np
import cv2

landmarks_all_original = np.load('vis_iris/landmarks_all_original.npy')
landmarks_loc_netout = np.load('vis_iris/landmarks_loc_netout.npy')
sampled_frames_crop = np.load('vis_iris/sampled_frames_crop.npy')
sampled_frames = np.load('vis_iris/sampled_frames.npy')
 
frames = sampled_frames

xx = 300
yy = 420
s = 20
lmks_idxs = [473, 474, 475, 476, 477]

# Trim to first 50 frames
nn = len(landmarks_all_original) - 3
frames = frames[0:nn,:,:,:]
landmarks_all_original = landmarks_all_original[0:nn,:,:]*s
landmarks_loc_netout = landmarks_loc_netout[0:nn,:,:]*s

# Optional: Create a video writer to save the result
h, w = frames[0].shape[:2]
out_path = f"vis_iris/vid1.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

for i in range(frames.shape[0]):
    frame = frames[i].copy()

    # print(landmarks[i, :, 0]+ xx)
    
    x_landmarks_all_original=landmarks_all_original[i, :, 0] + xx
    y_landmarks_all_original=-landmarks_all_original[i, :, 1] + yy
    x_landmarks_all_original = list(map(int, x_landmarks_all_original))
    y_landmarks_all_original = list(map(int, y_landmarks_all_original))
    
    x_landmarks_iris_original=landmarks_all_original[i, lmks_idxs, 0] + xx
    y_landmarks_iris_original=-landmarks_all_original[i, lmks_idxs, 1] + yy
    x_landmarks_iris_original = list(map(int, x_landmarks_iris_original))
    y_landmarks_iris_original = list(map(int, y_landmarks_iris_original))
    
    x_landmarks_loc_netout=landmarks_loc_netout[i, :, 0] + xx
    y_landmarks_loc_netout=-landmarks_loc_netout[i, :, 1] + yy
    x_landmarks_loc_netout = list(map(int, x_landmarks_loc_netout))
    y_landmarks_loc_netout = list(map(int, y_landmarks_loc_netout))
    
    for x, y in zip(x_landmarks_all_original, y_landmarks_all_original):
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        
    for x, y in zip(x_landmarks_iris_original, y_landmarks_iris_original):
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
    for x, y in zip(x_landmarks_loc_netout, y_landmarks_loc_netout):
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Show the frame
    # cv2.imwrite(f"vis_iris/frames/frame{i}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # writer.write(frame)  # Save frame to video
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # # Break on ESC or wait
    # if cv2.waitKey(50) & 0xFF == 27:  # 50 ms ~ 20 FPS
    #     break

writer.release()
cv2.destroyAllWindows()
a=1
