from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import trimesh

pio.renderers.default = 'browser'  # Opens in your system browser


# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
video_path = '/mnt/A3000/Recordings/v2_data/2025/05/13/AdaptiveTrusting-175502/12_0_2e11a422-a8ef-4627-a733-966a71f1eb2c_loud/video_full.mp4'
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()  # Read the first frame
cap.release()

if not success:
    raise RuntimeError("Failed to read frame from video")

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
detection_result = detector.detect(image)
matrix = detection_result.facial_transformation_matrixes[0]

a=1















# mesh = trimesh.load('canonical_face_model.obj', process=False)
# vertices = mesh.vertices
# faces = mesh.faces

# matrix = detection_result.facial_transformation_matrixes[0]

# # Apply transformation (make vertices homogeneous: N x 4)
# vertices_h = np.hstack([vertices, np.ones((vertices.shape[0], 1))])
# transformed = (matrix @ vertices_h.T).T
# transformed = transformed[:, :3] / transformed[:, 3:]


# # Extract x, y, z coordinates
# x, y, z = transformed[:, 0], transformed[:, 1], transformed[:, 2]

# # Extract i, j, k triangle indices
# i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

# # Create Mesh3D plot
# fig = go.Figure(data=[
#     go.Mesh3d(
#         x=x, y=y, z=z,
#         i=i, j=j, k=k,
#         color='lightpink',
#         opacity=1.0,
#         flatshading=True,
#         name='Transformed Face Mesh'
#     )
# ])

# fig.update_layout(
#     title='Canonical Face Model Transformed',
#     scene=dict(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         zaxis=dict(visible=False),
#         aspectmode='data'
#     )
# )

# fig.show()
# a=1
