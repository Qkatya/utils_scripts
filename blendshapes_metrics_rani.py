from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import plotly.io as pio
import random
from tqdm import tqdm
from scipy.signal import find_peaks

pio.renderers.default = "browser"  # open in your default browser
def create_video_slider_plot(video_path, blendshapes_data, blendshape_idx, blendshape_name="Blendshape", face_crop=True, crop_margin=50):
    """
    Create a Plotly plot with video frames (with slider + play/pause)
    and a static plot of one blendshape underneath.

    Args:
        video_path: Path to the video file
        blendshapes_data: numpy array (frames x blendshapes)
        blendshape_idx: index of blendshape to plot
        blendshape_name: label for blendshape curve
        face_crop: Whether to crop to face region (keeps original resolution)
        crop_margin: Extra pixels around face bounding box
    """
    import cv2
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Calculate face crop dimensions
    if face_crop:
        face_width = original_width // 3
        face_height = original_height // 3
        x_start = (original_width - face_width) // 2
        y_start = (original_height - face_height) // 2
        x_start = max(0, x_start - crop_margin)
        y_start = max(0, y_start - crop_margin)
        x_end = min(original_width, x_start + face_width + 2 * crop_margin)
        y_end = min(original_height, y_start + face_height + 2 * crop_margin)
        width = x_end - x_start
        height = y_end - y_start
    else:
        width, height = original_width, original_height
        x_start = y_start = 0
        x_end, y_end = original_width, original_height

    # Make subplots (video + 5 blendshape curves)
    fig = make_subplots(
        rows=6, cols=1,  # 1 video + 5 blendshapes
        row_heights=[0.45, 0.11, 0.11, 0.11, 0.11, 0.11],  # Video gets more space, each blendshape gets 11%
        shared_xaxes=False,
        vertical_spacing=0.07, #0.05,  # Reduced spacing to fit everything
        subplot_titles=("Video Player", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "eyeLookOutLeft", "eyeLookInLeft")
    )

    # Add video placeholder in row 1
    fig.add_trace(
        go.Image(
            z=np.zeros((height, width, 3), dtype=np.uint8),
            name="Video Frame"
        ),
        row=1, col=1
    )

    # Add all blendshape lines
    frame_numbers = np.arange(len(blendshapes_data))
    
    # Define all the blendshapes to plot
    blendshapes_to_plot = [
        ("eyeBlinkRight", 2, "red"),
        ("jawOpen", 3, "blue"),
        ("mouthSmileLeft", 4, "green"),
        ("eyeLookOutLeft", 5, "orange"),
        ("eyeLookInLeft", 6, "purple")
    ]
    
    for blendshape_name, row_idx, color in blendshapes_to_plot:
        try:
            blendshape_idx = BLENDSHAPES_ORDERED.index(blendshape_name)
            blendshape_values = blendshapes_data[:, blendshape_idx]
            if blendshape_name == "eyeBlinkRight":
                blink_th = 0.44
                peaks, _ = find_peaks(blendshape_values, height=blink_th, distance=5, width=(None, 20)) # width less than 20 frames, min dist between peaks 10 frames
                if np.mean(blendshape_values > blink_th) > 0.5:
                    peaks = []
                # Add number of peaks to the subplot title for eyeBlinkRight
                fig.layout.annotations[1].text = f"eyeBlinkRight (# Blinks: {len(peaks)})"
            if blendshape_name == "jawOpen":
                blendshape_values = np.diff(blendshape_values)
                blendshape_values = np.concatenate([[0], blendshape_values])
            fig.add_trace(
                go.Scatter(
                    x=frame_numbers,
                    y=blendshape_values,
                    mode="lines",
                    name=blendshape_name,
                    line=dict(color=color, width=2),
                    hovertemplate="Frame: %{x}<br>Value: %{y:.3f}<extra></extra>"
                ),
                row=row_idx, col=1
            )
        except ValueError:
            print(f"{blendshape_name} blendshape not found.")

    # Create frames for video
    frames = []
    step = max(1, total_frames // 100)  # sample max 100 frames
    for frame_idx in range(0, total_frames, step):
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_cropped = frame_rgb[y_start:y_end, x_start:x_end, :] if face_crop else frame_rgb
            frame_data = [go.Image(z=frame_cropped)]
        else:
            frame_data = [go.Image(z=np.zeros((height, width, 3), dtype=np.uint8))]
        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    fig.frames = frames

    # Slider for video only
    fig.update_layout(
        sliders=[{
            "currentvalue": {"prefix": "Frame: "},
            "len": 1.0,  # Full width of the plot
            "x": 0.0,    # Start from left edge
            "xanchor": "left",
            "y": -0.15,  # Position lower below the plot
            "yanchor": "bottom",
            "steps": [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate"}],
                    "label": str(i),
                    "method": "animate"
                } for i in range(0, total_frames, 1)  # Step by 1 frame for smooth movement
            ]
        }]
    )

    # Play/pause for video only
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.1,
            "y": 1.1,
            "xanchor": "left",
            "yanchor": "top",
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 50, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]}
            ]
        }]
    )

    fig.update_layout(height=800, width=900, showlegend=False)
    return fig

def create_combined_video_blendshape_plot(video_path, blendshapes_data, blendshape_name='eyeBlinkRight', face_crop=True, crop_margin=60):
    """
    Create a combined plot with video on top and blendshape graph below, using one shared slider.
    
    Args:
        video_path: Path to the video file
        blendshapes_data: numpy array of blendshape values
        blendshape_name: name of the blendshape to plot
        face_crop: Whether to crop video to face region
        crop_margin: Extra pixels around face bounding box
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Find the index of the blendshape
    try:
        blendshape_idx = BLENDSHAPES_ORDERED.index(blendshape_name)
    except ValueError:
        print(f"Blendshape '{blendshape_name}' not found. Using first blendshape.")
        blendshape_idx = 0
    
    # Get video properties
    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Get blendshape data length
    total_blendshape_frames = len(blendshapes_data)
    
    # Cut both to the same length (minimum of the two)
    min_frames = min(total_video_frames, total_blendshape_frames)
    
    print(f"Video frames: {total_video_frames}")
    print(f"Blendshape frames: {total_blendshape_frames}")
    print(f"Using minimum length: {min_frames}")
    
    # Truncate blendshape data to match minimum length
    blendshapes_data = blendshapes_data[:min_frames]
    
    # Calculate face crop dimensions
    if face_crop:
        face_width = original_width // 3
        face_height = original_height // 3
        x_start = (original_width - face_width) // 2
        y_start = (original_height - face_height) // 2
        
        x_start = max(0, x_start - crop_margin)
        y_start = max(0, y_start - crop_margin)
        x_end = min(original_width, x_start + face_width + 2 * crop_margin)
        y_end = min(original_height, y_start + face_height + 2 * crop_margin)
        
        # After cropping, downsample by 8x8
        cropped_width = x_end - x_start
        cropped_height = y_end - y_start
        width = cropped_width // 8
        height = cropped_height // 8
        
        print(f"Original dimensions: {original_width}x{original_height}")
        print(f"Face crop region: ({x_start}, {y_start}) to ({x_end}, {y_end})")
        print(f"Cropped dimensions: {cropped_width}x{cropped_height}")
        print(f"Final downsampled dimensions: {width}x{height}")
        print(f"Total data reduction: {original_width * original_height / (width * height):.1f}x")
    else:
        width = original_width
        height = original_height
        x_start = y_start = 0
        x_end = original_width
        y_end = original_height
    
    # Get frame numbers and values
    frame_numbers = np.arange(min_frames)
    blendshape_values = blendshapes_data[:, blendshape_idx]
    
    # Create subplots: video on top, blendshape graph below
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[f'Video Frame (Face Cropped)', f'{blendshape_name} Over Time'],
        vertical_spacing=0.1,
        row_heights=[0.3, 0.2]  # Reduced heights by 2x
    )
    
    # Add video frame placeholder
    fig.add_trace(
        go.Image(
            z=np.zeros((height, width, 3), dtype=np.uint8),
            name='Video Frame'
        ),
        row=1, col=1
    )
    
    # Add the blendshape line plot
    fig.add_trace(
        go.Scatter(
            x=frame_numbers,
            y=blendshape_values,
            mode='lines',
            name=blendshape_name,
            line=dict(color='red', width=2),
            hovertemplate='Frame: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Add a vertical line to show current frame position
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[blendshape_values.min(), blendshape_values.max()],
            mode='lines',
            name='Current Position',
            line=dict(color='blue', width=3, dash='dash'),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Update layout
    title = f'Video with {blendshape_name} Synchronization'
    if face_crop:
        title += f' (Face Cropped + 8x8 Downsampled)'
    
    fig.update_layout(
        title=title + f'<br><sub>Video: {min_frames} frames | Blendshapes: {min_frames} frames</sub>',
        height=600,  # Reduced height by 2x
        width=600,   # Reduced width by 2x
        showlegend=True,
        xaxis2_title='Frame Number',
        yaxis2_title=f'{blendshape_name} Value',
        xaxis_title='',
        yaxis_title=''
    )
    
    # Update axes
    fig.update_xaxes(range=[0, min_frames-1], row=2, col=1)
    fig.update_yaxes(range=[blendshape_values.min(), blendshape_values.max()], row=2, col=1)
    
    # Add slider - use video frames for navigation
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': -0.15,  # Position slider higher (was -0.25)
            'yanchor': 'bottom',
            'steps': [
                {
                    'args': [[i], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(i),
                    'method': 'animate'
                } for i in range(0, min_frames, max(1, min_frames//50))
            ]
        }]
    )
    
    # Add play/pause button
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0.1,
            'y': 0.95,
            'xanchor': 'left',
            'yanchor': 'top',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }]
    )
    
    # Create frames for animation
    frames = []
    for frame_idx in range(0, min_frames, max(1, min_frames//100)):
        # Read video frame
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop to face region if enabled
            if face_crop:
                frame_cropped = frame_rgb[y_start:y_end, x_start:x_end, :]
                # Remove 10 pixels from each side
                frame_cropped = frame_cropped[10:-10, 10:-10, :]
                # Additional downsampling by 8x8 for performance
                frame_cropped = frame_cropped[::8, ::8, :]
            else:
                frame_cropped = frame_rgb
            
            frame_data = [
                # Video frame
                go.Image(z=frame_cropped),
                # Blendshape line
                go.Scatter(
                    x=frame_numbers,
                    y=blendshape_values,
                    mode='lines',
                    name=blendshape_name,
                    line=dict(color='red', width=2)
                ),
                # Current position indicator
                go.Scatter(
                    x=[frame_idx, frame_idx],
                    y=[blendshape_values.min(), blendshape_values.max()],
                    mode='lines',
                    name='Current Position',
                    line=dict(color='blue', width=3, dash='dash')
                )
            ]
        else:
            # Fallback if frame reading fails
            frame_data = [
                go.Image(z=np.zeros((height, width, 3), dtype=np.uint8)),
                go.Scatter(x=frame_numbers, y=blendshape_values, mode='lines', 
                         name=blendshape_name, line=dict(color='red', width=2)),
                go.Scatter(x=[frame_idx, frame_idx], y=[blendshape_values.min(), blendshape_values.max()], 
                         mode='lines', name='Current Position', line=dict(color='blue', width=3, dash='dash'))
            ]
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx)
        ))
    
    fig.frames = frames
    
    return fig

BLENDSHAPES_ORDERED = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
                       'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                       'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
                       'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
                       'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']


parent_path = Path("/mnt/A3000/Recordings/v2_data")

runs = [
    "2025/05/13/QaQa-143122/25_0_31bad7f4-e6da-4364-b809-791dee129162_loud",
    "2025/05/13/QaQa-143122/24_0_418ee7e0-6be5-43f9-acde-6f31a180c63c_silent",
    "2025/05/13/QaQa-143122/21_1_b5319549-195a-4ac2-ab3f-4c0690f76260_loud",
    "2025/05/13/QaQa-143122/26_0_31bad7f4-e6da-4364-b809-791dee129162_silent",
    "2025/05/13/QaQa-143122/21_0_b5319549-195a-4ac2-ab3f-4c0690f76260_loud",
    "2025/05/13/QaQa-143122/26_1_31bad7f4-e6da-4364-b809-791dee129162_silent",
    "2025/05/13/QaQa-143122/28_0_2b55a1a4-6508-4bbe-862f-cd9541522cc0_silent",]

# df = pd.read_pickle("/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250402_split_1/LOUD_GIP_general_clean_250415_v2.pkl")
# idx_lst = random.sample(range(0, len(df)+1), 5) #[1004, 2877,1744, 3663, 556] #random.sample(range(0, len(df)+1), 50)
for i in tqdm(range(len(runs))):
    data_path = parent_path / runs[i]
    video = cv2.VideoCapture(str(data_path/ "video_full.mp4"))
    landmarks_and_blendshapes = np.load(data_path/ "landmarks_and_blendshapes.npz")
    blendshapes = landmarks_and_blendshapes['blendshapes']

    # Create the video plot with slider and blendshape plot below (face cropped for performance)
    # Find the index of eyeBlinkRight blendshape
    blendshape_idx = BLENDSHAPES_ORDERED.index('eyeBlinkRight')
    fig_video = create_video_slider_plot(
        data_path / "video_full.mp4", 
        blendshapes, 
        blendshape_idx, 
        blendshape_name='eyeBlinkRight',
        face_crop=True, 
        crop_margin=80
    )
    
    # Update the main title to include the sample index
    fig_video.update_layout(title=f"Sample {i}")
    
    fig_video.show()

