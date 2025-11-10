import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.signal
from plotly.subplots import make_subplots

from blosc2_helpers.blosc2_utils import load_frames_blosc2
from init_config import init_config
from utils.q_features_utils import calc_contrast_at_point, calc_median_contrast, optical_flow_tracking_multiscale_on_grid
from utils.video_utils import crop_frames_to_percentiles, read_video_frames
from scipy.signal import find_peaks


def plot_labels_and_contrast_and_webcam(row: pd.Series, base_raw_data_dir: Path, config, frames_per_sample: int = 4, flip_to: str = "left"):
    # Load all frames
    tar_path = base_raw_data_dir / row.tar_path
    frames = load_frames_blosc2(tar_path)
    tar_fps = 200

    # Flip if needed
    tar_side = Path(row.tar_path).name.split(".")[-2]
    if tar_side != flip_to:
        frames = np.flip(frames, axis=-2)  # Flip horizontally
    frames = frames.astype(np.float32)

    # Load all labels
    match config.data.label_type:
        case "ema":
            label_names = config.data.ema.all_names
            label_dir = Path("/mnt/ML/Development/shaked.dovrat/EMA")
            labels = np.load(label_dir / row.ema_features_path)  # Showing all EMA features.
        case "blendshapes":
            label_names = config.data.used_blendshapes_names
            label_dir = Path("/mnt/ML/Development/shaked.dovrat/blendshapes_50hz")
            labels = np.load(label_dir / row.blendshapes_path)
            labels = labels[:, config.data.used_label_idxs]  # Not showing all Blendshapes because there are too many.
            labels = np.nan_to_num(labels, nan=0.0)
        case _:
            raise ValueError(f"Invalid label type: {config.data.label_type}")

    label_times = np.arange(len(labels)) * frames_per_sample
    d_labels = np.diff(labels, axis=0)
    d_labels_zscore = (d_labels - d_labels.mean(axis=0)) / d_labels.std(axis=0)

    # Load webcam frames
    base_data_dir = Path("/mnt/A3000/Recordings/v2_data")
    webcam_path = base_data_dir / row.run_path / "webcam_video.avi"
    webcam_frames, webcam_fps = read_video_frames(webcam_path)
    webcam_frames = crop_frames_to_percentiles(webcam_frames, x_percentiles=(25, 75), y_percentiles=(20, 80))
    webcam_frames = webcam_frames[..., ::-1]  # BGR to RGB

    webcam_frames = np.array([cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)) for frame in webcam_frames])

    # go.Figure(go.Heatmap(z=frames[0])).show()  # Show first frame

    # Calculate optical flow features
    of_features, of_cell_centers_xy = optical_flow_tracking_multiscale_on_grid(frames)

    crop_ltrb = (246, 108, 500, 272)
    mask = (
        (of_cell_centers_xy[:, 0] > crop_ltrb[0])
        & (of_cell_centers_xy[:, 0] < crop_ltrb[2])
        & (of_cell_centers_xy[:, 1] > crop_ltrb[1])
        & (of_cell_centers_xy[:, 1] < crop_ltrb[3])
    )

    of_features = of_features[:, mask]
    # of_features_zscore = (of_features - of_features.mean(axis=(1, 2), keepdims=True)) / of_features.std(axis=(1, 2), keepdims=True)
    # of_features_smoothed = scipy.signal.medfilt(of_features_zscore, kernel_size=5)
    dx_median = np.median(of_features[:, :, 0], axis=1)
    dy_median = np.median(of_features[:, :, 1], axis=1)
    dx_zscore = (dx_median - dx_median.mean()) / dx_median.std()
    dy_zscore = (dy_median - dy_median.mean()) / dy_median.std()
    dx_smoothed = scipy.signal.medfilt(dx_zscore, kernel_size=5)
    dy_smoothed = scipy.signal.medfilt(dy_zscore, kernel_size=5)

    # Calculate contrast

    # pt_uv = (470, 110)
    # contrast = calc_contrast_at_point(frames, pt_uv)
    contrast = calc_median_contrast(frames)
    contrast_zscore = (contrast - contrast.mean()) / contrast.std()
    contrast_smoothed = scipy.signal.medfilt(contrast_zscore, kernel_size=5)

    frame_times = np.arange(len(frames))

    # --- Plotly Subplots ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.4, 0.6], subplot_titles=("Webcam Video", "Labels and Contrast")
    )

    # Initial webcam frame (first frame)
    webcam_img = webcam_frames[0]
    fig.add_trace(go.Image(z=webcam_img, name="Webcam Frame"), row=1, col=1)

    # Add label traces (bottom plot)
    for i in range(labels.shape[1]):
        name = label_names[i] if i < len(label_names) else f"Label {i}"
        fig.add_trace(go.Scatter(x=label_times, y=labels[:, i], mode="lines", name=name, yaxis="y1"), row=2, col=1)
    for i in range(labels.shape[1]):
        name = f"d{label_names[i]}" if i < len(label_names) else f"dLabel {i}"
        fig.add_trace(go.Scatter(x=label_times[:-1], y=d_labels_zscore[:, i], mode="lines", name=name, yaxis="y1"), row=2, col=1)

    # Add contrast trace (bottom plot)
    fig.add_trace(go.Scatter(x=frame_times, y=contrast_zscore, mode="lines", name="Contrast", yaxis="y2", line=dict(color="black", width=1)), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=frame_times, y=contrast_smoothed, mode="lines", name="Contrast Smoothed", yaxis="y2", line=dict(color="red", width=1)), row=2, col=1
    )
    fig.add_trace(go.Scatter(x=frame_times, y=dx_zscore, mode="lines", name="dx", yaxis="y2", line=dict(color="blue", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame_times, y=dy_zscore, mode="lines", name="dy", yaxis="y2", line=dict(color="green", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame_times, y=dx_smoothed, mode="lines", name="dx smoothed", yaxis="y2", line=dict(color="blue", width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame_times, y=dy_smoothed, mode="lines", name="dy smoothed", yaxis="y2", line=dict(color="green", width=1)), row=2, col=1)

    # Add vertical line (initially at frame 0, mapped to original frame index)
    vline = dict(
        type="line",
        x0=0,
        x1=0,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2",  # restrict to bottom subplot only
        line=dict(color="red", width=2, dash="dash"),
        layer="above",
    )
    fig.add_shape(vline)

    # --- Animation frames for webcam and vline ---
    plot_frames = []
    for i in range(len(webcam_frames)):
        webcam_img = webcam_frames[i]
        # Calculate the corresponding original frame index for this webcam frame
        webcam_time = i / webcam_fps
        corresponding_frame_idx = int(webcam_time * tar_fps)
        # Move vertical line to correct original frame index
        vline_frame = dict(
            type="line",
            x0=corresponding_frame_idx,
            x1=corresponding_frame_idx,
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            layer="above",
        )
        plot_frames.append(go.Frame(data=[go.Image(z=webcam_img)], name=str(i), layout=dict(shapes=[vline_frame])))

    # --- Slider and Play/Pause ---
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[[str(i)], dict(mode="immediate", frame=dict(duration=1000 / webcam_fps, redraw=True), transition=dict(duration=0))],
                    label=str(i),
                )
                for i in range(len(webcam_frames))
            ],
            transition=dict(duration=0),
            x=0.1,
            y=0,
            len=0.8,
        )
    ]
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=1.05,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(
                    label="Play", method="animate", args=[None, dict(frame=dict(duration=1000 / webcam_fps, redraw=True), fromcurrent=True, mode="immediate")]
                ),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )
    ]

    # --- Layout ---
    fig.update_layout(
        title=f"Labels and Contrast for {row.tar_path}",
        width=1200,
        height=800,
        margin=dict(l=60, r=60, t=60, b=60),
        sliders=sliders,
        updatemenus=updatemenus,
        showlegend=True,
        xaxis2=dict(title="Frame Index"),
        yaxis=dict(title="Webcam Frame"),
        yaxis2=dict(title="Label Value"),
        yaxis3=dict(title="Contrast (placeholder)", overlaying="y2", side="right"),
    )
    fig.frames = plot_frames
    fig.show()


def main():
    parser = argparse.ArgumentParser(description="Plot labels and contrast for a tar from a DataFrame row.")
    parser.add_argument("--df_path", type=str, required=False, help="Path to the DataFrame pickle file.")
    # parser.add_argument("--row_idx", type=int, default=0, help="Index of the row to visualize.")
    parser.add_argument("--base_raw_data_dir", type=str, default="/mnt/ML/RawData", help="Base directory for raw data.")
    parser.add_argument("--frames_per_sample", type=int, default=4, help="Number of frames per label sample.")
    parser.add_argument("--flip_to", type=str, default="left", help='Which side to flip frames to ("left" or "right")')
    parser.add_argument("--test", action="store_true", help="Run with hard-coded test arguments.")
    args = parser.parse_args()

    if args.test:
        args.df_path = (
            "/mnt/ML/Development/shaked.dovrat/dfs/tars_with_EMA_and_blosc2_and_blendshapes50hz_from_20250220_split_1_LOUD_157964train_16212valid.pkl"
        )
        # args.df_path = "/mnt/ML/Development/shaked.dovrat/dfs/tars_with_EMA_and_blosc2_from_20250220_split_1_LOUD_train_22039samples.pkl"
        args.row_idx = 0
    elif args.df_path is None:
        parser.error("--df_path is required unless --test is specified.")

    config = init_config()

    df = pd.read_pickle(args.df_path)

    # Row index by length:
    l = np.percentile(df.ema_length, 90)
    df = df[df.ema_length > l]
    df = df.sample(1, random_state=42)
    # args.row_idx = next(i for i in range(len(df)) if df.ema_length.iloc[i] > l)

    row = df.iloc[args.row_idx]
    plot_labels_and_contrast_and_webcam(row, Path(args.base_raw_data_dir), config, args.frames_per_sample, args.flip_to)

    # TODO: BAD UNSYNCED DATA WITH random_state=42.


if __name__ == "__main__":
    main()