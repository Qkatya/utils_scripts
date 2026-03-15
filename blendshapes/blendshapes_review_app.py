"""
Dash app for blendshapes data quality review: video samples + instruction-based blendshape histograms.
Run after sample_review_100.py to generate review_100/manifest.pkl and review_100/videos/.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from flask import send_file

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from instruction_blendshapes_mapping import (
    get_group_for_read_text,
    GROUP_TO_BLENDSHAPES,
    get_blendshape_indices,
    ALL_BLENDSHAPES_NAMES,
)

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
REVIEW_DIR = SCRIPT_DIR / "review_100"
MANIFEST_PATH = REVIEW_DIR / "manifest.pkl"
VIDEOS_DIR = REVIEW_DIR / "videos"
SPLIT_PKL = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_no_beep.pkl")
BASE_DATA_PATH = Path("/mnt/A3000/Recordings/v2_data")
EXCLUDED_READ_TEXTS = {
    "we're almost done - just two more sessions to go!\n\nnow we'd like you to perform an expression at three intensities: small, medium, and big",
    "great job so far!\n\nin the next session, we'll ask you to make repeated eye movements.\nplease follow the instructions on screen.",
    "in this session, you will make different facial expressions.\nplease follow the instructions on the screen.",
    "last session ahead.\nplease say each of the following texts out loud, in your normal voice.\nplease follow the instructions on the screen.",
    "thank you for helping us with this project. you're awesome.",
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Load manifest
if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}. Run sample_review_100.py first.")
manifest_df = pd.read_pickle(MANIFEST_PATH)
N_SAMPLES = len(manifest_df)


@app.server.route("/video/<int:idx>")
def serve_video(idx):
    if idx < 0 or idx >= N_SAMPLES:
        return "", 404
    path = VIDEOS_DIR / f"sample_{idx:03d}.mp4"
    if not path.exists():
        return "", 404
    return send_file(path, mimetype="video/mp4", as_attachment=False)


def video_tab():
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("Sample text", id="sample-title"),
                html.P(id="sample-counter", children="Sample 1 of N"),
                html.Div(
                    [
                        html.Video(id="review-video", src="/video/0", controls=True, style={"maxWidth": "100%"}),
                    ],
                    id="video-container",
                ),
                html.Div(
                    [
                        dbc.Button("Previous", id="btn-prev", color="primary", className="me-2"),
                        dbc.Button("Next", id="btn-next", color="primary"),
                    ],
                    className="mt-3",
                ),
                dcc.Store(id="current-index", data=0),
            ]
        ),
        className="mt-2",
    )


def stats_tab():
    group_options = [{"label": g, "value": g} for g in sorted(GROUP_TO_BLENDSHAPES.keys())]
    group_options.insert(0, {"label": "All blendshapes (overview)", "value": "__all__"})
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("Blendshape histograms by instruction"),
                html.P("Select an instruction group to see value distributions for its relevant blendshapes."),
                dbc.Label("Instruction group"),
                dcc.Dropdown(id="stats-group-dropdown", options=group_options, value="expression: smile"),
                dcc.Loading(dcc.Graph(id="stats-histograms"), type="default"),
                dcc.Store(id="stats-split-cache", data=None),
            ]
        ),
        className="mt-2",
    )


app.layout = dbc.Container(
    [
        html.H2("Blendshapes data quality review", className="mb-4"),
        dbc.Tabs(
            [
                dbc.Tab(video_tab(), label="Video review"),
                dbc.Tab(stats_tab(), label="Blendshape stats"),
            ],
            id="main-tabs",
        ),
    ],
    fluid=True,
    className="p-4",
)


@callback(
    [Output("current-index", "data"), Output("review-video", "src"), Output("sample-title", "children"), Output("sample-counter", "children")],
    [Input("btn-prev", "n_clicks"), Input("btn-next", "n_clicks")],
    State("current-index", "data"),
    prevent_initial_call=False,
)
def update_video(prev_clicks, next_clicks, current_index):
    ctx = dash.ctx
    if not ctx.triggered_id:
        idx = 0
    else:
        idx = current_index or 0
        if ctx.triggered_id == "btn-next":
            idx = min(idx + 1, N_SAMPLES - 1)
        elif ctx.triggered_id == "btn-prev":
            idx = max(idx - 1, 0)
    row = manifest_df.iloc[idx]
    title = (row["read_text"] or "")[:500]
    if len(str(row["read_text"] or "")) > 500:
        title += "..."
    return idx, f"/video/{idx}", title, f"Sample {idx + 1} of {N_SAMPLES}"


def load_filtered_split():
    df = pd.read_pickle(SPLIT_PKL)
    text_col = "read_text" if "read_text" in df.columns else "text"
    filtered = df[~df[text_col].astype(str).str.strip().isin(EXCLUDED_READ_TEXTS)].copy()
    return filtered


def get_blendshape_data_for_group(group_label: str, max_runs: int = 500):
    """Load npz for runs in this instruction group and return stacked blendshapes (T, 52) and indices for relevant blends.
    If group_label == '__all__', load random runs and return all 52 columns."""
    df = load_filtered_split()
    text_col = "read_text" if "read_text" in df.columns else "text"
    if group_label == "__all__":
        run_paths = df.drop_duplicates(subset=["run_path"])["run_path"].tolist()
        if len(run_paths) > max_runs:
            import random
            random.seed(42)
            run_paths = random.sample(run_paths, max_runs)
        blendshape_names = ALL_BLENDSHAPES_NAMES
        indices = list(range(52))
    else:
        df["_group"] = df[text_col].astype(str).str.strip().map(get_group_for_read_text)
        group_runs = df[df["_group"] == group_label].drop_duplicates(subset=["run_path"], keep="first")
        run_paths = group_runs["run_path"].tolist()
        if len(run_paths) > max_runs:
            import random
            random.seed(42)
            run_paths = random.sample(run_paths, max_runs)
        blendshape_names = GROUP_TO_BLENDSHAPES.get(group_label, [])
        indices = get_blendshape_indices(blendshape_names)
        if not indices:
            return None, []
    all_values = []
    for rp in run_paths:
        path = BASE_DATA_PATH / str(rp) / "landmarks_and_blendshapes.npz"
        if not path.exists():
            continue
        try:
            data = np.load(path)
            bs = data["blendshapes"]
            if bs.ndim == 2 and bs.shape[1] >= max(indices) + 1:
                all_values.append(bs[:, indices])
        except Exception:
            continue
    if not all_values:
        return None, blendshape_names
    stacked = np.vstack(all_values)
    return stacked, blendshape_names


@callback(
    Output("stats-histograms", "figure"),
    Input("stats-group-dropdown", "value"),
)
def update_histograms(group_value):
    if not group_value:
        return go.Figure().add_annotation(text="Select an instruction group", showarrow=False)
    stacked, names = get_blendshape_data_for_group(group_value)
    if stacked is None or not names:
        return go.Figure().add_annotation(text="No data for this group", showarrow=False)
    n = len(names)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, vertical_spacing=0.08, horizontal_spacing=0.06)
    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        fig.add_trace(go.Histogram(x=stacked[:, i], nbinsx=80, name=name, showlegend=False), row=r + 1, col=c + 1)
    title = "All blendshapes (overview)" if group_value == "__all__" else f"Blendshape values: {group_value}"
    fig.update_layout(title_text=title, height=max(400, 180 * rows))
    return fig


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
