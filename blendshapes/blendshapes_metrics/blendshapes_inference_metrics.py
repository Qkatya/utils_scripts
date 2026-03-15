
import os
import sys
import random
import pickle
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore
import nemo.collections.asr as nemo_asr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

fairseq_path = "/home/katya.ivantsiv/q-fairseq-train_bs_loud-blendshapes_loud-8878cd07-20250718_212401"
sys.path.insert(0, fairseq_path)
sys.path.insert(0, f"{fairseq_path}/examples/data2vec")
from fairseq import checkpoint_utils, utils
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

pio.renderers.default = "browser"

# ============================================================================
# CONSTANTS
# ============================================================================
BLENDSHAPES_ORDERED = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight',
                       'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 
                       'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
                       'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
                       'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

BLENDSHAPE_COLORS = {'eyeBlinkRight': '#e377c2', 'jawOpen': '#ff7f0e', 'mouthFunnel': '#2ca02c', 'cheekPuff': '#d62728', 'mouthSmileLeft': '#9467bd', 'mouthFrownLeft': '#8c564b'}
PLOT_COLORS = ['#ff7f0e', '#2ca02c', '#d62728']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def unnormalize_fastconformer_blendshapes(blendshape_labels, blendshapes_idx):
    boundaries_lst = [(1.2090332290881634e-08, 3.749652239548596e-06), (2.445738971346145e-07, 0.2749528124928477), (1.978288395321215e-07, 0.2759505271911622), (1.0057362942461623e-06, 0.7707802176475524), (7.059926429064944e-05, 0.7855742245912554), (4.5511307689594105e-05, 0.6722787052392961), (7.704340418968059e-07, 0.00024206499801948667), (4.007033815867089e-09, 1.7510708346435425e-06), (5.4851407860212475e-09, 7.810492093085486e-07), (0.0013135804329067469, 0.6775347143411636), (0.0006478337454609573, 0.6262029528617858), (3.462312452029437e-05, 0.8312738806009295), (0.000131698208861053, 0.826391476392746), (2.216839675384108e-05, 0.2110402494668961), (2.0124221919104457e-05, 0.3471406474709512), (8.35571256629919e-08, 0.350243017077446), (1.0900915185629856e-05, 0.20838836356997492), (8.824750693747774e-06, 0.09804979339241984), (1.4880988601362333e-06, 0.0945791814476252), (0.007547934073954821, 0.5727719873189926), (0.0041250200010836124, 0.5164364755153656), (0.00012323759438004345, 0.011711244843900223), (2.376515476498753e-05, 0.01267738505266607), (4.543169325188501e-07, 0.0015177460212726151), (3.409600140003022e-06, 0.018415350466966636), (3.4944002891279524e-06, 0.2855767607688904), (1.3396369524798502e-07, 0.0006583309383131573), (6.175905582495034e-07, 0.031902600452303885), (1.264479124074569e-05, 0.050973990932107), (2.231232610938605e-05, 0.05713949967175725), (1.1405156818966589e-09, 0.021382492315024138), (2.0549280055348618e-09, 0.024682952091097846), (3.3339190395054175e-06, 0.14517886489629747), (1.7259408124914444e-08, 0.01276048296131194), (1.8087397393173887e-06, 0.08748787157237532), (3.655995328699646e-07, 0.10239365324378022), (2.1377220036811195e-05, 0.22725961357355118), (3.9081238355720416e-05, 0.25498466938734066), (3.082554712818819e-06, 0.6337731003761291), (5.294374361142218e-08, 0.00830749128945172), (1.74538217834197e-05, 0.14103781953454025), (4.109945621166844e-06, 0.1965724654495717), (1.3196885220168042e-06, 0.20960539430379874), (1.560291821078863e-05, 0.12181339934468284), (1.7525019657682606e-08, 0.26781445741653453), (9.214115692657288e-09, 0.2741092413663864), (3.705958562250089e-08, 0.05159267019480475), (1.8201465934453154e-07, 0.07845675386488438), (1.978705341798559e-07, 0.21748517602682124), (1.297534311106574e-07, 0.271655547618866), (1.9878497070635603e-08, 1.1203549274796392e-05), (3.0873905654260625e-09, 3.264578106154661e-06)]
    boundaries = np.stack(boundaries_lst)[blendshapes_idx, :]
    unnormalized_labels = blendshape_labels.copy()
    unnormalized_labels[:, blendshapes_idx] = (unnormalized_labels[:, blendshapes_idx]) * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
    return unnormalized_labels

def zeropad_blendshapes_to_52_length(blendshapes, blendshapes_idx):
    blendshapes_52 = np.zeros((blendshapes.shape[0], 52))
    blendshapes_52[:, blendshapes_idx] = blendshapes
    return blendshapes_52 

def velocity_agreement(gt, pred, blendshape_name):
    """Calculate velocity agreement metrics between ground truth and prediction."""
    idx = BLENDSHAPES_ORDERED.index(blendshape_name)
    # # plot the gt and pred blendshapes
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(gt[:, idx])), y=gt[:, idx], name='GT Blendshapes'))
    # fig.add_trace(go.Scatter(x=np.arange(len(pred[:, idx])), y=pred[:, idx], name='Pred Blendshapes'))
    # fig.show()
    gt, pred = savgol_filter(gt[:, idx], 9, 2, mode='interp'), savgol_filter(pred[:, idx], 9, 2, mode='interp')
    da, db = np.diff(gt), np.diff(pred)
    # # plot gt and pred
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(gt)), y=gt, name='GT Blendshapes'))
    # fig.add_trace(go.Scatter(x=np.arange(len(pred)), y=pred, name='Pred Blendshapes'))
    # fig.show()
    # # plot da and db
    # fig = make_subplots(rows=1, cols=1)
    # fig.add_trace(go.Scatter(x=np.arange(len(da)), y=da, name='DA'))
    # fig.add_trace(go.Scatter(x=np.arange(len(db)), y=db, name='DB'))
    # fig.show()
    sign_match = np.mean(np.sign(da)==np.sign(db))
    r = np.corrcoef(da, db)[0,1]
    return sign_match, r 


def blinking_counter(blendshape_values, blink_th=2):
    """Count blinks using peak detection on z-scored values."""
    z_scores = zscore(blendshape_values)
    peaks, _ = find_peaks(z_scores, height=blink_th, prominence=1.5, distance=5, width=(None, 20))
    return 0 if np.mean(blendshape_values > 0.44) > 0.5 else len(peaks)

def load_fairseq_model(model_path, device, use_fp16):
    """Load and prepare a fairseq model."""
    print(f"Loading model from {model_path}")
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='', strict=False)
    model = models[0].eval()
    if use_fp16: model.half()
    model.to(device)
    print(f"Loaded model")
    return model, saved_cfg, task

def load_nemo_model(model_path, device, use_fp16):
    """Load and prepare a NeMo model."""
    print(f"Loading model from {model_path}")
    model = nemo_asr.models.EncDecCTCModel.restore_from(model_path).eval()
    if use_fp16: model.half()
    model.to(device)
    print(f"Loaded model")
    return model

# ============================================================================
# INFERENCE HELPER FUNCTIONS
# ============================================================================
def prepare_sample(data, device, st, en):
    """Prepare input sample for model inference."""
    stack = np.array(data[st:en])
    sample = torch.from_numpy(stack).half().float().unsqueeze(0).to(device).half()
    padding_mask = torch.Tensor([False] * sample.shape[1]).unsqueeze(0).to(device)
    return sample, padding_mask

def load_gt_blendshapes(row, gt_parent_path):
    """Load and downsample ground truth blendshapes."""
    gt_data_path = gt_parent_path / row['run_path']
    gt_blendshapes = np.load(gt_data_path / "landmarks_and_blendshapes.npz")['blendshapes']
    mask = np.ones(gt_blendshapes.shape[0], dtype=bool)
    mask[5::6] = False
    return gt_blendshapes[mask]

def infer_fairseq_model(model, sample, padding_mask, blendshapes_idx):
    with torch.inference_mode():
        net_output = model(**{"source": sample, "padding_mask": padding_mask})
        blendshapes = net_output["encoder_blends"].cpu().numpy().squeeze().transpose(1, 0)
        blendshapes = zeropad_blendshapes_to_52_length(blendshapes, blendshapes_idx)
    return blendshapes
      
def infer_nemo_model(model, data, device, blendshapes_idx):
    with torch.no_grad():
        processed_signal, processed_signal_length = model.preprocessor(input_signal=torch.tensor(data).unsqueeze(0).to(device).half(), length=torch.tensor(data.shape[0]).unsqueeze(0).to(device))
        encoder_output = model.encoder(audio_signal=processed_signal, length=processed_signal_length)
        output_blendshapes = model.decoder.blendshapes_head(encoder_output[0]).cpu().numpy().squeeze().transpose(1, 0)
        output_blendshapes = zeropad_blendshapes_to_52_length(output_blendshapes[:, :len(blendshapes_idx)], blendshapes_idx)
    return output_blendshapes

def normalize_blendshapes(blendshapes, model_type, blendshapes_idx, normalization_factors=None):
    blendshapes_unnorm = blendshapes.copy()
    
    if model_type == 'fairseq':
        # Fairseq models output normalized values, need to unnormalize using mean/std
        if normalization_factors is None:
            raise ValueError("normalization_factors required for fairseq models")
        std = normalization_factors["Std"].values[np.array(blendshapes_idx)]
        mean = normalization_factors["Mean"].values[np.array(blendshapes_idx)]
        blendshapes_unnorm[:, blendshapes_idx] = (blendshapes[:, blendshapes_idx] * std) + mean
        
    elif model_type == 'nemo':
        # NeMo models output normalized values, need to unnormalize using min/max boundaries
        blendshapes_unnorm = unnormalize_fastconformer_blendshapes(blendshapes, blendshapes_idx)
    
    return blendshapes_unnorm

def compute_metrics(gt, preds, blendshape_names):
    """Compute velocity agreement, RMSE, and blink metrics for all models."""
    results = {}
    for bs_name in blendshape_names:
        # print(bs_name)
        idx = BLENDSHAPES_ORDERED.index(bs_name)
        for i, pred in enumerate(preds, 1):
            sign_match, r = velocity_agreement(gt, pred, bs_name)
            results[f'sign_match_{bs_name}_model{i}'] = sign_match
            results[f'r_{bs_name}_model{i}'] = r
            # Calculate RMSE for this blendshape
            rmse = np.sqrt(np.mean((gt[:, idx] - pred[:, idx]) ** 2))/np.mean(gt[:, idx])
            # print(rmse)
            results[f'rmse_{bs_name}_model{i}'] = rmse
    
    eyeBlinkRight_idx = BLENDSHAPES_ORDERED.index("eyeBlinkRight")
    results['blink_counter_gt'] = blinking_counter(gt[:, eyeBlinkRight_idx], blink_th=2)
    for i, pred in enumerate(preds, 1):
        results[f'blink_counter_model{i}'] = blinking_counter(pred[:, eyeBlinkRight_idx-1], blink_th=1.8)
    return results

def plot_blendshape_comparison(gt, predictions, model_infos, use_diff=True):
    """Plot GT vs predicted blendshapes for visual comparison."""
    blendshapes_to_plot = ['eyeBlinkRight', 'jawOpen', 'mouthFunnel', 'cheekPuff', 'mouthFrownLeft']
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=tuple(f"<span style='font-size:20px'>{t}</span>" for t in blendshapes_to_plot), vertical_spacing=0.05)
    
    def add_traces(bs_name, row_num, use_diff=False, show_legends=None):
        if show_legends is None:
            show_legends = [False] * (len(predictions) + 1)
        
        idx = BLENDSHAPES_ORDERED.index(bs_name)
        gt_data = gt[:, idx]
        pred_data = [pred[:, idx] for pred in predictions]
        
        if use_diff:
            gt_data = np.diff(savgol_filter(gt_data, 9, 2, mode='interp'))
            pred_data = [np.diff(savgol_filter(x, 9, 2, mode='interp')) for x in pred_data]
        
        # Add GT trace
        fig.add_trace(go.Scatter(x=np.arange(len(gt_data)), y=gt_data, name='GT Blendshapes' if show_legends[0] else None, 
                                mode='lines', line=dict(color='blue', width=3), showlegend=show_legends[0]), row=row_num, col=1)
        
        # Add prediction traces
        for i, (pred, model_info) in enumerate(zip(pred_data, model_infos), 1):
            config = model_info['config']
            line_dict = dict(color=BLENDSHAPE_COLORS[bs_name], width=3)
            if config['line_style']:
                line_dict['dash'] = config['line_style']
            
            fig.add_trace(go.Scatter(x=np.arange(len(pred)), y=pred, 
                                    name=config['display_name'] if show_legends[i] else None, 
                                    mode='lines', line=line_dict, showlegend=show_legends[i]), row=row_num, col=1)
    
    # First row shows all legends
    show_legends_first = [True] * (len(predictions) + 1)
    add_traces('eyeBlinkRight', 1, use_diff=False, show_legends=show_legends_first)
    
    # Subsequent rows show only prediction legends (not GT)
    show_legends_rest = [False] + [True] * len(predictions)
    for i, bs_name in enumerate(blendshapes_to_plot[1:], 2):
        add_traces(bs_name, i, use_diff=use_diff, show_legends=show_legends_rest)
    
    fig.update_layout(title_text="Gt vs pred blendshapes", title_x=0.47, title_xanchor='center', title_font_size=30, legend=dict(font=dict(size=16)))
    fig.show()
    
# ============================================================================
# MODEL & DATA LOADING
# ============================================================================
class MyDefaultObj:
    def __init__(self):
        self.user_dir = f"{fairseq_path}/examples/data2vec"

utils.import_user_module(MyDefaultObj())

# Configuration
device = 'cuda:0'
use_fp16 = True
st, en = 0, None

# Paths
gt_parent_path = Path("/mnt/A3000/Recordings/v2_data")
features_path = Path('/mnt/ML/Production/ML_Processed_Data/Q_Features/v2_200fps_energy_std_sobel_stcorr/features')

# Model configurations: (path, type, display_name, blendshape_indices, line_style)
# type: 'fairseq' or 'nemo'
# blendshape_indices: indices to use for this model
# line_style: None (solid), 'dash', 'dot', 'dashdot'
MODEL_CONFIGS = [
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/blendshapes/blendshapes_loud/0/checkpoints/checkpoint_last.pt',
    #     'type': 'fairseq',
    #     'name': 'blendshapes_loud',
    #     'display_name': 'D2V trained on 400k samples',
    #     'blendshape_indices': list(range(1, 52)),
    #     'line_style': None
    # },
    {
        'path': '/mnt/ML/TrainResults/ido.kazma/D2V/V2/2025_04_15/new21_baseline_blendshapes_normalized/0/checkpoints/checkpoint_last.pt',
        'type': 'fairseq',
        'name': 'new21_baseline_blendshapes_normalized',
        'display_name': 'D2V trained on 2.5M samples',
        'blendshape_indices': list(range(1, 52)),
        'line_style': 'dashdot'
    },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only/checkpoints/causal_fastconformer_layernorm_landmarks_blendshapes_heads_only.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_layernorm_landmarks_blendshapes_heads_only',
    #     'display_name': 'NeMo FastConformer (Causal)',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dash'
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/quartznet_landmarks_blendshapes/checkpoints/quartznet_landmarks_blendshapes.nemo',
    #     'type': 'nemo',
    #     'name': 'quartznet_landmarks_blendshapes',
    #     'display_name': 'NeMo QuartzNet',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dot'
    # },
    # {
    #     'path': '/home/katya.ivantsiv/blendshapes_models/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides',
    #     'display_name': 'NeMo FastConformer (Causal) - All Blendshapes, partly trained',
    #     'blendshape_indices': list(range(1, 52)),
    #     'line_style': None
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides/checkpoints/causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides.nemo',
    #     'type': 'nemo',
    #     'name': 'causal_fastconformer_layernorm_landmarks_all_blendshapes_two_sides',
    #     'display_name': 'NeMo FastConformer (Causal) - All Blendshapes',
    #     'blendshape_indices': list(range(1, 52)),
    #     'line_style': None
    # },
    # {
    #     'path': '/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/fastconformer_blendshapes_landmarks/checkpoints/fastconformer_blendshapes_landmarks.nemo',
    #     'type': 'nemo',
    #     'name': 'fastconformer_blendshapes_landmarks',
    #     'display_name': 'NeMo FastConformer',
    #     'blendshape_indices': [6, 8, 10, 14, 16, 25, 26, 27, 29, 32, 38],
    #     'line_style': 'dashdot'
    # },
]

# Load all models
models = []
for config in MODEL_CONFIGS:
    print(f"\nLoading {config['name']}...")
    if config['type'] == 'fairseq':
        model, saved_cfg, task = load_fairseq_model(config['path'], device, use_fp16)
        models.append({'model': model, 'config': config, 'saved_cfg': saved_cfg, 'task': task})
    elif config['type'] == 'nemo':
        model = load_nemo_model(config['path'], device, use_fp16)
        models.append({'model': model, 'config': config})
    else:
        raise ValueError(f"Unknown model type: {config['type']}")

# Load dataframe
df_path = '/mnt/ML/Development/ML_Data_DB/v2/splits/full/20250402_split_1/LOUD_GIP_general_clean_250415_v2.pkl'
print(f"Loading dataframe from {df_path}")
df = pd.read_pickle(df_path)
print(f"Loaded dataframe with {len(df)} rows")

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Load normalization factors (for fairseq models)
blendshapes_normalize_path = '/home/ido.kazma/projects/notebooks-qfairseq/stats_df.pkl'
with open(blendshapes_normalize_path, 'rb') as f:
    blendshape_normalization_factors = pickle.load(f)

# Initialize metric storage dynamically based on number of models
num_models = len(MODEL_CONFIGS)
metrics = {f'{metric}_{bs}_model{m}': [] for metric in ['sign_match', 'r', 'rmse'] for bs in ['jawOpen', 'mouthFunnel', 'cheekPuff'] for m in range(1, num_models + 1)}
blink_counters = {f'blink_counter_{name}': [] for name in ['gt'] + [f'model{i}' for i in range(1, num_models + 1)]}

plot_gt_vs_pred_flag = True

# Sample selection
row_idxs = [3663]  # random.sample(range(0, len(df)), 50)
row_idxs = [3953]  # random.sample(range(0, len(df)), 50)
row_idxs = random.sample(range(0, len(df)), 50) #[556,1004, 2877,1744, 3663]
# row_idxs = [3663,3642,3953] #, 317, 6117, 1710, 3252, 1820, 3847, ] #random.sample(range(0, len(df)+1), 30) #[556,1004, 2877,1744, 3663]
# row_idxs = [3663] #, 317, 6117, 1710, 3252, 1820, 3847, ] #random.sample(range(0, len(df)+1), 30) #[556,1004, 2877,1744, 3663]
# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================
for row_ind in tqdm(row_idxs):
    row = df.iloc[row_ind]
    
    # Load data and prepare sample
    data = np.load(features_path / row.run_path / f'{row.tar_id}.npy')
    sample, padding_mask = prepare_sample(data, device, st, en)
    gt_blendshapes = load_gt_blendshapes(row, gt_parent_path)
    
    # Run inference on all models
    all_predictions = []
    for model_info in models:
        config = model_info['config']
        model = model_info['model']
        
        if config['type'] == 'fairseq':
            blendshapes = infer_fairseq_model(model, sample, padding_mask, config['blendshape_indices'])
        elif config['type'] == 'nemo':
            blendshapes = infer_nemo_model(model, data, device, config['blendshape_indices'])
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Unnormalize blendshapes
        blendshapes_unnorm = normalize_blendshapes(
            blendshapes, 
            config['type'], 
            config['blendshape_indices'],
            blendshape_normalization_factors
        )
        all_predictions.append(blendshapes_unnorm)
    
    # Plot blendshapes comparison
    # plot_blendshape_comparison(gt_blendshapes, all_predictions, models, use_diff=False)
    
    # Align lengths
    min_len = min(len(gt_blendshapes), *[len(pred) for pred in all_predictions])
    gt_blendshapes = gt_blendshapes[:min_len]
    all_predictions = [pred[:min_len] for pred in all_predictions]
    
    # Compute metrics
    results = compute_metrics(gt_blendshapes, all_predictions, ['jawOpen', 'mouthFunnel', 'cheekPuff'])
    for key, val in results.items():
        if key.startswith('blink'):
            blink_counters[key].append(val)
        else:
            metrics[key].append(val)
    
    # # Optional plotting
    # if plot_gt_vs_pred_flag:
    #     plot_blendshape_comparison(gt_blendshapes, blendshapes_unnorm, blendshapes_unnorm2, blendshapes3) # Plit diff of Blendshapes (looks better)


# ============================================================================
# BLINK DETECTION ANALYSIS
# ============================================================================
def blink_detection_stats(gt, pred, model_name):
    """Calculate and print blink detection statistics."""
    print(f"\n=== Blink Detection Stats for {model_name} ===")
    correct, missed, imagined = 0, 0, 0
    for gt_val, pred_val in zip(gt, pred):
        if gt_val == pred_val:
            correct += gt_val
        elif pred_val < gt_val:
            missed += (gt_val - pred_val)
            correct += pred_val
        else:
            correct += gt_val
            imagined += (pred_val - gt_val)
    total, totl_pred = correct + missed + imagined, correct + missed
    print(f"Correct: {correct}, Missed: {missed}, Imagined: {imagined}, Total: {total}")
    print(f"Correct: {100*correct/totl_pred:.1f}%, Missed: {100*missed/totl_pred:.1f}%, Imagined: {100*imagined/total:.1f}% (Total: {sum(gt)} blinks)")

print(f"\n=== Blink Detection Analysis ===")
blink_gt = np.array(blink_counters['blink_counter_gt'])
for i in range(1, num_models + 1):
    model_name = MODEL_CONFIGS[i-1]['name']
    blink_detection_stats(blink_gt, np.array(blink_counters[f'blink_counter_model{i}']), f"Model {i} ({model_name})")


# ============================================================================
# FINAL RESULTS VISUALIZATION
# ============================================================================
def plot_model_comparison(metrics, blendshape_names, model_configs):
    """Create comparison plot for model performance across blendshapes."""
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=("'%' of time movement is in the same direction", 
                                       "Strength of velocity correlation",
                                       "RMSE"), 
                        shared_yaxes=False, 
                        horizontal_spacing=0.1)
    model_labels = [config['display_name'] for config in model_configs]
    
    for bs_name, color in zip(blendshape_names, PLOT_COLORS):
        # Extract metrics for this blendshape
        sign_means = [np.nanmean(metrics[f'sign_match_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        sign_stds = [np.nanstd(metrics[f'sign_match_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        r_means = [np.nanmean(metrics[f'r_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        r_stds = [np.nanstd(metrics[f'r_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        rmse_means = [np.nanmean(metrics[f'rmse_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        rmse_stds = [np.nanstd(metrics[f'rmse_{bs_name}_model{m}']) for m in range(1, len(model_configs) + 1)]
        
        fig.add_trace(go.Scatter(x=model_labels, y=sign_means, mode='lines+markers', name=f'{bs_name} (Sign Match)', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=sign_stds, visible=True, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=model_labels, y=r_means, mode='lines+markers', name=f'{bs_name}', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=r_stds, visible=True, color=color), showlegend=True), row=1, col=2)
        fig.add_trace(go.Scatter(x=model_labels, y=rmse_means, mode='lines+markers', name=f'{bs_name} (RMSE)', line=dict(color=color, width=3), marker=dict(size=10, color=color), error_y=dict(type='data', array=rmse_stds, visible=True, color=color), showlegend=False), row=1, col=3)
    
    fig.update_layout(title='Model Performance Comparison', height=500, width=1500, hovermode='x unified', legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_xaxes(title_text="Model", row=1, col=2)
    fig.update_xaxes(title_text="Model", row=1, col=3)
    fig.show()

plot_model_comparison(metrics, ['jawOpen', 'mouthFunnel', 'cheekPuff'], MODEL_CONFIGS)
a=1
