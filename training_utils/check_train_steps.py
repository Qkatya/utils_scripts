import torch

src_ckpt = "/mnt/ML/ModelsTrainResults/katya.ivantsiv/NeMo/landmarks/causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side/checkpoints/causal_fastconformer_layernorm_landmarks_all_blendshapes_one_side--val_wer=1.6158-epoch=6-last.ckpt"

checkpoint = torch.load(src_ckpt, map_location='cpu', weights_only=False)
current_step = checkpoint.get('global_step', checkpoint.get('step', None))

print(f"Current step: {current_step}")