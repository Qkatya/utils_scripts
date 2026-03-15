#import numpy as np
from jinja2 import Template
import os
from jinja2 import Environment, StrictUndefined
env = Environment(undefined=StrictUndefined)
username = 'lilach.barkat'

def apply_with_fail(txt,**kwargs):
    txt_template = env.from_string(txt)
    for k in kwargs:
        assert '{{' + k in txt, f'{k} not in template!'
    
    return txt_template.render(kwargs)


s_txt = '''#!/bin/bash
#SBATCH --job-name={{exp_name_pt}}   # Job name
#SBATCH --time=120:00:00               # Time limit hrs:min:sec
#SBATCH --output=/home/lilach.barkat/slurm_logs/{{experiment_base_folder}}/{{scpt_dir}}/{{exp_name_pt}}.txt   # Standard output and error log
#SBATCH --gres=gpu:{{n_gpus}}
#SBATCH --tasks={{n_gpus}}
#SBATCH --cpus-per-task=8
#SBATCH --mem={{160000*n_gpus}}
#SBATCH --partition={{partition}}
#SBATCH --nice={{nice}}
#SBATCH --qos={{qos}}
{{node_list}}

export OPENBLAS_NUM_THREADS=2
export GOTO_NUM_THREADS=2
export OMP_NUM_THREADS=2

source  /home/lilach.barkat/venvs/fairseq/bin/activate
cd /home/lilach.barkat/projects/q-fairseq-main
export PYTHONPATH="$PYTHONPATH:/home/lilach.barkat/projects/q-fairseq-main/examples/data2vec"
whereis python

'''

srun_txt = '''srun bash -c '\\
\\
\\python fairseq_cli/hydra_train.py -m --config-dir examples/data2vec/config/v2 \\
--config-name {{config_name}} \\
optimization.lr=[{{lr}}] \\
+task.data={{task_data}} \\
hydra.sweep.dir={{sweep_dir}} \\
distributed_training.distributed_world_size={{n_gpus}} \\
dataset.num_workers=8 \\
+task.min_sample_size=250 \\
task.max_sample_size=2000 \\
+task.use_h5={{use_h5}} \\
optimization.max_update={{max_updates}} \\
common.fp16={{fp16}} \\
common.bf16={{bf16}} \\
dataset.max_tokens={{max_tokens}} \\
optimization.update_freq=[{{update_frequency}}] \\
+model.modalities.laser.laser_conv_input_dim={{num_dims}} \\
+model.modalities.laser.batch_norm_input={{bn}} \\
"model.modalities.laser.feature_encoder_spec=\\"{{encoder_spec}}\\"" \\
{{H_W}} \\
+model.modalities.laser.conv_2plus1d={{conv_2plus1d}} \\
+task.feature_description={{features_base_path|default()}} \\
"+task.feat_idx=\\"{{channels|default((0,1,2,3))}}\\"" \\
+task.pretraining=True \\
+model.modalities.laser.avg_pool={{avg_pool}} \\
+model.modalities.laser.encoder={{encoder}} \\
+model.modalities.laser.skip_connection={{skip_connection}} \\
+model.modalities.laser.timm_output_dim={{timm_output_dim}} \\
+model.modalities.laser.timm_time_strides={{timm_time_strides}} \\
+model.modalities.laser.timm_pretrained_model_name={{timm_pretrained_model_name}} \\
+model.modalities.laser.GenericEncoder_uuid={{GenericEncoder_uuid}} \\
+model.modalities.laser.res_encoder_output_dim={{res_encoder_output_dim}} \\
+model.modalities.laser.res_encoder_activation={{res_encoder_activation}} \\
+model.SDPA={{SDPA}} \\
checkpoint.keep_interval_updates=10 \\
checkpoint.keep_last_epochs=20 \\
dataset.disable_dry_run=True \\
'
'''

pretrain_txt = s_txt + srun_txt

import hashlib
def hash_encoder(encoder_config):
    # Convert the encoder configuration to a string
    encoder_config_str = str(encoder_config)
    
    # Calculate the SHA-256 hash of the string
    sha256_hash = hashlib.sha256(encoder_config_str.encode()).hexdigest()
    
    # Take the first 8 characters of the hash as the encoder name
    encoder_name = sha256_hash[:8]
    
    # Store the encoder name as the key and the configuration as the value
    # encoder_mapping[encoder_name] = encoder_config
    # with open(f'/home/lilach.barkat/slurm/d2v/encoders/{encoder_name}.txt','wt') as f:
    #     f.write(encoder_config_str)
    print(encoder_config, encoder_name)
    return encoder_name


def channels_to_name(channels):
    import numpy as np
    if channels == (0,1,2,3):
        return ''
    
    if np.all(np.sort(channels) == np.arange(min(channels), 1 + max(channels))):
        channels_txt = f'_{min(channels)}to{max(channels)}'
        #print(channels_txt)
        return channels_txt
    else:
        return '_'+'_'.join([str(i) for i in np.sort(channels)])
    
channels_to_name((5,6,7))

import subprocess

## Fixed Generan params
# Which config to use, currently only those are supported
# pretrain_config_name = 'base_laser_only_enc_search.yaml'
pretrain_config_name =  'large_laser_only_task_conv3d_v1.yaml'
finetune_config_name = 'vox_10h_laser.yaml'
partition = 'A6000_L40S_VAST'
nice = 0
qos = 'normal'
fp16 = 'True'
bf16 = 'False'
SDPA = 'False' # make the model run Faster
max_updates_pre = 600_000
max_updates_fine = 400_000 
max_tokens_pre = 7_600 
max_tokens_fine = 45_000 
update_freq_pre = 2
update_freq_fine = 1
lr_pre  = 0.0004
lr_fine  = 0.00007
extra_trans_blocks_depth = 1
use_choice_sentences = 80
lip_embds  =          'True'
lip_reading_path = '/mnt/ML/Production/ML_Processed_Data/Audio_Features/lip_embedding/vdssskip6th_retinaface_ts1_ss4/results_summary.pkl'
lip_reading_treshold = 0.5
use_h5 =        'True'
mask_prob_fine = 0.05
hubert_soft  =        'True'
hubert_asr   =        'True'
avg_pool     =        'False'
conv_2plus1d =        'False'
bn =                  'True'
normalize    =        'False'
skip_connection =     'True'
GenericEncoder_params_str = ''
GenericEncoder_uuid = ''
res_encoder_activation = ''
res_encoder_output_dim = 0
timm_output_dim = 0
timm_time_strides = 0
timm_pretrained_model_name = ''

## Fixed Experiment params
n_gpus = 8

## Sweep params
features_base_path = '' #<-- if we want to use different features, can be ignored!

dataset = '20241210_split_1'
pretrain_task_data = '/mnt/ML/Development/ML_Data_DB/v2/splits/delta/20241210_pretrain_novalid'
finetune_task_data = '/mnt/ML/Development/oren.amsalem/splits/splits_v2/h5/20241218_split_1_cleanloud_hbert_harm0p8'
validation_datasets = 'asaf_kagan_full_20241230_silent@rani_alon_full_20241230_silent@SILENT_GIP_october_demo@LOUD_GIP_gen_conversationa_0912@LOUD_GIP_free_speech_question@LOUD_GIP_general_whisper_clean_1127@LOUD_GIP_tatoeba@LOUD_GIP_tiny_stories@SILENT_GIP_command_all_1127@SILENT_GIP_tiny_stories@SILENT_GIP_general_subject_clean_1127'

###############################

# encoder = 'ResEncoder'
encoder = 'LaserEncoder_V2Conv2D'

## LaserEncoder_V2  - LaserEncoder_V1R 
if encoder in ['LaserEncoder_V2', 'LaserEncoder_V1']:
    encoders = [[(35, (1, 3, 3), (1, 1, 1)), (64, (2, 5, 5), (2, 1, 1)), (128, (2, 4, 10), (2, 1, 1))]]

    conv_2plus1d =        'True'
    bn =                  'True'
    avg_pool            = 'True'

# LaserEncoder_V2Conv1D encoder = [[output_dim, kernel_size, stride, padding],..] = layer_desc
elif encoder == 'LaserEncoder_V2Conv1D':
    encoders = [[(1024, 1, 1, 0), (2048, 4, 4, 0)]]
    bn =                  'True'

# LaserEncoder_V2Conv2D encoder = [[output_dim, kernel_size, stride, padding],..] = layer_desc
elif encoder == 'LaserEncoder_V2Conv2D':
    encoders = [[(32, (1,3), (1,1), (0,1)), (128, (1, 3), (1, 1), (0,1)), (512, (2,3), (2,1), (0,1)), (64, (2,3), (2,1), (0,0))],
                [(32, (1,3), (1,1), (0,1)), (128, (1, 3), (1, 1), (0,1)), (512, (1,3), (1,1), (0,1)), (128, (2, 1), (2, 1), (0,0)), (32, (2,1), (2,1), (0,0))]]
    bn =                  'True'
    avg_pool            = 'True'

elif encoder == 'ImageEncoder':
    timm_output_dim = 512
    timm_time_strides = 2 # 2 for V2 or 1 for V1R
    timm_pretrained_model_name = 'efficientnet_b2'
    encoders = [[2000, [(1, (1,1,1), (timm_time_strides*2,1,1))]]]
    # normalize =             'True'

elif encoder == 'ResEncoder':
    encoders = [[(64, (4, 2, 2), (4, 1, 1), (2, 1, 1))]]
    res_encoder_output_dim = 128
    res_encoder_activation = 'relu'
    bn =                  'True'
    conv_2plus1d =        'True'

elif encoder == 'GenericEncoder':
    GenericEncoder_uuid='3049e313-516f-40ec-bc0c-744b3dd047d4'
    GenericEncoder_params_str='8004959f0c0000000000008c085f5f6d61696e5f5f948c0d456e636f646572506172616d739493942981947d94288c0d6c61796572735f706172616d73945d942868008c0b4c61796572506172616d739493942981947d94288c0f7472697669616c6974795f70726f6294473fc5775aec0524f38c08745f6b65726e656c944b028c0a785f616666696e697479944b018c0a795f616666696e69747994473fd6e076425f9ca28c116368616e6e656c5f657870616e73696f6e94473fe7fc308b89c6828c0f61637469766174696f6e5f66756e63948c1b746f7263682e6e6e2e6d6f64756c65732e61637469766174696f6e948c0447454c559493948c0a6e6f726d5f6c61796572948c1d746f7263682e6e6e2e6d6f64756c65732e696e7374616e63656e6f726d948c0e496e7374616e63654e6f726d33649493948c1173657065726174696f6e5f67726f75707394284b024b004b014b0174948c046269617394888c0f69735f656e645f6f665f626c6f636b94898c057061645f7894888c057061645f799488756268082981947d9428680b473fed208ea9a6c000680c4b05680d473fe4647580621318680e473fd3f85569a43def680f474000843bb628ae11681068118c075369676d6f6964949394681468176818284b004b004b014b027494681a88681b89681c88681d89756268082981947d9428680b473fe7f9751b55e1f2680c4b01680d473fe142b95c300550680e473fddb92c34486792680f473fdd561757c80eea681068118c094c65616b7952654c5594939468148c1a746f7263682e6e6e2e6d6f64756c65732e62617463686e6f726d948c0b42617463684e6f726d33649493946818284b004b024b004b017494681a88681b89681c88681d88756268082981947d9428680b473fbb91fbbbea6318680c4b03680d473fea6290d7030f27680e473fe471754bf90af5680f473fe49f44749007cc68106813681468176818284b004b034b014b027494681a89681b89681c88681d88756268082981947d9428680b473fd79cf6f171220d680c4b02680d473fcd13353514e870680e473fe7e5d4d059edb5680f473fe234a8c8d834fa681068118c0452654c55949394681468296818284b014b004b024b037494681a89681b89681c89681d88756268082981947d9428680b473fcca65a961a6e2c680c4b05680d473fea1cae7d1c3cb2680e473fdfcdf083ca4749680f473fd4680ffb662e0868106813681468296818284b014b034b004b027494681a88681b89681c89681d89756268082981947d9428680b473fdfda41c5766872680c4b01680d473fe85afa00896d55680e473fd031f4852eb05f680f474000452d75e8fb7068106821681468296818284b004b024b014b037494681a89681b89681c89681d89756268082981947d9428680b473fe4592ff758bca8680c4b02680d473fe06d490774cd18680e473fe7e68bdff0257b680f47bfe062f7df7da0dd68106821681468296818284b004b024b004b017494681a89681b89681c89681d88756268082981947d9428680b473facf576923a58ac680c4b05680d473fdd5b8f7f0af5a3680e473feabe1786fb5a58680f47400110928db1383c681068118c0454616e68949394681468296818284b024b034b014b007494681a88681b89681c89681d89756268082981947d9428680b473fd29095a07e69cf680c4b02680d473fd13c282aef9bde680e473fdd6cbddd05a2ce680f473fec34b5b1ed6c7368106813681468296818284b004b034b024b017494681a89681b88681c88681d88756268082981947d9428680b473fd02080dac3e160680c4b04680d473fe1037b84d4d1d2680e473fd822912727bc0e680f473ff5f6c71f6021e1681068118c0453454c55949394681468296818284b024b024b014b007494681a89681b89681c89681d88756268082981947d9428680b473fd774c14e534a16680c4b02680d4b01680e473fe2221c108bea60680f473fd3573bad8c1ad868106821681468296818284b034b014b024b007494681a88681b89681c88681d89756268082981947d9428680b473fc32d5dccdf6ece680c4b05680d473fd8e134163e6e59680e473fed1578cd054d79680f474007c67f4d7410a06810683f681468176818284b024b024b014b007494681a89681b89681c88681d88756268082981947d9428680b473fd13551a502ed70680c4b01680d473feb269f0417985e680e473f9cb2bbfe1bf084680f473fe89a921803d15f68106831681468296818284b004b024b014b007494681a89681b89681c89681d89756268082981947d9428680b473fdab49542313db6680c4b02680d473fe9de5a6b976c80680e473fb1a3b005bc3714680f473ff5226cef17ce4168106821681468176818284b024b034b004b017494681a88681b89681c88681d89756268082981947d9428680b473fe94a74c965fe1a680c4b04680d473fed4905643f9d8f680e473fd620af0c0b9e99680f473fa64633ac2a87d368106831681468176818284b014b014b024b007494681a88681b89681c88681d88756268082981947d9428680b473fefa5391e37297e680c4b04680d473feec8809c526ec0680e473fe3958846a161d3680f473ff70630541a9f7868106831681468176818284b004b014b024b007494681a88681b89681c88681d89756268082981947d9428680b473fe235b7f3705500680c4b01680d473fda6a32c3de66f2680e473fecdc60999012ee680f47bff069015af0ffaa68106821681468176818284b034b024b014b007494681a88681b89681c89681d89756268082981947d9428680b473fd3f15cbf3b7f0c680c4b02680d473fd11f873f0445f9680e473fd9af7e0c06e82a680f473ff1aac42c859b4d68106821681468176818284b014b004b024b027494681a89681b89681c89681d88756268082981947d9428680b473fe9fa54897eef03680c4b02680d473fe983c9398b24a9680e473fd52b2d5f79ac54680f473fbb4e8247fab4b068106813681468176818284b014b014b004b017494681a89681b88681c89681d89756268082981947d9428680b473fe2e8a098104266680c4b05680d473fd738fb70798d62680e473febb519ded62f49680f47bfae1ec3c1993a1d68106813681468296818284b004b024b004b017494681a88681b89681c89681d89756268082981947d9428680b473fd2237550dad396680c4b01680d473feb503743c51567680e4b01680f474002dc013cbbfb9668106813681468176818284b004b014b014b007494681a88681b89681c88681d88756268082981947d9428680b473fe137e37675a989680c4b02680d473fe410d13a00bd2e680e4b01680f473ff061078526a59f68106831681468296818284b014b004b024b027494681a88681b89681c88681d89756268082981947d9428680b473fe725f398cc45f2680c4b01680d473fda2578078bdb80680e473fddb5fd2712ef4e680f473fe364ade2cb1fd568106826681468296818284b034b004b024b017494681a89681b89681c89681d88756268082981947d9428680b473fe9416827401aef680c4b04680d473fe01500deeab521680e473fecd81039375194680f473fee2176491b435f68106813681468296818284b004b014b014b027494681a89681b89681c88681d89756268082981947d9428680b473fccf77f25de6d10680c4b01680d473fb1a35eef86df91680e473fc9362e2ac28dd8680f473ff07fe2e017a9d568106847681468176818284b034b004b024b017494681a89681b89681c88681d88756268082981947d9428680b473fd5d44a946d454c680c4b02680d473fe7168c905a4a00680e473fec1e0b6fe976a0680f473fe6e10db20acda068106821681468176818284b004b014b014b027494681a89681b89681c88681d88756268082981947d9428680b473fdab33ec8f3cb5a680c4b05680d473fe66fb36d16eb85680e473fdce6a832a4825f680f47bfcba44068a96d4068106813681468296818284b004b004b004b007494681a88681b89681c89681d89756268082981947d9428680b473fe7e1633299b43e680c4b03680d473fd6ada34eb9a82f680e473fe74ebd5993d558680f473ff4ccd2a299502268106847681468176818284b014b004b014b007494681a88681b89681c89681d89756268082981947d9428680b473fe4a1cd74d88d2b680c4b01680d473fd9d3cdbcd226aa680e473fb24241566200df680f473ffd9dd87ffaebf3681068136814682968186840681a88681b89681c88681d897562658c106d61785f6d6f64656c5f706172616d73944ad59446008c0c6d696e5f74726573686f6c6494473fece9bb3dd6a95a8c0f746f74616c5f785f6b65726e656c73944b098c0f746f74616c5f795f6b65726e656c73944b0f8c0b696e5f6368616e6e656c73944b0d8c0c6f75745f6368616e6e656c73944d00028c0d62696e6172795f736561726368948975622e'
    encoders = [[2000, [(1, (1,1,1), (-4,1,1))]]]

else:
    raise Exception("Unknown encoder")


feature_grad_mult =  '0.0' # or 1.0
if feature_grad_mult!= '1.0':
    print([f'{feature_grad_mult=}\n']*10)
seed = 13 # None | number #  just create another run from scratch, it does not really change any seed
H_W = '"+model.modalities.laser.input_HW=\\"(10,16)\\""'



if fp16 == 'True' and bf16 == 'True':
    raise Exception("Can not use fp16 and bf16 at the same time!")

# if normalize == 'True' and bn == 'True':
#     raise Exception("Can not use normalize and bn at the same time!")

node_list =  '' ##SBATCH --nodelist=gpu[12,13,16,17,18,19]'
#node_list = 'SBATCH --nodelist=gpu0'
channels = tuple(list(range(13)))

    
num_dims = len(channels)

experiment_base_folder = 'd2v_V2_v1'
base_dir = f'/home/lilach.barkat/slurm/run_scripts/{experiment_base_folder}'
scpt_dir = '3D'

'''
/home/lilach.barkat/slurm/run_scripts/d2v_V2_v1/3D/pretrain_name_pt.sbatch
/home/lilach.barkat/slurm/run_scripts/d2v_V2_v1/3D/pretrain_name_finetune_name.sbatch
/mnt/ML/TrainResults/lilach.barkat/d2v_V2_v1/3D/pretrain_name/finetune_name/checkpoints/checkpoint_last.pt
/home/lilach.barkat/slurm_logs/d2v_V2_v1/pretrain_name_pt.txt
/home/lilach.barkat/slurm_logs/d2v_V2_v1/pretrain_name_finetune_name.txt

'''
model_save_path_base = '/mnt/ML/TrainResults/lilach.barkat'

assert os.path.exists(f'/home/lilach.barkat/slurm_logs/{experiment_base_folder}/{scpt_dir}'), f'log folder does not exsits such  /home/lilach.barkat/slurm_logs/{experiment_base_folder}/{scpt_dir} '

run_validation = False
os.chdir(f"{base_dir}/{scpt_dir}")
for encoder_spec in encoders:
 
    chan_txt = channels_to_name(channels)
    
    text_constuct = []
    text_constuct.append(f'_{dataset}' if dataset else '')
    text_constuct.append(f'_large_no_nrm')
    if encoder in ['LaserEncoder_V2','LaserEncoder_V1', 'LaserEncoder_V2Conv2D', 'LaserEncoder_V2Conv1D','ImageEncoder', 'ResEncoder']:
        text_constuct.append(f'_bn' if eval(bn) else '')

    # text_constuct.append(f'_nrm' if eval(normalize) else '')
    if encoder in ['ResEncoder']:
        encoder_name = hash_encoder(encoder_spec)
        text_constuct.append(f'_res{res_encoder_output_dim}')
        text_constuct.append(f'_act{res_encoder_activation}')
        text_constuct.append(f'_2p1d' if eval(conv_2plus1d) else '')

    if encoder in ['LaserEncoder_V2', 'LaserEncoder_V1', 'LaserEncoder_V2Conv1D', 'LaserEncoder_V2Conv2D']:
        encoder_name = hash_encoder(encoder_spec)
        text_constuct.append(f'_le{encoder.split("_")[1]}')
        text_constuct.append(f'_sc' if eval(skip_connection) else '')
        if encoder in ['LaserEncoder_V2', 'LaserEncoder_V1', 'LaserEncoder_V2Conv2D']:
            text_constuct.append(f'_avgpool' if eval(avg_pool) else '')
            if encoder in ['LaserEncoder_V2', 'LaserEncoder_V1']:
                text_constuct.append(f'_2p1d' if eval(conv_2plus1d) else '')
    
    elif encoder == 'GenericEncoder':
        encoder_name = GenericEncoder_uuid[:6]
    elif encoder == 'ImageEncoder':
        encoder_name = f'{timm_pretrained_model_name}'
        text_constuct.append(f'_dim{timm_output_dim}')
        text_constuct.append(f'_st{timm_time_strides}')
        


    text_constuct.append(f'_Pstep{max_updates_pre/1000}k')
    text_constuct.append(f'_mt{max_tokens_pre/1000}k')
    text_constuct.append(f'_uf{update_freq_pre}')
    
    text_constuct.append(f'_prelr{lr_pre}'.replace('.','p'))
    
    text_constuct.append(f'_s{seed}' if seed else '')
    text_constuct.append(f'_bf16' if eval(bf16) else '')
    text_constuct.append(f'_fp32' if (not eval(bf16) and not eval(fp16)) else '')
    text_constuct.append(f'noSDPA' if not eval(SDPA) else '')
    text_constuct.append(channels_to_name(channels))

    
    exp_name = f'{encoder_name}'
    for txt in text_constuct:
        exp_name += txt
    exp_name = exp_name.replace('.','p')
    
    print(exp_name)
    exp_name_pt = exp_name +'_pt'   
    print(exp_name_pt)
    sweep_dir = f'{model_save_path_base}/{experiment_base_folder}/{scpt_dir}/{exp_name}'
    print(f'{sweep_dir=}')
    
    pretrain_sweep_dir = f'{sweep_dir}/pretrain'
    
    pretrain_sbatch = apply_with_fail(pretrain_txt,
                                      experiment_base_folder=experiment_base_folder,scpt_dir=scpt_dir,
                                      num_dims=num_dims, encoder_spec=encoder_spec,
                                      bn=bn.lower(), n_gpus=n_gpus, 
                                      skip_connection = skip_connection.lower(),
                                      task_data=pretrain_task_data,
                                      encoder=encoder,
                                      config_name=pretrain_config_name,
                                      max_tokens=max_tokens_pre,
                                      max_updates=max_updates_pre,
                                      update_frequency=update_freq_pre,
                                      sweep_dir=pretrain_sweep_dir,
                                      conv_2plus1d=conv_2plus1d.lower(), 
                                      exp_name_pt=exp_name_pt,
                                      features_base_path=features_base_path,
                                      channels=channels, 
                                      node_list=node_list,
                                      H_W=H_W,
                                      avg_pool=avg_pool.lower(),
                                      timm_output_dim=timm_output_dim,
                                      timm_time_strides=timm_time_strides,
                                      timm_pretrained_model_name=timm_pretrained_model_name,
                                      GenericEncoder_uuid=GenericEncoder_uuid,
                                      res_encoder_output_dim=res_encoder_output_dim,
                                      res_encoder_activation=res_encoder_activation,
                                      bf16=bf16.lower(),
                                      fp16=fp16.lower(),
                                      SDPA=SDPA.lower(),
                                      use_h5=use_h5.lower(),
                                      lr=lr_pre,
                                      partition=partition,
                                      nice=nice,
                                      qos=qos,)

    print(pretrain_sbatch)
    #print(pretrain_sbatch)
    pretrain_job = f'{base_dir}/{scpt_dir}/{exp_name}_pt.sbatch'
    '''
    with open(pretrain_job,'wt') as f:
        f.write(pretrain_sbatch)
    print(f'{pretrain_job=}')

    pretrain_output = subprocess.check_output(['sbatch',pretrain_job])
    
    print(pretrain_output)
    
    jobid = pretrain_output.decode('ascii').strip().split(' ')[-1]
    #'''

    text_constuct = ['']
    text_constuct.append(f'_fnlr{lr_fine}'.replace('.','p') if lr_fine != 0.0001 else '')
    text_constuct.append(f'_Fstep{max_updates_fine/1000}k')
    text_constuct.append(f'_mt{max_tokens_fine/1000}k')
    text_constuct.append(f'_uf{update_freq_fine}')
    
    text_constuct.append(f'_fgm{feature_grad_mult}' if float(feature_grad_mult)!=0 else '')
    text_constuct.append(f'_hbrt' if eval(hubert_soft) else '')
    text_constuct.append(f'_hbrtasr' if eval(hubert_asr) else '')
    text_constuct.append(f'noSDPA' if not eval(SDPA) else '')
    
    text_constuct.append(f'_exblock{extra_trans_blocks_depth}' if extra_trans_blocks_depth else '')
    text_constuct.append(f'_choice{use_choice_sentences}' if use_choice_sentences else '')
    text_constuct.append(f'_lip' if eval(lip_embds) else '')
    text_constuct.append(f'_liptreshold{lip_reading_treshold}' if lip_reading_treshold else '')

    text_constuct.append(f'_mask_prob{mask_prob_fine}'.replace('.','p') if mask_prob_fine != 0.05 else '')


    
    ft_features_txt = ''.join(text_constuct)
    exp_name_ft = exp_name +'_ft' + ft_features_txt 
    exp_name_ft=exp_name_ft.replace('.','p')
    finetune_sweep_dir = f'{sweep_dir}/finetune{ft_features_txt}'
    finetune_sbatch = apply_with_fail(finetune_txt,
                                      experiment_base_folder=experiment_base_folder,scpt_dir=scpt_dir,
                                      max_tokens=max_tokens_fine,
                                      max_updates=max_updates_fine,
                                      update_frequency=update_freq_fine,
                                      n_gpus=n_gpus, task_data=finetune_task_data, config_name=finetune_config_name,
                                      sweep_dir=finetune_sweep_dir, pretrained_path=f'{pretrain_sweep_dir}/0/checkpoints/checkpoint_last.pt',
                                      feature_grad_mult=feature_grad_mult,
                                      features_base_path=features_base_path,
                                      channels=channels,
                                      node_list=node_list,
                                      hubert_soft = hubert_soft.lower(),
                                      hubert_asr = hubert_asr.lower(),
                                      lr=lr_fine,
                                      bf16=bf16.lower(),
                                      fp16=fp16.lower(),
                                      SDPA=SDPA.lower(),
                                      extra_trans_blocks_depth=extra_trans_blocks_depth,
                                      use_choice_sentences=use_choice_sentences,
                                      lip_embds=lip_embds.lower(),
                                      lip_reading_path=lip_reading_path,
                                      lip_reading_treshold=lip_reading_treshold,
                                      use_h5=use_h5.lower(),
                                      mask_prob=mask_prob_fine,
                                      exp_name_ft=exp_name_ft,
                                      validation_datasets=validation_datasets,
                                      partition=partition,
                                      nice=nice,
                                      qos=qos,
                                     )
                                  # add_raw_image=add_raw_image)
    #sdf
    #print(finetune_sbatch)
    # '''
    fintune_job = f'{base_dir}/{scpt_dir}/{exp_name_ft}.sbatch'
    # with open(fintune_job,'wt') as f:
    #     f.write(finetune_sbatch)
    print(f'{fintune_job=}')
    # finetune_output = subprocess.check_output(['sbatch',f'--dependency=afterok:{jobid}', fintune_job])
    # finetune_output = subprocess.check_output(['sbatch', fintune_job])
    # jobid_ft = finetune_output.decode('ascii').strip().split(' ')[-1]
    # print(finetune_output)
    #'''