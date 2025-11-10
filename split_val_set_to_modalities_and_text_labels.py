import pandas as pd
from pathlib import Path
import sys
import numpy.core.numeric as numeric
sys.modules['numpy._core.numeric'] = numeric


val_Set_paths = [Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/bursk_exp_full200fps/valid_original_frame_num.pkl")]
# val_Set_paths = [Path("/mnt/ML/Development/dolev.orgad/low_fps_Q_Features/splits/linear/100fps/valid.pkl"),
#                 Path("/mnt/ML/Development/dolev.orgad/low_fps_Q_Features/splits/burst/100fps/valid.pkl"),
#                 Path("/mnt/ML/Development/dolev.orgad/low_fps_Q_Features/splits/burst/50fps/valid.pkl"),
#                 Path("/mnt/ML/Development/dolev.orgad/low_fps_Q_Features/splits/linear/50fps/valid.pkl")]

text_to_label = pd.read_pickle(open("/mnt/A3000/Scratch/users/michael.doron/text_to_label_dict.pkl", 'rb'))

for val_Set_path in val_Set_paths:
    df = pd.read_pickle(val_Set_path)
    save_folder = Path(f'/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/burst_exp_full_200fps_valid')
    # save_folder = Path(f'/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/burst_exp_{val_Set_path.parts[-3]}_{val_Set_path.parts[-2]}_valid')

    # Create subsets based on instruction_type
    df_loud = df[df['instruction_type'] == 'loud'].copy()
    df_lip = df[df['instruction_type'] == 'lip'].copy()
    df_silent = df[df['instruction_type'] == 'silent'].copy()

    print(f"Original dataset size: {len(df)}")
    print(f"Loud subset size: {len(df_loud)}")
    print(f"Lip subset size: {len(df_lip)}")
    print(f"Silent subset size: {len(df_silent)}")


    sys.path.append("/home/oren.amsalem/projects/split_stuff/")
    import text_to_text_mapping                    # first import
    # importlib.reload(text_to_text_mapping)         # re-executes the file
    label_to_label_set = text_to_text_mapping.label_to_label_set


    def apply_label(x):
        x1 = frozenset(map(str.strip, x.split(',')))
        if x1 in label_to_label_set:
            return label_to_label_set[x1]
        else:
            if 'beholder_wer_score' in x:
                return 'october_demo'
            else:
                print(x)
                return x

    df_loud['text_labels'] = df_loud.read_text.apply(lambda x:text_to_label[x])
    df_lip['text_labels'] = df_lip.read_text.apply(lambda x:text_to_label[x])
    df_silent['text_labels'] = df_silent.read_text.apply(lambda x:text_to_label[x])

    df_loud['text_label_clean'] = df_loud.text_labels.apply(apply_label)
    df_lip['text_label_clean'] = df_lip.text_labels.apply(apply_label)
    df_silent['text_label_clean'] = df_silent.text_labels.apply(apply_label)

    df_loud_tiny_story = df_loud[df_loud['text_label_clean']=='tiny_story']
    df_lip_tiny_story = df_lip[df_lip['text_label_clean']=='tiny_story']
    df_silent_tiny_story = df_silent[df_silent['text_label_clean']=='tiny_story']

    df_loud_general = df_loud[df_loud['text_label_clean']=='general']
    df_lip_general = df_lip[df_lip['text_label_clean']=='general']
    df_silent_general = df_silent[df_silent['text_label_clean']=='general']

    df_loud_commands = df_loud[df_loud['text_label_clean']=='commands']
    df_lip_commands = df_lip[df_lip['text_label_clean']=='commands']
    df_silent_commands = df_silent[df_silent['text_label_clean']=='commands']

    # Save each subset to separate h5 files
    print(f'save folder: {save_folder}')
    print(f'{val_Set_path.parts[-3]} {val_Set_path.parts[-2]}')
    for subset_name, subset_df in [('loud_tiny_story', df_loud_tiny_story), ('lip_tiny_story', df_lip_tiny_story), ('silent_tiny_story', df_silent_tiny_story), ('loud_general', df_loud_general), ('lip_general', df_lip_general), ('silent_general', df_silent_general), ('loud_commands', df_loud_commands), ('lip_commands', df_lip_commands), ('silent_commands', df_silent_commands)]:
        output_path = save_folder / f'valid_{subset_name}.pkl'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subset_df.to_pickle(output_path)
        print(f'{subset_name} size: {len(subset_df)}')
        
        print('==========================================')
        print(f'saved {subset_name} to {output_path}')
        print('==========================================')

a=1
