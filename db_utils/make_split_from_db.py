import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine


il_base_raw_data_path = Path("/mnt/A3000/Recordings/v2_data")


def get_IL_raw_data():
    query = """
        SELECT 
            q_frames_id::text as tar_id,
            instruction_type,
            text as read_text,
            q_stage_relative_path as run_path,
            q_frames_num_frames as frame_num,
            placement as side
        FROM dwh.q_frames_details 
        WHERE recording_date = '2026-01-11' AND participant_name='KatyaIvantsiv'
    """
    
    eng = create_engine("postgresql://bits_viewer:qviewer1@q-data-db.q.ai:5432/q-data-db-prod")
    data = pd.read_sql(query, con=eng)
    return data

il_data = get_IL_raw_data()
il_data = il_data.reset_index(drop=True)

# save df to pkl
output_path = Path("/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/katya_blendshapes_onboarding.pkl")
il_data.to_pickle(output_path)

print(f"IL data: {il_data.shape}")
print(f"Saved to: {output_path}")