import pandas as pd
import psycopg2
from pathlib import Path

# Same DB as add_side_from_db / process_split_add_side_to_h5 (psycopg2 pattern)
_DB = {
    "host": "q-data-db-replica.q.ai",
    "port": 5432,
    "database": "q-data-db-prod",
    "user": "bits_viewer",
    "password": "qviewer1",
}


def query_db(query: str) -> pd.DataFrame:
    """Run SQL and return a DataFrame. Same pattern as add_side_column.py query_db()."""
    conn = psycopg2.connect(**_DB)
    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def get_blendshapes_no_beep_from_db():
    query = """
        SELECT
            qfd.q_frames_id AS tar_id,
            qfd.instruction_type,
            qfd.text AS read_text,
            qfd.text,
            qfd.q_stage_relative_path AS run_path,
            qfd.q_frames_num_frames AS frame_num,
            qfd.placement,
            qfd.sensor_version,
            qfd.recording_date,
            qfd.stage_id,
            pwm.display_name AS profile_display_name,
            pwm.id AS profile_id,
            ed.descriptor_value::bool AS has_glasses
        FROM dwh.q_frames_details qfd
            LEFT JOIN profiles_with_metadata pwm ON pwm.id = qfd.profile_id
            LEFT JOIN experiment_descriptors ed ON (ed.experiment_id = qfd.session_id AND ed.descriptor_type = 'glasses')
        WHERE
            NOT qfd.is_cancelled
            AND pwm.name IN ('1087_blendshapes_no_beep', 'blendshapes_no_beep')
    """
    return query_db(query)

# def get_blendshapes_no_beep_from_db():
#     query = """
#         SELECT
#             qfd.q_frames_id AS tar_id,
#             qfd.instruction_type,
#             qfd.text AS read_text,
#             qfd.text,
#             qfd.q_stage_relative_path AS run_path,
#             qfd.q_frames_num_frames AS frame_num,
#             qfd.placement,
#             qfd.sensor_version,
#             qfd.recording_date,
#             qfd.stage_id,
#             pwm.display_name AS profile_display_name,
#             pwm.id AS profile_id
#         FROM dwh.q_frames_details qfd
#         LEFT JOIN profiles_with_metadata pwm ON pwm.id = qfd.profile_id
#         INNER JOIN profile p ON p.id = qfd.profile_id AND p.name = 'blendshapes_no_beep'
#     """
#     return query_db(query)


def to_split_names(df):
    """Apply same compatibility renames/aliases as create_newID_split_ROIs_1000fps.ipynb
    so column names match what they use in train.pkl / valid.pkl.
    """
    df = df.copy()
    df["tar_id"] = df["tar_id"].astype(str)
    df["stage_id"] = df["stage_id"].astype(str)
    df = df.drop_duplicates(subset=["tar_id"])
    df["side"] = df["placement"]
    df["profile_name"] = df["profile_display_name"]
    df["is_stage_cancelled"] = False
    df["is_stage_skipped"] = False
    df["flow_name"] = df["profile_display_name"].fillna("")
    df["subject"] = df["profile_display_name"].fillna("").astype(str)
    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
    return df


def get_blendshapes_no_beep_split(out_pkl=None):
    """Fetch blendshapes_no_beep from DB, apply split-style names, optionally save to pkl.
    Returns DataFrame with names used in create_newID_split_ROIs_1000fps.ipynb.
    If out_pkl is set, saves to that path (e.g. 'blendshapes_no_beep.pkl').
    """
    df = get_blendshapes_no_beep_from_db()
    df = to_split_names(df)
    if out_pkl is not None:
        Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out_pkl)
        print(f"Saved {len(df)} rows to {out_pkl}")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch blendshapes_no_beep split from DB and save pkl.")
    p.add_argument("-o", "--out", default="/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_no_beep.pkl", help="Output pkl path")
    p.add_argument("--no-save", action="store_true", help="Only load and print shape, do not save")
    args = p.parse_args()
    df = get_blendshapes_no_beep_split(out_pkl=None if args.no_save else args.out)
    print(f"Rows: {len(df)}, columns: {list(df.columns)}")
    print(df.head())
