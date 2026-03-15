"""
Sample N runs from blendshapes_no_beep split (excluding session intro/outro text),
copy video_full.mp4 locally, and write a manifest for the review Dash app.
"""
from pathlib import Path
import argparse
import random
import shutil
import pandas as pd

# Session intro/outro text to exclude (exact match on read_text)
EXCLUDED_READ_TEXTS = [
    "we're almost done - just two more sessions to go!\n\nnow we'd like you to perform an expression at three intensities: small, medium, and big",
    "great job so far!\n\nin the next session, we'll ask you to make repeated eye movements.\nplease follow the instructions on screen.",
    "in this session, you will make different facial expressions.\nplease follow the instructions on the screen.",
    "last session ahead.\nplease say each of the following texts out loud, in your normal voice.\nplease follow the instructions on the screen.",
    "thank you for helping us with this project. you're awesome.",
]


def main():
    parser = argparse.ArgumentParser(description="Sample runs and copy videos for blendshapes review.")
    parser.add_argument(
        "--split-pkl",
        default="/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/blendshapes_no_beep.pkl",
        help="Path to blendshapes_no_beep split pkl",
    )
    parser.add_argument(
        "--base-path",
        default="/mnt/A3000/Recordings/v2_data",
        help="Base path; run_path is joined to this",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for manifest and videos (default: blendshapes/review_100 next to this script)",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="Number of unique run_paths to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else script_dir / "review_100"
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_path)
    df = pd.read_pickle(args.split_pkl)

    # Filter: exclude rows whose read_text is exactly one of the 5 strings
    excluded_set = set(EXCLUDED_READ_TEXTS)
    text_col = "read_text" if "read_text" in df.columns else "text"
    filtered = df[~df[text_col].astype(str).str.strip().isin(excluded_set)].copy()

    # Unique run_paths; keep first row per run_path for read_text
    unique_runs = filtered.drop_duplicates(subset=["run_path"], keep="first")
    run_paths = unique_runs["run_path"].tolist()
    random.seed(args.seed)
    if len(run_paths) < args.num_samples:
        print(f"Warning: only {len(run_paths)} unique run_paths after filter; sampling all.")
        chosen_paths = run_paths
    else:
        chosen_paths = random.sample(run_paths, args.num_samples)
    path_to_row = unique_runs.set_index("run_path").to_dict("index")
    rows = []
    for i, run_path in enumerate(chosen_paths):
        row = path_to_row[run_path]
        read_text = row[text_col]
        video_filename = f"sample_{i:03d}.mp4"
        src = base_path / run_path / "video_full.mp4"
        dst = videos_dir / video_filename
        if src.exists():
            shutil.copy2(src, dst)
            video_status = "copied"
        else:
            video_status = "missing"
            print(f"Missing: {src}")
        rows.append({
            "sample_idx": i,
            "run_path": run_path,
            "read_text": read_text,
            "video_filename": video_filename,
            "video_status": video_status,
        })

    manifest_df = pd.DataFrame(rows)
    manifest_path = out_dir / "manifest.pkl"
    manifest_df.to_pickle(manifest_path)
    manifest_df.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Sampled {len(rows)} runs. Manifest saved to {manifest_path}")
    print(f"Videos in {videos_dir}")


if __name__ == "__main__":
    main()






