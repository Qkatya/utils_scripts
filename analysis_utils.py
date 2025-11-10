from collections import Counter   

 
def validation_analysis(df, run_paths=None, sampled_runpaths=None):
    counts = Counter(df.drop_duplicates(subset='run_path')['validation_status'])
    counts_existed = Counter(df.drop_duplicates(subset='run_path')['num_failed_frames'])
    
    if sampled_runpaths and run_paths:
        total_line = f"❗ TOTAL: {len(sampled_runpaths)} samples of {len(run_paths)} unique run paths, df size {len(df)}"
    elif run_paths:
        total_line = f"❗ TOTAL: {len(run_paths)} unique run paths, df size {len(df)}"
    else:
        total_line = f"❗ TOTAL: df size {len(df)}"
        
    summary_lines = [
        "Validation Summary:",
        total_line,
        f"✅ Success: {counts.get('valid', 0)}",
        f"✅ Out of Success, Existed: {counts_existed.get(-2, 0)}",
    ]
    for reason in ['wrong_dim', 'wrong_num_lmks', 'has_nan', 'lmks_vals_out_of_range', 'no_lmks']:
        summary_lines.append(f"❌ {reason}: {counts.get(reason, 0)}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    return summary_text