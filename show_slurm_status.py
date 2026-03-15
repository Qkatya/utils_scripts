import argparse
import collections
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
# from utils.users import get_username

import numpy as np

# from utils.general_utils import format_timedelta

def get_username():
    result = subprocess.run(["whoami"], capture_output=True, text=True)
    return result.stdout.strip()

def format_timedelta(td):
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds:
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"

def ssh_run_command(host, command):
    """
    Executes an SSH command on a remote server without using third-party libraries.

    :param host: The hostname or IP address of the remote server.
    :param command: The command to be executed on the remote server.
    :return: A tuple containing the stdout and stderr of the executed command.
             If an exception occurs, returns (None, error_message).
    """
    try:
        # Construct the SSH command
        # ssh_command = ["ssh", "-p", str(port), f"{username}@{host}", command]
        ssh_command = ["ssh", host, command]

        # Execute the SSH command
        result = subprocess.run(
            ssh_command,
            text=True,  # Ensure output is in string format
            capture_output=True,  # Capture stdout and stderr
        )

        # Return stdout and stderr
        return result.stdout, result.stderr
    except Exception as e:
        return None, str(e)


def find_running_tasks(job_id: int, job_name: str):
    squeue_output = ssh_run_command("slurm-node01.q.ai", "squeue -u $USER -h")[0]
    lines = squeue_output.strip().splitlines()
    lines = [line for line in lines if job_name[:7] in line]
    job_and_task_ids = [s.strip().split(" ")[0].split("_") for s in lines]
    if job_and_task_ids:
        job_ids = np.array(job_and_task_ids)[:, 0].astype(int)
        task_id_strs = np.array(job_and_task_ids)[:, 1]
        task_id_strs = task_id_strs[job_ids == job_id]
        running_task_ids = []
        scheduled_task_ids = []
        for task_id_str in task_id_strs:
            if "[" in task_id_str:  # Jobs that haven't started yet
                range_strs = task_id_str.split(",")
                range_strs[0] = range_strs[0][1:]  # Remove "["
                range_strs[-1] = range_strs[-1][:-1]  # Remove "]"
                for range_str in range_strs:
                    dash_idx = range_str.index("-")
                    scheduled_task_ids += list(range(int(range_str[:dash_idx]), int(range_str[dash_idx + 1 :]) + 1))
                # dash_idx = task_id_str.index("-")
                # scheduled_task_ids = list(range(int(task_id_str[1:dash_idx]), int(task_id_str[dash_idx + 1 : -1])))
            else:  # A job that's currently running
                running_task_ids.append(int(task_id_str))
        return running_task_ids, scheduled_task_ids
    return [], []


def _get_latest_log_dir(base_log_dir: Path) -> Path:
    log_dirs = base_log_dir.glob("job_id_*")
    latest_log_dir = max(log_dirs, key=lambda d: d.stat().st_mtime)
    return latest_log_dir
    # job_id = int(latest_log_dir.name.split("_")[-1])
    # return job_id


def _parse_time_line(log_lines, prefix) -> datetime.date:
    time_line = next((l for l in log_lines if prefix in l), None)
    if time_line is None:
        return None
    return datetime.strptime(time_line[len(prefix) :], "%a %b %d %H:%M:%S %Z %Y")


def _parse_duration_from_logs(log_files) -> str:
    start_times = []
    end_times = []
    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
                log_content = file.read()
        except Exception as err:
            print(f"Failed to read log file '{log_file}': {err}")

        lines = log_content.split("\n")

        if (start_time := _parse_time_line(lines, "START TIME: ")) is not None:
            start_times.append(start_time)
        if (end_time := _parse_time_line(lines, "END TIME: ")) is not None:
            end_times.append(end_time)

    if start_times and end_times:
        start_time = min(start_times)
        end_time = max(end_times)
        duration_str = format_timedelta(end_time - start_time)
    else:
        duration_str = "Unknown"
    return duration_str


def main(job_id: int | None, job_name: str = ""):
    FAIL_RATIO_THRESH = 0.05
    # base_log_dir = Path(f"/home/{get_username()}/slurm_logs") / job_name
    base_log_dir = Path(f"/mnt/ML/TrainResults/{get_username()}/SLURM")
    FAIL_RATIO_THRESH = 0.05

    if job_id is None:
        log_dir = _get_latest_log_dir(base_log_dir)
        job_id = int(log_dir.name.split("_")[-1])
    else:
        log_dir = base_log_dir / f"job_id_{job_id}_job_name_{job_name}.txt"
    modified_time = log_dir.stat().st_mtime
    modified_time_str = time.ctime(modified_time)

    print(f"Job ID: {job_id}")
    print(f"Last modified: {modified_time_str}")

    running_task_ids, scheduled_task_ids = find_running_tasks(job_id, job_name)
    # running_task_ids, scheduled_task_ids = [], []

    log_files = list(log_dir.glob("*.txt"))
    num_tasks_running = len(running_task_ids)
    num_tasks_scheduled = len(scheduled_task_ids)
    num_tasks_total = len(log_files) + num_tasks_scheduled
    num_tasks_finished = 0
    all_stats = collections.Counter()
    for log_file in log_files:
        task_id = int(log_file.stem.split("_")[1])

        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as file:
                log_content = file.read()
            lines = log_content.split("\n")
            finish_stats_line = next((line for line in lines if f"Finished {job_name}" in line), None)

            if finish_stats_line is None:
                if task_id in running_task_ids:
                    running_stats_line = next((line for line in reversed(lines) if f"{job_name} stats" in line), None)
                    if running_stats_line is not None:
                        stats_str = running_stats_line[running_stats_line.index("{") :]
                        stats = json.loads(stats_str.replace("'", '"'))
                        all_stats += stats
                        if stats["failed"] > FAIL_RATIO_THRESH * stats["total"]:
                            print(f"Log of failed task {log_file.name}:\n")
                            print(log_content)
                            # print("\n".join([l for l in lines if "DELET" in l]))
                else:
                    num_tasks_finished += 1
                    processing_line = next((line for line in lines if "Processing" in line and "rows" in line), None)
                    if processing_line is not None:
                        # Finished task that did not write a finish stats line - assuming all jobs failed
                        num_rows = int(processing_line.split(" ")[1])
                        all_stats["failed"] += num_rows
                    print(f"Log of failed task {log_file.name}:\n")
                    print(log_content)
                    # print("\n".join([l for l in lines if "DELET" in l]))
            else:
                num_tasks_finished += 1
                stats_str = finish_stats_line[finish_stats_line.index("{") :]
                stats = json.loads(stats_str.replace("'", '"'))
                all_stats += stats

                if stats["failed"] > FAIL_RATIO_THRESH * stats["total"]:
                    print(f"Log of failed task {log_file.name}:\n")
                    print(log_content)
                    # print("\n".join([l for l in lines if "DELET" in l]))
        except Exception as err:
            print(f"Failed to read log file '{log_file}': {err}")

    all_finished = num_tasks_total == num_tasks_finished
    failed = all_stats["failed"] > 0
    duration_str = _parse_duration_from_logs(log_files)
    nf = all_stats["failed"]
    ef = all_stats["exists"]

    print(f"Job ID: {job_id}")
    print(f"Last modified: {modified_time_str}")
    print(f"Finished {num_tasks_finished}/{num_tasks_total} tasks")
    print(f"{num_tasks_running} tasks still running")
    print(f"{num_tasks_scheduled} tasks scheduled")
    # print(f"Tasks' stats: {all_stats}")
    # print("FINISHED" if all_finished else "NOT FINISHED")
    # print("FAILED" if failed else ("SUCCEEDED" if all_finished else "NO FAILURES YET"))
    print(f"Existing files: {ef}")
    print(f"Not existing files: {nf}")
    print(f"Job duration: {duration_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, help="Takes latest by default")
    parser.add_argument("--job_name", type=str, default="feature_extraction_200fps", help="One of: copy_raw_data, feature_extraction_200fps")
    args = parser.parse_args()

    main(args.job_id, args.job_name)