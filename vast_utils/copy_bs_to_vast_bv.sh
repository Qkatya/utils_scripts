#!/bin/bash
#SBATCH --job-name=copy_bs_to_vast # Job name
#SBATCH --time=180:00:00               # Time limit hrs:min:sec
#SBATCH --output=/mnt/ML/ModelsTrainResults/katya.ivantsiv/SLURM/job_id_%A_%a_job_name_%x.txt
#SBATCH --tasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=180G
#SBATCH --partition=CPU_Partition
#SBATCH --qos=normal
#SBATCH --nice=1
#SBATCH --array=0

# sbatch /home/katya.ivantsiv/utils_scripts/vast_utils/copy_bs_to_vast_bv.sh

echo START TIME: $(date)

################# INPUTS
cd /home/katya.ivantsiv/utils_scripts/vast_utils
source /home/katya.ivantsiv/python-envs/venv-fairseq/bin/activate 
export PYTHONPATH=$PWD

################# /INPUTS

srun bash -c "PYTHONPATH=$PWD python copy_bs_to_vast.py"

echo END TIME: $(date)