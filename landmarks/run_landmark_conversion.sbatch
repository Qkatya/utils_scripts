#!/bin/bash
#SBATCH --job-name=landmark_convert
#SBATCH --output=landmark_convert_%j.out
#SBATCH --error=landmark_convert_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com

# Load any necessary modules (adjust based on your cluster)
# module load python/3.8
# module load numpy

# Set the path to your landmark directory
LANDMARK_DIR="/mnt/ML/Development/katya.ivantsiv/landmarks"

# Set the path to the script
SCRIPT_PATH="$(pwd)/convert_landmarks_to_float16.py"

# Create log directory
mkdir -p logs

# First, run a dry run to see what we're dealing with
echo "Starting dry run to analyze files..."
python $SCRIPT_PATH $LANDMARK_DIR --dry-run --n-jobs 96 --batch-size 1000 > logs/dry_run_$(date +%Y%m%d_%H%M%S).log 2>&1

# Check if dry run was successful
if [ $? -eq 0 ]; then
    echo "Dry run completed successfully. Starting actual conversion..."
    
    # Run the actual conversion
    python $SCRIPT_PATH $LANDMARK_DIR --n-jobs 96 --batch-size 1000 > logs/conversion_$(date +%Y%m%d_%H%M%S).log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Conversion completed successfully!"
    else
        echo "Conversion failed. Check the log files for details."
        exit 1
    fi
else
    echo "Dry run failed. Check the log files for details."
    exit 1
fi

echo "Job completed at $(date)" 