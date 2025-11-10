#!/bin/bash
#SBATCH --job-name=sample_landmark_convert
#SBATCH --output=sample_landmark_convert_%j.out
#SBATCH --error=sample_landmark_convert_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=katya.ivantsiv@example.com

# Load any necessary modules (adjust based on your cluster)
# module load python/3.8
# module load numpy

# Set the path to the script
SCRIPT_PATH="$(pwd)/convert_canonical_landmarks_to_float16.py"

# Create log directory
mkdir -p logs

# Update the script to use sample folder
echo "Updating script to use sample folder..."
sed -i 's|LANDMARKS_BASE_PATH = "/mnt/ML/Development/katya.ivantsiv/landmarks"|LANDMARKS_BASE_PATH = "/mnt/ML/Development/katya.ivantsiv/landmarks_sample"|' $SCRIPT_PATH

echo "Starting conversion on sample folder..."
python $SCRIPT_PATH > logs/sample_conversion_$(date +%Y%m%d_%H%M%S).log 2>&1

if [ $? -eq 0 ]; then
    echo "Sample conversion completed successfully!"
else
    echo "Sample conversion failed. Check the log files for details."
    exit 1
fi

# Restore original path
echo "Restoring original path..."
sed -i 's|LANDMARKS_BASE_PATH = "/mnt/ML/Development/katya.ivantsiv/landmarks_sample"|LANDMARKS_BASE_PATH = "/mnt/ML/Development/katya.ivantsiv/landmarks"|' $SCRIPT_PATH

echo "Job completed at $(date)" 