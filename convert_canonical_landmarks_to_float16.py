#!/usr/bin/env python3
"""
Convert canonical landmarks from float32 to float16.
Loads unique run_paths from dataframe and processes landmarks in parallel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import numpy.lib.format as npformat
import mmap

# Configure paths
DATAFRAME_PATH = "/mnt/ML/ModelsTrainResults/katya.ivantsiv/splits/loud_and_whisper_and_lip_20250713_064722.pkl"
LANDMARKS_BASE_PATH = "/mnt/ML/Development/katya.ivantsiv/landmarks"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dtype_without_loading(file_path):
    """Get dtype of numpy file without loading the entire array."""
    try:
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Read magic number and header
            magic = npformat.read_magic(mm)
            shape, fortran_order, dtype = npformat.read_array_header_1_0(mm)
            
            mm.close()
            return dtype
            
    except Exception as e:
        return f"Error: {str(e)}"
    
def convert_landmark_file(run_path: str) -> tuple[bool, str]:
    """Convert a single landmark file from float32 to float16 safely."""
    try:
        # Construct landmark file path
        landmark_path = Path(LANDMARKS_BASE_PATH) / run_path / "landmarks.npy"
        
        if not landmark_path.exists():
            return False, f"File not found: {landmark_path}"
        
        landmarks_dtype = get_dtype_without_loading(landmark_path)
        
        # Check if it's already float16
        if landmarks_dtype == np.dtype('float16'):
            return True, "Already float16"
        
        # Check if it's float32
        if landmarks_dtype != np.dtype('float32') and landmarks_dtype != np.dtype('float64'):
            return False, f"Unexpected dtype: {landmarks_dtype}"
        
        # Load the original file
        landmarks = np.load(landmark_path)
        
        # Convert to float16
        landmarks_float16 = landmarks.astype(np.float16)
        
        # Create temporary file path
        temp_path = landmark_path.with_suffix('.tmp')
        
        # Save to temporary file
        np.save(temp_path, landmarks_float16)
        
        temp_load_path = Path(str(temp_path) + '.npy')
        # Verify the saved file can be loaded correctly
        verification = np.load(temp_load_path)
        if verification.dtype != np.dtype('float16'):
            temp_load_path.unlink(missing_ok=True)
            return False, "Verification failed - saved file is not float16"
        
        # Replace original file with temporary file
        landmark_path.unlink()  # Delete original
        temp_load_path.rename(landmark_path)  # Rename temp to original
        
        return True, "Success"
        
    except Exception as e:
        # Clean up temporary file if it exists
        temp_path = landmark_path.with_suffix('.tmp')
        temp_load_path = Path(str(temp_path) + '.npy')
        if temp_load_path.exists():
            temp_load_path.unlink(missing_ok=True)
        return False, str(e)

def main():
    logger.info("Loading dataframe...")
    df = pd.read_pickle(DATAFRAME_PATH)
    
    # Get unique run_paths
    unique_run_paths = df['run_path'].unique()
    logger.info(f"Found {len(unique_run_paths)} unique run_paths")
    
    # Process files in parallel
    logger.info("Starting parallel conversion...")
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(convert_landmark_file)(run_path) for run_path in tqdm(unique_run_paths, desc="Converting landmarks")
    )
    
    # Count results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    logger.info(f"Conversion completed!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    # Show first few errors
    errors = [(run_path, msg) for (run_path, (success, msg)) in zip(unique_run_paths, results) if not success]
    if errors:
        logger.info("First 5 errors:")
        for run_path, msg in errors[:5]:
            logger.error(f"{run_path}: {msg}")

if __name__ == "__main__":
    main() 