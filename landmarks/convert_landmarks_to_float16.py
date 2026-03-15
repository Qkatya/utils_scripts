#!/usr/bin/env python3
"""
Script to convert landmark files from float32 to float16 format.
Safely processes files with progress tracking and statistics.
Supports multiprocessing for large-scale conversions.
"""

import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from typing import List, Tuple
import multiprocessing as mp
from functools import partial
import time
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_landmark_files(root_path: Path) -> List[Path]:
    """
    Recursively find all landmarks.npy files in the directory tree.
    
    Args:
        root_path: Root directory to search
        
    Returns:
        List of paths to landmarks.npy files
    """
    landmark_files = []
    for file_path in root_path.rglob("landmarks.npy"):
        if file_path.is_file():
            landmark_files.append(file_path)
    return landmark_files

def convert_file_safely(file_path: Path) -> Tuple[bool, str]:
    """
    Safely convert a single landmarks.npy file from float32 to float16.
    
    Args:
        file_path: Path to the landmarks.npy file
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Load the original file
        landmarks = np.load(file_path)
        
        # Check if it's already float16
        if landmarks.dtype == np.dtype('float16'):
            return True, "Already float16"
        
        # Check if it's float32
        if landmarks.dtype != np.dtype('float32'):
            return False, f"Unexpected dtype: {landmarks.dtype}"
        
        # Convert to float16
        landmarks_float16 = landmarks.astype(np.float16)
        
        # Create temporary file path
        temp_path = file_path.with_suffix('.tmp')
        
        # Save to temporary file
        np.save(temp_path, landmarks_float16)
        
        temp_load_path = Path(str(temp_path) + '.npy')
        # Verify the saved file can be loaded correctly
        verification = np.load(temp_load_path)
        if verification.dtype != np.dtype('float16'):
            temp_load_path.unlink(missing_ok=True)
            return False, "Verification failed - saved file is not float16"
        
        # Replace original file with temporary file
        file_path.unlink()  # Delete original
        temp_load_path.rename(file_path)  # Rename temp to original
        
        return True, "Success"
        
    except Exception as e:
        # Clean up temporary file if it exists
        temp_path = file_path.with_suffix('.tmp')
        temp_load_path = Path(str(temp_path) + '.npy')
        if temp_load_path.exists():
            temp_load_path.unlink(missing_ok=True)
        return False, str(e)

def analyze_file_sample(file_path: Path) -> Tuple[str, str]:
    """
    Analyze a single file to determine its dtype.
    
    Args:
        file_path: Path to the landmarks.npy file
        
    Returns:
        Tuple of (dtype: str, error_message: str)
    """
    try:
        landmarks = np.load(file_path)
        return str(landmarks.dtype), ""
    except Exception as e:
        return "error", str(e)

def fast_dry_run(landmark_files: List[Path], sample_size: int = 1000) -> dict:
    """
    Perform a fast dry run by sampling files instead of loading all of them.
    
    Args:
        landmark_files: List of all landmark file paths
        sample_size: Number of files to sample for analysis
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Performing fast dry run with {sample_size} sample files...")
    
    # Sample files for analysis
    if len(landmark_files) <= sample_size:
        sample_files = landmark_files
    else:
        sample_files = random.sample(landmark_files, sample_size)
    
    # Analyze sample files
    dtype_counts = {}
    error_count = 0
    error_messages = []
    
    for i, file_path in enumerate(sample_files):
        if i % 100 == 0:  # Log progress every 100 files
            logger.info(f"Analyzing sample files: {i}/{len(sample_files)} ({i/len(sample_files)*100:.1f}%)")
        
        dtype, error = analyze_file_sample(file_path)
        if error:
            error_count += 1
            error_messages.append(f"{file_path}: {error}")
        else:
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    
    # Extrapolate results to full dataset
    total_files = len(landmark_files)
    sample_ratio = len(sample_files) / total_files
    
    extrapolated_results = {}
    for dtype, count in dtype_counts.items():
        extrapolated_results[dtype] = int(count / sample_ratio)
    
    extrapolated_results['error'] = int(error_count / sample_ratio)
    
    return {
        'total_files': total_files,
        'sample_size': len(sample_files),
        'dtype_counts': dtype_counts,
        'extrapolated_results': extrapolated_results,
        'error_messages': error_messages[:10],  # Keep first 10 errors
        'sample_ratio': sample_ratio
    }

def process_file_batch(file_batch: List[Path], dry_run: bool = False) -> Tuple[int, int, int, int, List[str]]:
    """
    Process a batch of files and return statistics.
    
    Args:
        file_batch: List of file paths to process
        dry_run: If True, only analyze files without converting
        
    Returns:
        Tuple of (successful_conversions, already_float16, failed_conversions, unexpected_dtype, error_messages)
    """
    successful_conversions = 0
    already_float16 = 0
    failed_conversions = 0
    unexpected_dtype = 0
    error_messages = []
    
    for file_path in file_batch:
        try:
            if dry_run:
                # Just analyze the file
                landmarks = np.load(file_path)
                if landmarks.dtype == np.dtype('float16'):
                    already_float16 += 1
                elif landmarks.dtype == np.dtype('float32'):
                    successful_conversions += 1
                else:
                    unexpected_dtype += 1
            else:
                # Actually convert the file
                success, message = convert_file_safely(file_path)
                if success:
                    if message == "Already float16":
                        already_float16 += 1
                    else:
                        successful_conversions += 1
                else:
                    failed_conversions += 1
                    error_messages.append(f"{file_path}: {message}")
        except Exception as e:
            failed_conversions += 1
            error_messages.append(f"{file_path}: {str(e)}")
    
    return successful_conversions, already_float16, failed_conversions, unexpected_dtype, error_messages

def chunk_list(lst: List, n: int) -> List[List]:
    """Split a list into n chunks."""
    avg = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end
    return result

def main():
    parser = argparse.ArgumentParser(description="Convert landmark files from float32 to float16")
    parser.add_argument("root_path", type=str, help="Root directory containing landmark files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--n-jobs", type=int, default=None, 
                       help="Number of parallel jobs (default: number of CPU cores)")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Number of files per batch for parallel processing")
    parser.add_argument("--sample-size", type=int, default=1000,
                       help="Number of files to sample for dry run analysis")
    args = parser.parse_args()
    
    root_path = Path(args.root_path)
    
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        return
    
    if not root_path.is_dir():
        logger.error(f"Path is not a directory: {root_path}")
        return
    
    # Set number of jobs
    n_jobs = args.n_jobs if args.n_jobs is not None else mp.cpu_count()
    logger.info(f"Using {n_jobs} parallel jobs")
    
    logger.info(f"Searching for landmark files in: {root_path}")
    start_time = time.time()
    
    # Find all landmark files
    landmark_files = find_landmark_files(root_path)
    
    if not landmark_files:
        logger.warning("No landmarks.npy files found!")
        return
    
    search_time = time.time() - start_time
    logger.info(f"Found {len(landmark_files)} landmark files in {search_time:.2f} seconds")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        
        # Perform fast dry run
        dry_run_results = fast_dry_run(landmark_files, args.sample_size)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("DRY RUN ANALYSIS RESULTS")
        logger.info("="*50)
        logger.info(f"Total files found: {dry_run_results['total_files']}")
        logger.info(f"Sample size: {dry_run_results['sample_size']}")
        logger.info(f"Sample ratio: {dry_run_results['sample_ratio']:.4f}")
        logger.info("\nSample Analysis:")
        for dtype, count in dry_run_results['dtype_counts'].items():
            logger.info(f"  {dtype}: {count} files")
        
        logger.info("\nExtrapolated Results (estimated):")
        extrapolated = dry_run_results['extrapolated_results']
        logger.info(f"  float32 (would be converted): {extrapolated.get('float32', 0):,}")
        logger.info(f"  float16 (already converted): {extrapolated.get('float16', 0):,}")
        logger.info(f"  Other dtypes: {sum(v for k, v in extrapolated.items() if k not in ['float32', 'float16', 'error']):,}")
        logger.info(f"  Errors: {extrapolated.get('error', 0):,}")
        
        if dry_run_results['error_messages']:
            logger.info(f"\nSample errors found:")
            for error in dry_run_results['error_messages']:
                logger.error(f"  {error}")
        
        logger.info("\nThis was a dry run. No files were modified.")
        logger.info("Run without --dry-run to perform actual conversion.")
        return
    
    # Statistics for actual conversion
    total_files = len(landmark_files)
    successful_conversions = 0
    failed_conversions = 0
    already_float16 = 0
    unexpected_dtype = 0
    all_error_messages = []
    
    logger.info("Starting conversion process...")
    
    # Process files in batches
    batches = chunk_list(landmark_files, max(1, len(landmark_files) // args.batch_size))
    
    with mp.Pool(processes=n_jobs) as pool:
        process_func = partial(process_file_batch, dry_run=False)
        
        # Process batches with progress logging
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} files)")
            result = process_func(batch)
            
            batch_success, batch_already, batch_failed, batch_unexpected, batch_errors = result
            
            successful_conversions += batch_success
            already_float16 += batch_already
            failed_conversions += batch_failed
            unexpected_dtype += batch_unexpected
            all_error_messages.extend(batch_errors)
            
            # Log progress
            processed = (i + 1) * len(batch)
            progress_pct = (processed / total_files) * 100
            logger.info(f"Progress: {processed:,}/{total_files:,} files ({progress_pct:.1f}%)")
            logger.info(f"  - Converted: {successful_conversions:,}")
            logger.info(f"  - Already float16: {already_float16:,}")
            logger.info(f"  - Failed: {failed_conversions:,}")
    
    total_time = time.time() - start_time
    
    # Print statistics
    logger.info("\n" + "="*50)
    logger.info("CONVERSION STATISTICS")
    logger.info("="*50)
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully converted: {successful_conversions}")
    logger.info(f"Already float16: {already_float16}")
    logger.info(f"Failed conversions: {failed_conversions}")
    logger.info(f"Unexpected dtype: {unexpected_dtype}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per file: {total_time/total_files:.4f} seconds")
    logger.info(f"Files per second: {total_files/total_time:.2f}")
    
    if all_error_messages:
        logger.info(f"\nFirst 10 errors:")
        for error in all_error_messages[:10]:
            logger.error(error)
        if len(all_error_messages) > 10:
            logger.info(f"... and {len(all_error_messages) - 10} more errors")
    
    logger.info(f"\nConversion completed!")
    logger.info(f"Successfully converted {successful_conversions} files to float16")

if __name__ == "__main__":
    main() 