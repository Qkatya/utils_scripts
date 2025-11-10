#!/usr/bin/env python3
"""
Script to copy the current repository to a new folder with a project name.
Excludes venv* directories, .md files, and .ipynb notebooks.
"""

import os
import sys
import shutil
import argparse
import subprocess
import datetime
from pathlib import Path


def should_exclude(path, name):
    """Check if a file or directory should be excluded from copying."""
    # Exclude venv* directories
    if os.path.isdir(path) and name.startswith('venv'):
        return True
    
    # # Exclude .git directory
    # if os.path.isdir(path) and name == '.git':
    #     return True
    
    # Exclude Python cache directories
    if os.path.isdir(path) and name in ['__pycache__', '.pytest_cache']:
        return True
    
    # Exclude Python cache files
    if os.path.isfile(path) and name.endswith('.pyc'):
        return True
    
    # Exclude .md files
    if os.path.isfile(path) and name.endswith('.md'):
        return True
    
    # Exclude .ipynb files
    if os.path.isfile(path) and name.endswith('.ipynb'):
        return True
    
    return False


def copy_repository(source_dir, target_dir):
    """Copy the repository excluding specified files and directories."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying repository from {source_path} to {target_path}")
    
    for item in source_path.iterdir():
        item_name = item.name
        
        # Skip if item should be excluded
        if should_exclude(item, item_name):
            print(f"Excluding: {item_name}")
            continue
        
        # Skip the target directory if it's inside the source
        if item.resolve() == target_path.resolve():
            print(f"Skipping target directory: {item_name}")
            continue
        
        target_item = target_path / item_name
        
        try:
            if item.is_dir():
                print(f"Copying directory: {item_name}")
                shutil.copytree(item, target_item, dirs_exist_ok=True)
            else:
                print(f"Copying file: {item_name}")
                shutil.copy2(item, target_item)
        except Exception as e:
            print(f"Error copying {item_name}: {e}")
    
    print(f"Repository copied successfully to: {target_path}")


def get_git_info():
    """Get current git branch and commit hash."""
    try:
        # Get current branch
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                       text=True, stderr=subprocess.DEVNULL).strip()
        
        # Get current commit hash (short)
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], 
                                            text=True, stderr=subprocess.DEVNULL).strip()
        
        return branch, commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git", "no-commit"


def main():
    parser = argparse.ArgumentParser(description='Copy repository to a new folder with project name')
    parser.add_argument('project_name', help='Name of the project (will be appended to q-auto-avst-)')
    
    args = parser.parse_args()
    
    # Get current directory (repository root)
    current_dir = os.getcwd()
    
    # Get git information
    branch, commit_hash = get_git_info()
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create target directory name
    target_dir_name = f"{os.path.basename(current_dir)}-{args.project_name}-{branch}-{commit_hash}-{timestamp}"
    
    # Get parent directory of current directory
    parent_dir = os.path.dirname(current_dir)
    
    # Create full target path
    target_dir = os.path.join(parent_dir, target_dir_name)
    
    # Check if target directory already exists
    if os.path.exists(target_dir):
        response = input(f"Directory {target_dir} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)
        else:
            print(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
    
    # Copy the repository
    copy_repository(current_dir, target_dir)


if __name__ == "__main__":
    main() 