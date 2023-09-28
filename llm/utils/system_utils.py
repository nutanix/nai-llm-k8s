import os
import sys

def check_if_path_exists(filepath, param = ""):
    if not os.path.exists(filepath):
        print(f"Filepath does not exist {param} - {filepath}")
        sys.exit(1)

def create_folder_if_not_exits(path):
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")