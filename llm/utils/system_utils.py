import os
import sys
import shutil

def check_if_path_exists(filepath, param = ""):
    if not os.path.exists(filepath):
        print(f"Filepath does not exist {param} - {filepath}")
        sys.exit(1)

def create_folder_if_not_exits(path):
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")

def delete_all_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    try:
        shutil.rmtree(directory_path)
        print(f"Deleted all contents from '{directory_path}'")
    except Exception as e:
        print(f"Error deleting contents from '{directory_path}': {str(e)}")
