import os
import sys
import shutil

def check_if_path_exists(filepath, param = ""):
    if not os.path.exists(filepath):
        print(f"Filepath does not exist {param} - {filepath}")
        sys.exit(1)

def create_folder_if_not_exists(path):
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")

def delete_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    try:
        shutil.rmtree(directory_path)
        print(f"Deleted all contents from '{directory_path}'")
    except Exception as e:
        print(f"Error deleting contents from '{directory_path}': {str(e)}")

def copy_file(source_file, destination_file):
    try:
        shutil.copy(source_file, destination_file)
    except Exception as e:
        print(f"## Error: {e}")

"""
This function provides a list of file names in a directory
and its sub-directories

Args:
    path (str): The path to the directory.

Returns:
    ["file.txt", "sub-directory/file.txt"]
"""
def get_all_files_in_directory(path):
    output =[]
    for (dir_path, _, file_names) in os.walk(path):
        sub_dir = dir_path.removeprefix(path)
        if sub_dir:
            output.extend([f"{sub_dir}/{file_name}" for file_name in file_names])
        else:
            output.extend(file_names)
    
    return output
