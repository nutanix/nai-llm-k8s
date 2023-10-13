"""
System Utils
Utility functions to handle file and folder operations
"""
import os
import sys
import shutil


def check_if_path_exists(filepath, err="", is_dir=False):
    """
    check_if_path_exists
    This function checks if a given path exists.
    Args:
        filepath (str): Path to check.
        err (str, optional): Error message to print if path doesn't exists. Defaults to "".
        is_dir (bool, optional): Set to True if path is a directory, else False. Defaults to "".
    """
    if (not is_dir and not os.path.isfile(filepath)) or (
        is_dir and not os.path.isdir(filepath)
    ):
        print(f"Filepath does not exist {err} - {filepath}")
        sys.exit(1)


def create_folder_if_not_exists(path):
    """
    This function creates a folder in the specified path if it does not already exist

    Args:
        path (str): The path where the folder is to be created if it does not exist.
    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path}")


def delete_directory(directory_path):
    """
    This function deletes directory in the specified path

    Args:
        directory (str): The path of the directory that needs to be deleted.
    Raises:
        Exception: If any error occurs during deleting directory.
    Returns:
        None
    """
    if not os.path.exists(directory_path):
        return
    try:
        shutil.rmtree(directory_path)
        print(f"Deleted all contents from '{directory_path}'")
    except OSError as e:
        print(f"Error deleting contents from '{directory_path}': {str(e)}")


def copy_file(source_file, destination_file):
    """
    This function copies a file from source file path to destination file path

    Args:
        source_file (str): The path of the file that needs to be copied.
        destination_file (str): The path where the file is to be copied.
    Raises:
        Exception: If any error occurs during copying file.
    Returns:
        None
    """
    try:
        shutil.copy(source_file, destination_file)
    except OSError as e:
        print(f"## Error: {e}")
