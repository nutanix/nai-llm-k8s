"""
Utility functions to handle file and folder operations
"""
import os
import sys
import shutil
from pathlib import Path
from typing import List


def check_if_path_exists(filepath: str, err: str = "", is_dir: bool = False) -> None:
    """
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


def create_folder_if_not_exists(path: str) -> None:
    """
    This function creates a folder in the specified path if it does not already exist

    Args:
        path (str): The path where the folder is to be created if it does not exist.
    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    print(f"The new directory is created! - {path} \n")


def delete_directory(directory_path: str) -> None:
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
        print(f"Deleted all contents from '{directory_path}' \n")
    except OSError as e:
        print(f"Error deleting contents from '{directory_path}': {str(e)}")


def copy_file(source_file: str, destination_file: str) -> None:
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


def get_all_files_in_directory(directory: str) -> List[str]:
    """
    This function provides a list of file names in a directory
    and its sub-directories
    Args:
        directory (str): The path to the directory.
    Returns:
        ["file.txt", "sub-directory/file.txt"]
    """
    output = []
    directory_path = Path(directory)
    output = [
        str(file.relative_to(directory_path))
        for file in directory_path.rglob("*")
        if file.is_file()
    ]
    return output


def check_if_folder_empty(path: str) -> bool:
    """
    This function checks if a directory is empty.
    Args:
        path (str): Path of the dirctory to check.
    Returns:
        bool: True if directory is empty, False otherwise.
    """
    dir_items = os.listdir(path)
    return len(dir_items) == 0


def get_files_sizes(file_paths: List[str]) -> float:
    """
    Calculate the total size of the specified files.

    Args:
        file_paths (list): A list of file paths for which the sizes should be calculated.

    Returns:
        total_size (float): The sum of sizes (in bytes) of all the specified files.
    """
    total_size = 0

    for file_path in file_paths:
        try:
            size = os.path.getsize(file_path)
            total_size += size
        except FileNotFoundError:
            print(f"File not found: {file_path}")

    return total_size
