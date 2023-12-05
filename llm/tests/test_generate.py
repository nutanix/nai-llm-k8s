"""
This module runs pytest tests for generate.py file.

Attributes:
    MODEL_NAME: Name of the model used for testing (gpt2).
    OUTPUT: absolute path of the output location in local nfs mount.
    MODEL_CONFIG_PATH: Path to model_config.json file.
    MODEL_TEMP_CONFIG_PATH: Path to backup model_config.json file.
"""
import os
import argparse
import json
import shutil
import pytest
import generate
from utils.system_utils import copy_file

MODEL_NAME = "gpt2"
OUTPUT = os.path.dirname(__file__)
MODEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "model_config.json"
)
MODEL_TEMP_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "temp_model_config.json"
)


def set_args(
    model_name="",
    output="",
    model_path="",
    repo_version=None,
    handler_path="",
):
    """
    This function sets the arguments to run generate.py.

    Args:
        repo_version (str, optional): Repository version of the model. Defaults to "".
        model_path (str, optional): Path to model files. Defaults to MODEL_PATH.
        output (str, optional): absolute path of the output location in local nfs mount.
        handler_path (str, optional): Path to Torchserve handler. Defaults to "".

    Returns:
        argparse.Namespace: Parameters to run generate.py.
    """
    args = argparse.Namespace()
    args.model_name = model_name
    args.output = output
    args.model_path = model_path
    args.skip_download = True
    args.repo_version = repo_version
    args.repo_id = ""
    args.handler_path = handler_path
    args.hf_token = None
    args.debug = False
    return args


def empty_folder(folder_path):
    """
    This function empties a folder.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove all files in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Remove all subfolders in the folder
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    shutil.rmtree(subfolder_path)
        else:
            print(f"Folder '{folder_path}' does not exist.")
    except (FileNotFoundError, IsADirectoryError) as e:
        print(f"An error occurred: {str(e)}")


def custom_model_setup():
    """
    This function is used to setup custom model case.
    Returns:
        model_path: absolute path of model files
    """
    copy_file(MODEL_CONFIG_PATH, MODEL_TEMP_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump({}, file)

    repo_version = "11c5a3d5811f50298f278a704980280950aedb10"
    model_path = os.path.join(
        os.path.dirname(__file__), MODEL_NAME, repo_version, "download"
    )
    return model_path


def custom_model_restore():
    """
    This function restores the 'model_config.json' file.
    """
    os.remove(MODEL_CONFIG_PATH)
    copy_file(MODEL_TEMP_CONFIG_PATH, MODEL_CONFIG_PATH)
    os.remove(MODEL_TEMP_CONFIG_PATH)


def test_empty_model_name_failure():
    """
    This function tests empty model name.
    Expected result: Failure.
    """
    args = set_args(output=OUTPUT)
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_empty_output_failure():
    """
    This function tests empty output path.
    Expected result: Failure.
    """
    args = set_args(MODEL_NAME)
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_wrong_model_name_failure():
    """
    This function tests wrong model name.
    Expected result: Failure.
    """
    args = set_args("wrong_model_name", OUTPUT)
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_wrong_output_failure():
    """
    This function tests wrong output path.
    Expected result: Failure.
    """
    args = set_args(MODEL_NAME, "/wrong_output_path")
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_wrong_repo_version_failure():
    """
    This function tests wrong repo version.
    Expected result: Failure.
    """
    args = set_args(MODEL_NAME, OUTPUT, repo_version="wrong_repo_version")
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_wrong_handler_path_failure():
    """
    This function tests wrong handler path.
    Expected result: Failure.
    """
    args = set_args(MODEL_NAME, OUTPUT, handler_path="/wrong_path.py")
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_no_model_files_failure():
    """
    This function tests skip download without model files.
    Expected result: Failure.
    """
    args = set_args(MODEL_NAME, OUTPUT)
    args.skip_download = False
    try:
        generate.run_script(args)
    except SystemExit as e:
        assert e.code == 1
    else:
        assert False


def test_default_success():
    """
    This function tests the default GPT2 model.
    Expected result: Success.
    """
    args = set_args(MODEL_NAME, OUTPUT)
    try:
        result = generate.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def test_vaild_repo_version_success():
    """
    This function tests a valid repo version.
    Expected result: Success.
    """
    args = set_args(
        MODEL_NAME, OUTPUT, repo_version="e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
    )
    try:
        result = generate.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def test_short_repo_version_success():
    """
    This function tests a valid short repo version
    and if model and MAR file already exists.
    Expected result: Success.
    """
    args = set_args(MODEL_NAME, OUTPUT, repo_version="11c5a3d581")
    try:
        result = generate.run_script(args)
    except SystemExit:
        assert False
    else:
        assert result is True


def test_custom_model_with_modelfiles_success():
    """
    This function tests the custom model case.
    This is done by clearing the 'model_config.json' and
    generating the 'GPT2' MAR file.
    Expected result: Success.
    """
    model_path = custom_model_setup()
    args = set_args(MODEL_NAME, OUTPUT, model_path)
    args.skip_download = False
    try:
        result = generate.run_script(args)
        custom_model_restore()
    except SystemExit:
        assert False
    else:
        assert result is True


def test_custom_model_no_model_files_failure():
    """
    This function tests the custom model case when
    model files folder is empty.
    Expected result: Failure.
    """
    model_path = custom_model_setup()
    model_store_path = os.path.join(
        os.path.dirname(__file__), MODEL_NAME, "model-store"
    )
    empty_folder(model_path)
    empty_folder(model_store_path)
    args = set_args(MODEL_NAME, OUTPUT, model_path)
    args.skip_download = False
    try:
        generate.run_script(args)
    except SystemExit as e:
        custom_model_restore()
        assert e.code == 1
    else:
        assert False


def test_custom_model_with_repo_id_success():
    """
    This function tests the custom model case where
    model files are to be downloaded for provided
    repo ID.
    This is done by clearing the 'model_config.json' and
    generating the 'GPT2' MAR file.
    Expected result: Success.
    """
    model_path = custom_model_setup()
    args = set_args(MODEL_NAME, OUTPUT, model_path)
    args.repo_id = "gpt2"
    try:
        result = generate.run_script(args)
        custom_model_restore()
    except SystemExit:
        assert False
    else:
        assert result is True


def test_custom_model_wrong_repo_id_failure():
    """
    This function tests the custom model case when
    model repo ID is wrong.
    Expected result: Failure.
    """
    model_path = custom_model_setup()
    model_store_path = os.path.join(
        os.path.dirname(__file__), MODEL_NAME, "model-store"
    )
    empty_folder(model_path)
    empty_folder(model_store_path)
    args = set_args(MODEL_NAME, OUTPUT, model_path)
    args.repo_id = "wrong_repo_id"
    try:
        generate.run_script(args)
    except SystemExit as e:
        custom_model_restore()
        assert e.code == 1
    else:
        assert False


def test_custom_model_wrong_repo_version_failure():
    """
    This function tests the custom model case when
    model repo version is wrong.
    Expected result: Failure.
    """
    model_path = custom_model_setup()
    model_store_path = os.path.join(
        os.path.dirname(__file__), MODEL_NAME, "model-store"
    )
    empty_folder(model_path)
    empty_folder(model_store_path)
    args = set_args(MODEL_NAME, OUTPUT, model_path)
    args.repo_id = "gpt2"
    args.repo_version = "wrong_version"
    try:
        generate.run_script(args)
    except SystemExit as e:
        custom_model_restore()
        assert e.code == 1
    else:
        assert False


# Run the tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
