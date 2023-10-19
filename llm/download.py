"""
Downloads model files, generates Model Archive (MAR) 
and config.properties file
"""
import os
import argparse
import json
import sys
import re
from collections import Counter
from huggingface_hub import snapshot_download
import utils.marsgen as mg
import utils.hfutils as hf
from utils.generate_data_model import GenerateDataModel
from utils.system_utils import (
    check_if_path_exists,
    create_folder_if_not_exists,
    delete_directory,
    copy_file,
    get_all_files_in_directory,
)

CONFIG_DIR = "config"
CONFIG_FILE = "config.properties"
MODEL_STORE_DIR = "model-store"
MODEL_FILES_LOCATION = "download"
HANDLER = "handler.py"
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")
FILE_EXTENSIONS_TO_IGNORE = [
    ".safetensors",
    ".safetensors.index.json",
    ".h5",
    ".ot",
    ".tflite",
    ".msgpack",
    ".onnx",
]


def get_ignore_pattern_list(extension_list):
    """
    This function takes a list of file extensions and returns a list of patterns
    that can be used to filter out files with these extensions.
    Args:
        extension_list (list): A list of file extensions.
    Returns:
        list: A list of patterns with '*' prepended to each extension, suitable for filtering files.
    """
    return ["*" + pattern for pattern in extension_list]


def compare_lists(list1, list2):
    """
    This function checks if two lists are equal by
    comparing their contents, regardless of the order.
    Args:
        list1 (list): The first list to compare.
        list2 (list): The second list to compare.

    Returns:
        bool: True if the lists have the same elements, False otherwise.
    """
    return Counter(list1) == Counter(list2)


def filter_files_by_extension(filenames, extensions_to_remove):
    """
    This function takes a list of filenames and a list
    of extensions to remove. It returns a new list of filenames
    after filtering out those with specified extensions.
    Args:
        filenames (list): A list of filenames to be filtered.
        extensions_to_remove (list): A list of file extensions to remove.
    Returns:
        list: A list of filenames after filtering.
    """
    pattern = "|".join([re.escape(suffix) + "$" for suffix in extensions_to_remove])
    # for the extensions in FILE_EXTENSIONS_TO_IGNORE
    # pattern will be '\.safetensors$|\.safetensors\.index\.json$'
    filtered_filenames = [
        filename for filename in filenames if not re.search(pattern, filename)
    ]
    return filtered_filenames


def set_values(params):
    """
    Set values for the GenerateDataModel object based on the command-line arguments.
    Args:
        params: An argparse.Namespace object containing command-line arguments.
    Returns:
        GenerateDataModel: An instance of the GenerateDataModel
                           class with values set based on the arguments.
    """
    gen_model = GenerateDataModel()

    gen_model.model_name = params.model_name
    gen_model.download_model = params.no_download
    gen_model.output = params.output
    gen_model.is_custom = False

    gen_model.mar_utils.handler_path = params.handler_path

    gen_model.repo_info.repo_version = params.repo_version
    gen_model.repo_info.hf_token = params.hf_token

    gen_model.debug = params.debug
    read_config_for_download(gen_model)
    check_if_path_exists(gen_model.output, "output", is_dir=True)

    get_model_files_and_mar(gen_model, params)

    return gen_model


def set_config(gen_model: GenerateDataModel):
    """
    This function creates a configuration file for the downloaded model and sets certain parameters.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        None
    """
    if gen_model.is_custom:
        model_spec_path = os.path.join(gen_model.output, gen_model.model_name)
    else:
        model_spec_path = os.path.join(
            gen_model.output, gen_model.model_name, gen_model.repo_info.repo_version
        )

    config_folder_path = os.path.join(model_spec_path, CONFIG_DIR)
    create_folder_if_not_exists(config_folder_path)

    source_config_file = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    config_file_path = os.path.join(config_folder_path, CONFIG_FILE)
    copy_file(source_config_file, config_file_path)

    check_if_path_exists(config_file_path, "Config")
    mar_filename = f"{gen_model.model_name}.mar"
    check_if_path_exists(
        os.path.join(model_spec_path, MODEL_STORE_DIR, mar_filename), "Model store"
    )  # Check if mar file exists

    config_info = [
        "\ninstall_py_dep_per_model=true\n",
        "model_store=/mnt/models/model-store\n",
        f'model_snapshot={{"name":"startup.cfg","modelCount":1,'
        f'"models":{{"{gen_model.model_name}":{{'
        f'"1.0":{{"defaultVersion":true,"marName":"{gen_model.model_name}.mar","minWorkers":1,'
        f'"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}}}}}',
    ]

    with open(config_file_path, "a", encoding="utf-8") as config_file:
        config_file.writelines(config_info)


def get_model_files_and_mar(gen_model: GenerateDataModel, params):
    """
    This function sets model path and mar output values.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
         params: An argparse.Namespace object containing command-line arguments.
    Returns:
        None
    """
    if gen_model.is_custom:
        gen_model.mar_utils.model_path = params.model_path
        gen_model.mar_utils.mar_output = os.path.join(
            gen_model.output,
            gen_model.model_name,
            MODEL_STORE_DIR,
        )
    else:
        gen_model.mar_utils.model_path = os.path.join(
            gen_model.output,
            gen_model.model_name,
            gen_model.repo_info.repo_version,
            MODEL_FILES_LOCATION,
        )
        gen_model.mar_utils.mar_output = os.path.join(
            gen_model.output,
            gen_model.model_name,
            gen_model.repo_info.repo_version,
            MODEL_STORE_DIR,
        )


def check_if_model_files_exist(gen_model: GenerateDataModel):
    """
    This function compares the list of files in the downloaded model
    directory with the list of files in the HuggingFace repository.
    It takes into account any files to ignore based on predefined extensions.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        bool: True if the downloaded model files match the expected
              repository files, False otherwise.
    """
    extra_files_list = get_all_files_in_directory(gen_model.mar_utils.model_path)
    repo_files = hf.get_repo_files_list(gen_model)
    repo_files = filter_files_by_extension(repo_files, FILE_EXTENSIONS_TO_IGNORE)
    return compare_lists(extra_files_list, repo_files)


def check_if_mar_file_exist(gen_model: GenerateDataModel):
    """
    This function checks if the Model Archive (MAR) file for the
    downloaded model exists in the specified output directory.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        bool: True if the MAR file exists, False otherwise.
    """
    mar_filename = f"{gen_model.model_name}.mar"
    if os.path.exists(gen_model.mar_utils.mar_output):
        directory_contents = os.listdir(gen_model.mar_utils.mar_output)
        return len(directory_contents) == 1 and directory_contents[0] == mar_filename

    return False


def read_config_for_download(gen_model: GenerateDataModel):
    """
    This function reads repo id, version and handler name
    from model_config.json and sets values for the GenerateDataModel object.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        None
    Raises:
        sys.exit(1): If model name is not valid, the function will
                     terminate the program with an exit code of 1.
    """
    check_if_path_exists(MODEL_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
        models = json.loads(f.read())
        if gen_model.model_name in models:
            # validation to check if model repo commit id is valid or not
            model = models[gen_model.model_name]
            gen_model.repo_info.repo_id = model["repo_id"]

            hf.hf_token_check(gen_model.repo_info.repo_id, gen_model.repo_info.hf_token)

            if gen_model.repo_info.repo_version == "":
                gen_model.repo_info.repo_version = model["repo_version"]

            gen_model.repo_info.repo_version = hf.get_repo_commit_id(
                repo_id=gen_model.repo_info.repo_id,
                revision=gen_model.repo_info.repo_version,
                token=gen_model.repo_info.hf_token,
            )

            if (
                gen_model.mar_utils.handler_path == ""
                and model.get("handler")
                and model["handler"]
            ):
                gen_model.mar_utils.handler_path = os.path.join(
                    os.path.dirname(__file__),
                    model["handler"],
                )
            check_if_path_exists(gen_model.mar_utils.handler_path, "Handler")
        elif not gen_model.download_model:
            gen_model.is_custom = True
            if gen_model.mar_utils.handler_path == "":
                gen_model.mar_utils.handler_path = os.path.join(
                    os.path.dirname(__file__),
                    HANDLER,
                )
            if gen_model.repo_info.repo_version == "":
                gen_model.repo_info.repo_version = "1.0"
        else:
            print(
                "## Please check your model name, it should be one of the following : "
            )
            print(list(models.keys()))
            sys.exit(1)


def run_download(gen_model: GenerateDataModel):
    """
    This function checks if model files are present at given model path
    otherwise downloads the given version's model files at that path.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        GenerateDataModel: An instance of the GenerateDataModel class.
    """
    if os.path.exists(gen_model.mar_utils.model_path) and check_if_model_files_exist(
        gen_model
    ):
        print(
            (
                "## Skipping downloading as model files of the needed"
                " repo version are already present\n"
            )
        )
        return gen_model
    print("## Starting model files download\n")
    delete_directory(gen_model.mar_utils.model_path)
    create_folder_if_not_exists(gen_model.mar_utils.model_path)
    snapshot_download(
        repo_id=gen_model.repo_info.repo_id,
        revision=gen_model.repo_info.repo_version,
        local_dir=gen_model.mar_utils.model_path,
        local_dir_use_symlinks=False,
        token=gen_model.repo_info.hf_token,
        ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE),
    )
    print("## Successfully downloaded model_files\n")
    return gen_model


def create_mar(gen_model: GenerateDataModel):
    """
    This function checks if the Model Archive (MAR) file for the downloaded
    model exists in the specified model path otherwise generates the MAR file.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        None
    """
    if check_if_mar_file_exist(gen_model):
        print("## Skipping generation of model archive file as it is present\n")
    else:
        check_if_path_exists(gen_model.mar_utils.model_path, "model_path", is_dir=True)
        if not gen_model.is_custom:
            if not check_if_model_files_exist(gen_model):
                # checking if local model files are same the repository files
                print("## Model files do not match HuggingFace repository Files")
                sys.exit(1)
        else:
            print(
                f"\n## Generating MAR file for custom model files: {gen_model.model_name}"
            )

        create_folder_if_not_exists(gen_model.mar_utils.mar_output)

        mg.generate_mars(
            gen_model=gen_model,
            model_config=MODEL_CONFIG_PATH,
            model_store_dir=gen_model.mar_utils.mar_output,
            debug=gen_model.debug,
        )


def run_script(params):
    """
    Execute a series of steps to run a script for downloading model files,
    creating model archive file, and config file for a LLM.
    Args:
        params (dict): A dictionary containing the necessary parameters
                     and configurations for the script.

    Returns:
        None
    """
    gen_model = set_values(params)
    if gen_model.download_model:
        gen_model = run_download(gen_model)

    create_mar(gen_model)
    set_config(gen_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        required=True,
        metavar="mn",
        help="name of the model",
    )
    parser.add_argument(
        "--no_download", action="store_false", help="flag to not download"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        metavar="mf",
        help="absolute path of the model files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        required=True,
        metavar="mx",
        help="absolute path of the output location in local nfs mount",
    )
    parser.add_argument(
        "--handler_path",
        type=str,
        default="",
        metavar="hp",
        help="absolute path of handler",
    )
    parser.add_argument(
        "--repo_version",
        type=str,
        default="",
        metavar="rv",
        help="commit id of the HuggingFace Repo",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        metavar="hft",
        help="HuggingFace Hub token to download LLAMA(2) models",
    )
    parser.add_argument("--debug", action="store_true", help="flag to debug")
    args = parser.parse_args()
    run_script(args)
