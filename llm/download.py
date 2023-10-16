"""
Downloads model files, generates Model Archive (MAR) 
and config.properties file
"""
import os
import argparse
import json
import sys
import re
import dataclasses
from collections import Counter
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
import utils.marsgen as mg
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


@dataclasses.dataclass
class MarUtils:
    """
    A class for representing information about a Model Archive (MAR).

    Attributes:
        mar_output (str): The path to the MAR output directory.
        model_path (str): The path to the model directory.
        handler_path (str): The path to the model handler script.
    """

    mar_output = str()
    model_path = str()
    handler_path = str()


@dataclasses.dataclass
class RepoInfo:
    """
    A class for specifying details related to a model repository in HuggingFace.

    Attributes:
        repo_id (str): The identifier of the model repository.
        repo_version (str): The version of the model in the repository.
        hf_token (str): The Hugging Face token for authentication.
    """

    repo_id = str()
    repo_version = str()
    hf_token = str()


@dataclasses.dataclass
class DownloadDataModel:
    """
    A class representing a model download configuration for data retrieval.

    Attributes:
    - model_name (str): The name of the model to be downloaded.
    - download_model (bool): A boolean indicating whether to download
                             the model (True) or not (False).
    - model_path (str): The path where the downloaded model will be stored locally.
    - output (str): Mount path to the nfs server to be used in the kube
                    PV where model files and model archive file be stored.
    - mar_output (str): The output directory specifically for the Model Archive (MAR) files.
    - repository_info (dict): Dictionary that will contain the unique identifier
                              or name of the model repository, the version of the model
                              within the repository and the path to the model's handler
                              script for custom processing
    - hf_token (str): The Hugging Face API token for authentication when
                      accessing models from the Hugging Face Model Hub.
    - debug (bool): A boolean indicating whether to enable debugging mode for
                    the download process (True) or not (False).
    """

    model_name = str()
    download_model = bool()
    output = str()
    mar_utils = MarUtils()
    repo_info = RepoInfo()
    debug = bool()


def set_values(params):
    """
    Set values for the DownloadDataModel object based on the command-line arguments.
    Args:
        params: An argparse.Namespace object containing command-line arguments.
    Returns:
        DownloadDataModel: An instance of the DownloadDataModel
                           class with values set based on the arguments.
    """
    dl_model = DownloadDataModel()

    dl_model.model_name = params.model_name
    dl_model.download_model = params.no_download
    dl_model.output = params.output

    dl_model.mar_utils.handler_path = params.handler_path

    dl_model.repo_info.repo_version = params.repo_version
    dl_model.repo_info.hf_token = params.hf_token

    dl_model.debug = params.debug
    read_config_for_download(dl_model)
    check_if_path_exists(dl_model.output, "output", is_dir=True)

    dl_model.mar_utils.model_path = os.path.join(
        dl_model.output,
        dl_model.model_name,
        dl_model.repo_info.repo_version,
        MODEL_FILES_LOCATION,
    )
    dl_model.mar_utils.mar_output = os.path.join(
        dl_model.output,
        dl_model.model_name,
        dl_model.repo_info.repo_version,
        MODEL_STORE_DIR,
    )
    return dl_model


def set_config(dl_model):
    """
    This function creates a configuration file for the downloaded model and sets certain parameters.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        None
    """
    model_spec_path = os.path.join(
        dl_model.output, dl_model.model_name, dl_model.repo_info.repo_version
    )
    config_folder_path = os.path.join(model_spec_path, CONFIG_DIR)
    create_folder_if_not_exists(config_folder_path)

    source_config_file = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    config_file_path = os.path.join(config_folder_path, CONFIG_FILE)
    copy_file(source_config_file, config_file_path)

    check_if_path_exists(config_file_path, "Config")
    mar_filename = f"{dl_model.model_name}.mar"
    check_if_path_exists(
        os.path.join(model_spec_path, MODEL_STORE_DIR, mar_filename), "Model store"
    )  # Check if mar file exists

    config_info = [
        "\ninstall_py_dep_per_model=true\n",
        "model_store=/mnt/models/model-store\n",
        f'model_snapshot={{"name":"startup.cfg","modelCount":1,'
        f'"models":{{"{dl_model.model_name}":{{'
        f'"1.0":{{"defaultVersion":true,"marName":"{dl_model.model_name}.mar","minWorkers":1,'
        f'"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}}}}}',
    ]

    with open(config_file_path, "a", encoding="utf-8") as config_file:
        config_file.writelines(config_info)


def check_if_model_files_exist(dl_model):
    """
    This function compares the list of files in the downloaded model
    directory with the list of files in the HuggingFace repository.
    It takes into account any files to ignore based on predefined extensions.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        bool: True if the downloaded model files match the expected
              repository files, False otherwise.
    """
    extra_files_list = get_all_files_in_directory(dl_model.mar_utils.model_path)
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files(
        repo_id=dl_model.repo_info.repo_id,
        revision=dl_model.repo_info.repo_version,
        token=dl_model.repo_info.hf_token,
    )
    repo_files = filter_files_by_extension(repo_files, FILE_EXTENSIONS_TO_IGNORE)
    return compare_lists(extra_files_list, repo_files)


def check_if_mar_file_exist(dl_model):
    """
    This function checks if the Model Archive (MAR) file for the
    downloaded model exists in the specified output directory.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        bool: True if the MAR file exists, False otherwise.
    """
    mar_filename = f"{dl_model.model_name}.mar"
    if os.path.exists(dl_model.mar_utils.mar_output):
        directory_contents = os.listdir(dl_model.mar_utils.mar_output)
        return len(directory_contents) == 1 and directory_contents[0] == mar_filename

    return False


def read_config_for_download(dl_model):
    """
    This function reads repo id, version and handler name
    from model_config.json and sets values for the DownloadDataModel object.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        None
    Raises:
        sys.exit(1): If model name,repo_id or repo_version is not valid, the
                     function will terminate the program with an exit code of 1.
    """
    check_if_path_exists(MODEL_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
        models = json.loads(f.read())
        if dl_model.model_name in models:
            try:
                # validation to check if model repo commit id is valid or not
                model = models[dl_model.model_name]
                dl_model.repo_info.repo_id = model["repo_id"]
                if (
                    dl_model.repo_info.repo_id.startswith("meta-llama")
                    and dl_model.repo_info.hf_token is None
                ):
                    # Make sure there is HF hub token for LLAMA(2)
                    print(
                        (
                            "HuggingFace Hub token is required for llama download. "
                            "Please specify it using --hf_token=<your token>. Refer "
                            "https://huggingface.co/docs/hub/security-tokens"
                        )
                    )
                    sys.exit(1)

                if dl_model.repo_info.repo_version == "":
                    dl_model.repo_info.repo_version = model["repo_version"]

                hf_api = HfApi()
                commit_info = hf_api.list_repo_commits(
                    repo_id=dl_model.repo_info.repo_id,
                    revision=dl_model.repo_info.repo_version,
                    token=dl_model.repo_info.hf_token,
                )
                dl_model.repo_info.repo_version = commit_info[0].commit_id

                if (
                    dl_model.mar_utils.handler_path == ""
                    and model.get("handler")
                    and model["handler"]
                ):
                    dl_model.mar_utils.handler_path = os.path.join(
                        os.path.dirname(__file__),
                        model["handler"],
                    )
                check_if_path_exists(dl_model.mar_utils.handler_path, "Handler")
            except (RepositoryNotFoundError, RevisionNotFoundError, KeyError):
                print(
                    (
                        "## Error: Please check either repo_id, repo_version "
                        "or huggingface token is not correct"
                    )
                )
                sys.exit(1)
        else:
            print(
                "## Please check your model name, it should be one of the following : "
            )
            print(list(models.keys()))
            sys.exit(1)


def run_download(dl_model):
    """
    This function checks if model files are present at given model path
    otherwise downloads the given version's model files at that path.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        DownloadDataModel: An instance of the DownloadDataModel class.
    """
    if os.path.exists(dl_model.mar_utils.model_path) and check_if_model_files_exist(
        dl_model
    ):
        print(
            (
                "## Skipping downloading as model files of the needed"
                " repo version are already present\n"
            )
        )
        return dl_model
    print("## Starting model files download\n")
    delete_directory(dl_model.mar_utils.model_path)
    create_folder_if_not_exists(dl_model.mar_utils.model_path)
    snapshot_download(
        repo_id=dl_model.repo_info.repo_id,
        revision=dl_model.repo_info.repo_version,
        local_dir=dl_model.mar_utils.model_path,
        local_dir_use_symlinks=False,
        token=dl_model.repo_info.hf_token,
        ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE),
    )
    print("## Successfully downloaded model_files\n")
    return dl_model


def create_mar(dl_model):
    """
    This function checks if the Model Archive (MAR) file for the downloaded
    model exists in the specified model path otherwise generates the MAR file.
    Args:
        dl_model (DownloadDataModel): An instance of the DownloadDataModel
                                      class with relevant information.
    Returns:
        None
    """
    if check_if_mar_file_exist(dl_model):
        print("## Skipping generation of model archive file as it is present\n")
    else:
        check_if_path_exists(dl_model.mar_utils.model_path, "model_path", is_dir=True)
        if not check_if_model_files_exist(dl_model):
            # checking if local model files are same the repository files
            print("## Model files do not match HuggingFace repository Files")
            sys.exit(1)

        create_folder_if_not_exists(dl_model.mar_utils.mar_output)

        mg.generate_mars(
            dl_model=dl_model,
            model_config=MODEL_CONFIG_PATH,
            model_store_dir=dl_model.mar_utils.mar_output,
            debug=dl_model.debug,
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
    dl_model = set_values(params)
    if dl_model.download_model:
        dl_model = run_download(dl_model)

    create_mar(dl_model)
    set_config(dl_model)


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
