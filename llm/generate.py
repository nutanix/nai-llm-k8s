"""
Downloads model files, generates Model Archive (MAR) 
and config.properties file
"""

import os
import argparse
import json
import sys
from typing import List
from huggingface_hub import snapshot_download
import utils.marsgen as mg
import utils.hf_utils as hf
import utils.tsutils as ts
from utils.generate_data_model import GenerateDataModel
from utils.system_utils import (
    check_if_path_exists,
    create_folder_if_not_exists,
    delete_directory,
    copy_file,
    check_if_folder_empty,
)

CONFIG_DIR = "config"
CONFIG_FILE = "config.properties"
MODEL_STORE_DIR = "model-store"
HANDLER = "handler.py"
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_config.json")
PREFERRED_MODEL_FORMATS = [".safetensors", ".bin"]  # In order of Preference
OTHER_MODEL_FORMATS = [
    "*.pt",
    "*.h5",
    "*.gguf",
    "*.msgpack",
    "*.tflite",
    "*.ot",
    "*.onnx",
]


def get_ignore_pattern_list(gen_model: GenerateDataModel) -> List[str]:
    """
    This method creates a list of file extensions to ignore from a priority list based on files
    present in the Hugging Face Repo. It filters out extensions not found in the repository and
    returns them as ignore patterns prefixed with '*' which is expected by Hugging Face client.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel class
    Returns:
        list(str): A list of patterns with '*' prepended to each extension,
                    suitable for filtering files.
    """
    repo_file_extensions = hf.get_repo_file_extensions(gen_model)
    for desired_extension in PREFERRED_MODEL_FORMATS:
        if desired_extension in repo_file_extensions:
            ignore_list = [
                "*" + ignore_extension
                for ignore_extension in PREFERRED_MODEL_FORMATS
                if ignore_extension != desired_extension
            ]
            ignore_list.extend(OTHER_MODEL_FORMATS)
            return ignore_list
    return []


def set_config(gen_model: GenerateDataModel) -> None:
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

    (
        initial_workers,
        batch_size,
        max_batch_delay,
        response_timeout,
    ) = ts.get_params_for_registration(gen_model.model_name)

    config_info = [
        "\ninstall_py_dep_per_model=true\n",
        "model_store=/mnt/models/model-store\n",
        f'model_snapshot={{"name":"startup.cfg","modelCount":1,'
        f'"models":{{"{gen_model.model_name}":{{'
        f'"{gen_model.repo_info.repo_version}":{{"defaultVersion":true,'
        f'"marName":"{gen_model.model_name}.mar",'
        f'"minWorkers":{initial_workers or 1},'
        f'"maxWorkers":{initial_workers or 1},'
        f'"batchSize":{batch_size or 1},"maxBatchDelay":{max_batch_delay or 500},'
        f'"responseTimeout":{response_timeout or 2000}}}}}}}}}',
    ]

    with open(config_file_path, "a", encoding="utf-8") as config_file:
        config_file.writelines(config_info)


def check_if_mar_file_exist(gen_model: GenerateDataModel) -> bool:
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


def read_config_for_download(gen_model: GenerateDataModel) -> None:
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

            if not gen_model.repo_info.repo_version:
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
            if not gen_model.repo_info.repo_version:
                gen_model.repo_info.repo_version = "1.0"
        elif gen_model.repo_info.repo_id:
            hf.hf_token_check(gen_model.repo_info.repo_id, gen_model.repo_info.hf_token)
            gen_model.repo_info.repo_version = hf.get_repo_commit_id(
                repo_id=gen_model.repo_info.repo_id,
                revision=gen_model.repo_info.repo_version,
                token=gen_model.repo_info.hf_token,
            )
            gen_model.is_custom = True
            if gen_model.mar_utils.handler_path == "":
                gen_model.mar_utils.handler_path = os.path.join(
                    os.path.dirname(__file__),
                    HANDLER,
                )
        else:
            print(
                "## If you want to create a model archive file for supported models, "
                "make sure you're model name is present in the below : "
            )
            print(list(models.keys()))
            print(
                "\nIf you want to create a model archive file for"
                " either a Custom Model or other HuggingFace models, "
                "refer to the official GPT-in-a-Box documentation: "
                "https://opendocs.nutanix.com/gpt-in-a-box/overview/"
            )
            sys.exit(1)


def run_download(gen_model: GenerateDataModel) -> GenerateDataModel:
    """
    This function checks if model files are present at given model path
    otherwise downloads the given version's model files at that path.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        GenerateDataModel: An instance of the GenerateDataModel class.
    """
    print("## Starting model files download\n")
    delete_directory(gen_model.mar_utils.model_path)
    create_folder_if_not_exists(gen_model.mar_utils.model_path)

    tmp_hf_cache = os.path.join(gen_model.mar_utils.model_path, "tmp_hf_cache")
    create_folder_if_not_exists(tmp_hf_cache)

    snapshot_download(
        repo_id=gen_model.repo_info.repo_id,
        revision=gen_model.repo_info.repo_version,
        local_dir=gen_model.mar_utils.model_path,
        token=gen_model.repo_info.hf_token,
        local_dir_use_symlinks=False,
        cache_dir=tmp_hf_cache,
        force_download=True,
        ignore_patterns=get_ignore_pattern_list(gen_model),
    )
    delete_directory(tmp_hf_cache)
    print("## Successfully downloaded model_files\n")
    return gen_model


def create_mar(gen_model: GenerateDataModel) -> None:
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
        if check_if_folder_empty(gen_model.mar_utils.model_path):
            print(
                f"\n##Error: {gen_model.model_name} model files for the custom"
                f" model not found in the provided path: {gen_model.mar_utils.model_path}"
            )
            sys.exit(1)

        print(
            f"\n## Generating MAR file for custom model files: {gen_model.model_name} \n"
        )
        create_folder_if_not_exists(gen_model.mar_utils.mar_output)

        mg.generate_mars(
            gen_model=gen_model,
            model_config=MODEL_CONFIG_PATH,
            model_store_dir=gen_model.mar_utils.mar_output,
            debug=gen_model.debug,
        )


def run_script(params: argparse.Namespace) -> bool:
    """
    Execute a series of steps to run a script for downloading model files,
    creating model archive file, and config file for a LLM.
    Args:
        params (argparse.Namespace): A Namespace object containing the necessary
                                    parameters and configurations for the script.

    Returns:
        None
    """
    gen_model = GenerateDataModel(params)
    read_config_for_download(gen_model)
    check_if_path_exists(gen_model.output, "output", is_dir=True)
    gen_model.set_model_files_and_mar(params)

    if gen_model.is_custom and not gen_model.mar_utils.model_path:
        print(
            "## Error: For HuggingFace models and Custom models model path should be set"
        )
        sys.exit(1)

    if gen_model.download_model:
        gen_model = run_download(gen_model)

    create_mar(gen_model)
    set_config(gen_model)
    return True


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
        "--repo_id",
        type=str,
        default="",
        metavar="ri",
        help="HuggingFace repository ID (In case of custom model download)",
    )
    parser.add_argument(
        "--skip_download", action="store_false", help="flag to not download"
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
        default=None,
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
