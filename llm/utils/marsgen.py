"""
Generate a Model Archive (MAR) file for a specified LLM.
"""
import json
import os
import sys
import subprocess
import threading
import time
from typing import Dict, List
import tqdm
from utils.system_utils import check_if_path_exists, get_all_files_in_directory
from utils.generate_data_model import GenerateDataModel

REQUIREMENTS_FILE = "model_requirements.txt"


def monitor_marfile_size(
    file_path: str, approx_marfile_size: float, stop_monitoring: threading.Event
) -> None:
    """
    Monitor the generation of a Model Archive File and display progress.

    Args:
        file_path (str): The path to the Model Archive File.
        approx_marfile_size (float): The approximate size of the Model Archive File in bytes.
        stop_monitoring (EVENT): event which states when to stop monitoring
    Return:
        None
    """
    print("Model Archive File is Generating...\n")
    previous_file_size = 0
    progress_bar = tqdm.tqdm(
        total=approx_marfile_size,
        unit="B",
        unit_scale=True,
        desc="Creating Model Archive",
    )
    while not stop_monitoring.is_set():
        try:
            current_file_size = os.path.getsize(file_path)
        except FileNotFoundError:
            current_file_size = 0
        size_change = current_file_size - previous_file_size
        previous_file_size = current_file_size
        progress_bar.update(size_change)
        time.sleep(2)
    progress_bar.update(approx_marfile_size - current_file_size)
    progress_bar.close()
    print(
        f"\nModel Archive file size: {os.path.getsize(file_path) / (1024 ** 3):.2f} GB\n"
    )


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


def generate_mars(
    gen_model: GenerateDataModel,
    model_config: str,
    model_store_dir: str,
    debug: bool = False,
) -> None:
    """
    This function generates a Model Archive (MAR) file for a specified LLM using
    the provided model configuration, model store directory, and optional debug information.

    Args:
        gen_model (LLM): An object representing the LLM to generate the MAR for.
        model_config (str): The path to the JSON model configuration file.
        model_store_dir (str): The directory where the MAR file will be stored.
        debug (bool, optional): A flag indicating whether to
                                print debug information (default is False).

    Returns:
        None
    """
    if debug:
        print(
            (
                f"## Starting generate_mars, mar_config:{model_config}, "
                f"model_store_dir:{model_store_dir}\n"
            )
        )

    cwd = os.getcwd()
    os.chdir(os.path.dirname(model_config))

    with open(model_config, encoding="utf-8") as f:
        models = json.loads(f.read())
        if gen_model.model_name not in models:
            if not gen_model.is_custom:
                print(
                    "## Please check your model name, it should be one of the following : "
                )
                print(list(models.keys()))
                sys.exit(1)

        files_path = {}
        files_path["extra_files"] = None
        extra_files_list = get_all_files_in_directory(gen_model.mar_utils.model_path)
        extra_files_list = [
            os.path.join(gen_model.mar_utils.model_path, file)
            for file in extra_files_list
        ]
        files_path["extra_files"] = ",".join(extra_files_list)

        files_path["requirements_file"] = os.path.join(
            os.path.dirname(__file__), REQUIREMENTS_FILE
        )
        check_if_path_exists(files_path["requirements_file"])

        model_archiver_args = {
            "model_name": gen_model.model_name,
            "version": gen_model.repo_info.repo_version,
            "handler": gen_model.mar_utils.handler_path,
            "extra_files": files_path["extra_files"],
            "requirements_file": files_path["requirements_file"],
            "export_path": model_store_dir,
        }
        cmd = model_archiver_command_builder(
            model_archiver_args=model_archiver_args,
            debug=debug,
        )

        if debug:
            print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

        try:
            stop_monitoring = threading.Event()
            approx_marfile_size = get_files_sizes(extra_files_list) / 1.15
            mar_size_thread = threading.Thread(
                target=monitor_marfile_size,
                args=(
                    os.path.join(model_store_dir, f"{gen_model.model_name}.mar"),
                    approx_marfile_size,
                    stop_monitoring,
                ),
            )
            mar_size_thread.start()
            subprocess.check_call(cmd, shell=True)
            stop_monitoring.set()
            mar_size_thread.join()
            print(f"## {gen_model.model_name}.mar is generated.\n")
        except subprocess.CalledProcessError as exc:
            print("## Creation failed !\n")
            if debug:
                print(f"## {gen_model.model_name} creation failed !, error: {exc}\n")
            sys.exit(1)

    os.chdir(cwd)


def model_archiver_command_builder(
    model_archiver_args: Dict[str, str],
    runtime: int = None,
    archive_format: str = None,
    force: bool = True,
    debug: bool = False,
) -> str:
    """
    This function generates the torch model archiver command that
    will be used for generating model archive file
    Args:
        model_archiver_args(dict):Dictionary containing the following:
            model-name(str):Exported model name. Exported file will be named as
                            model-name.mar and saved in current working directory
                            if no --export-path is specified, else it will be
                            saved under the export path

            handler(str):TorchServe's default handler name  or handler python
                        file path to handle custom TorchServe inference logic.

            export-path(str):Path where the exported .mar file will be saved. This
                            is an optional parameter. If --export-path is not
                            specified, the file will be saved in the current
                            working directory.

            extra-files(str):Comma separated path to extra dependency files.

            version(str):Model's version.

            requirements-file(str): Path to requirements.txt file
                                    containing a list of model specific python
                                    packages to be installed by TorchServe for
                                    seamless model serving.

        runtime(int, optional): The runtime specifies which language to run your
                                inference code on. The default runtime is
                                RuntimeType.PYTHON. At the present moment we support
                                the following runtimes python, python3

        archive-format(str, optional):"default": This creates the model-archive
                                                 in <model-name>.mar format.
                                                 This is the default archiving format.
                                                 Models archived in this format
                                                 will be readily hostable on TorchServe.

        force(flag):When the --force flag is specified, an existing
                    .mar file with same name as that provided in --model-
                    name in the path specified by --export-path will
                    overwritten

    Returns:
        str: torch model archiver command
    """
    cmd = "torch-model-archiver"
    if model_archiver_args["model_name"]:
        cmd += f" --model-name {model_archiver_args['model_name']}"
    if model_archiver_args["version"]:
        cmd += f" --version {model_archiver_args['version']}"
    if model_archiver_args["handler"]:
        cmd += f" --handler {model_archiver_args['handler']}"
    if model_archiver_args["extra_files"]:
        cmd += f" --extra-files \"{model_archiver_args['extra_files']}\""
    if runtime:
        cmd += f" --runtime {runtime}"
    if archive_format:
        cmd += f" --archive-format {archive_format}"
    if model_archiver_args["requirements_file"]:
        cmd += f" --requirements-file {model_archiver_args['requirements_file']}"
    if model_archiver_args["export_path"]:
        cmd += f" --export-path {model_archiver_args['export_path']}"
    if force:
        cmd += " --force"
    print("## Generating MAR file, will take few mins.\n")
    if debug:
        print(cmd)
    return cmd
