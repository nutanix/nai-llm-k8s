"""
Marsgen
Generate a Model Archive (MAR) file for a specified LLM.
"""
import json
import os
import sys
import subprocess
from utils.system_utils import check_if_path_exists


def generate_mars(dl_model, model_config, model_store_dir, debug=False):
    """
    This function generates a Model Archive (MAR) file for a specified LLM using
    the provided model configuration, model store directory, and optional debug information.

    Args:
        dl_model (LLM): An object representing the LLM to generate the MAR for.
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
        if dl_model.model_name not in models:
            print(
                "## Please check your model name, it should be one of the following : "
            )
            print(list(models.keys()))
            sys.exit(1)

        model = models[dl_model.model_name]

        extra_files = None
        extra_files_list = os.listdir(dl_model.mar_utils.model_path)
        extra_files_list = [
            os.path.join(dl_model.mar_utils.model_path, file)
            for file in extra_files_list
        ]
        extra_files = ",".join(extra_files_list)

        requirements_file = None
        if model.get("requirements_file") and model["requirements_file"]:
            requirements_file = os.path.join(
                os.path.dirname(__file__), model["requirements_file"]
            )
            check_if_path_exists(requirements_file)

        model_archiver_args = {
            "model_name": dl_model.model_name,
            "version": dl_model.repo_info.repo_version,
            "handler": dl_model.mar_utils.handler_path,
            "extra_files": extra_files,
            "requirements_file": requirements_file,
            "export_path": model_store_dir,
        }
        cmd = model_archiver_command_builder(
            model_archiver_args=model_archiver_args,
            debug=debug,
        )

        if debug:
            print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

        try:
            subprocess.check_call(cmd, shell=True)
            marfile = f"{dl_model.model_name}.mar"
            print(f"## {marfile} is generated.\n")
        except subprocess.CalledProcessError as exc:
            print("## Creation failed !\n")
            if debug:
                print(f"## {model['model_name']} creation failed !, error: {exc}\n")
            sys.exit(1)

    os.chdir(cwd)


def model_archiver_command_builder(
    model_archiver_args,
    runtime=None,
    archive_format=None,
    force=True,
    debug=False,
):
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
    print("\n## Generating mar file, will take few mins.\n")
    if debug:
        print(cmd)
    return cmd
