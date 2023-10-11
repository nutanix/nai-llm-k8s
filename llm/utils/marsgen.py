import json
import os
import sys
import subprocess
from utils.system_utils import check_if_path_exists


def generate_mars(dl_model, model_config, model_store_dir, debug=False):
    debug and print(f"## Starting generate_mars, mar_config:{model_config}, model_store_dir:{model_store_dir}\n")
    cwd = os.getcwd()
    os.chdir(os.path.dirname(model_config))

    with open(model_config) as f:
        models = json.loads(f.read())
        if dl_model.model_name not in models:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)

        model = models[dl_model.model_name]

        handler = None
        if model.get("handler") and model["handler"]:
            handler = dl_model.handler_path
            check_if_path_exists(handler)

        extra_files = None
        extra_files_list = os.listdir(dl_model.model_path)
        extra_files_list = [os.path.join(dl_model.model_path, file) for file in extra_files_list]
        extra_files = ','.join(extra_files_list)

        requirements_file = None
        if model.get("requirements_file") and model["requirements_file"]:
            requirements_file = os.path.join(os.path.dirname(__file__), model["requirements_file"])
            check_if_path_exists(requirements_file)

        export_path = model_store_dir

        cmd = model_archiver_command_builder(model_name=dl_model.model_name,
                                             version=dl_model.repo_version,
                                             handler=handler,
                                             extra_files=extra_files,
                                             requirements_file=requirements_file,
                                             export_path=export_path,
                                             debug=debug)

        debug and print(f"## In directory: {os.getcwd()} | Executing command: {cmd}\n")

        try:
            subprocess.check_call(cmd, shell=True)
            marfile = "{}.mar".format(dl_model.model_name)
            print(f"## {marfile} is generated.\n")
        except subprocess.CalledProcessError as exc:
            print("## Creation failed !\n")
            debug and print("## {} creation failed !, error: {}\n".format(model["model_name"], exc))
            sys.exit(1)

    os.chdir(cwd)


def model_archiver_command_builder(model_name=None, version=None, model_file=None,
                                   handler=None, extra_files=None, runtime=None,
                                   archive_format=None, requirements_file=None,
                                   export_path=None, force=True, debug=False):
    cmd = "torch-model-archiver"
    if model_name:
        cmd += f" --model-name {model_name}"
    if version:
        cmd += f" --version {version}"
    if model_file:
        cmd += f" --model-file {model_file}"
    if handler:
        cmd += f" --handler {handler}"
    if extra_files:
        cmd += " --extra-files \"{0}\"".format(extra_files)
    if runtime:
        cmd += f" --runtime {runtime}"
    if archive_format:
        cmd += f" --archive-format {archive_format}"
    if requirements_file:
        cmd += f" --requirements-file {requirements_file}"
    if export_path:
        cmd += f" --export-path {export_path}"
    if force:
        cmd += " --force"
    print(f"\n## Generating mar file, will take few mins.\n")
    debug and print(cmd)
    return cmd
