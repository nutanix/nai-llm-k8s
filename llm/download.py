import os
import argparse
import json
import sys
from huggingface_hub import snapshot_download
import utils.marsgen as mg
from utils.system_utils import check_if_path_exists, create_folder_if_not_exits, delete_directory, copy_file

CONFIG_DIR = 'config'
CONFIG_FILE = 'config.properties'
MODEL_STORE_DIR = 'model-store'
MODEL_FILES_LOCATION = 'download'


def get_ignore_pattern_list(extension_list):
    return ["*"+pattern for pattern in extension_list]

class DownloadDataModel(object):
    model_name = str()
    download_model = bool()
    model_path=str()
    output = str()
    mar_output=str()
    repo_id = str()
    handler_path = str()
    hf_token = str()
    debug = bool()


def set_values(args):
    dl_model = DownloadDataModel()
    dl_model.model_name = args.model_name
    dl_model.download_model = args.no_download
    dl_model.output = args.output
    dl_model.handler_path = args.handler_path
    dl_model.hf_token = args.hf_token
    dl_model.debug = args.debug
    return dl_model


def run_download(dl_model):
    check_if_path_exists(dl_model.model_path, "model_path")
    mar_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    check_if_path_exists(mar_config_path)

    with open(mar_config_path) as f:
        models = json.loads(f.read())
        if dl_model.model_name in models:
            dl_model.repo_id = models[dl_model.model_name]['repo_id']
        else:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)
    
    if dl_model.repo_id.startswith("meta-llama") and dl_model.hf_token is None: # Make sure there is HF hub token for LLAMA(2)
        print(f"HuggingFace Hub token is required for llama download. Please specify it using --hf_token=<your token>. Refer https://huggingface.co/docs/hub/security-tokens")
        sys.exit(1)

    
    print("## Starting model files download\n")
    snapshot_download(repo_id=dl_model.repo_id,
                      local_dir=dl_model.model_path,
                      local_dir_use_symlinks=False,
                      token=dl_model.hf_token,
                      ignore_patterns=get_ignore_pattern_list(mg.FILE_EXTENSIONS_TO_IGNORE))
    print("## Successfully downloaded model_files\n")
    return dl_model


def create_mar(dl_model):
    check_if_path_exists(dl_model.model_path, "model_path")
    check_if_path_exists(dl_model.mar_output, "mar_output")
    mar_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    check_if_path_exists(mar_config_path)
    if dl_model.handler_path == "":
        with open(mar_config_path) as f:
            models = json.loads(f.read())
            if dl_model.model_name in models:
                dl_model.handler_path = os.path.join(os.path.dirname(__file__), models[dl_model.model_name]["handler"])
                dl_model.repo_id = models[dl_model.model_name]['repo_id']

    mg.generate_mars(dl_model=dl_model, 
                     mar_config=mar_config_path,
                     model_store_dir=dl_model.mar_output,
                     debug=dl_model.debug)


def set_config(model_name, mount_path):
    model_spec_path = os.path.join(mount_path, model_name)
    config_folder_path = os.path.join(model_spec_path, CONFIG_DIR)
    delete_directory(config_folder_path)
    create_folder_if_not_exits(config_folder_path)

    src_config_file = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    config_file_path = os.path.join(config_folder_path, CONFIG_FILE)
    copy_file(src_config_file, config_file_path)

    check_if_path_exists(config_file_path, 'Config')
    check_if_path_exists(os.path.join(model_spec_path, MODEL_STORE_DIR, model_name+'.mar'), 'Model store') # Check if mar file exists

    config_info = ['\ninstall_py_dep_per_model=true\n', 'model_store=/mnt/models/model-store\n','model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"'+model_name+'":{"1.0":{"defaultVersion":true,"marName":"'+model_name+'.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}']

    with open(config_file_path, "a") as config_file:
        config_file.writelines(config_info)



def run_script(args):
    dl_model = set_values(args)
    check_if_path_exists(dl_model.output, "output")
    path = os.path.join(dl_model.output, dl_model.model_name, MODEL_FILES_LOCATION)
    dl_model.model_path = path
    if dl_model.download_model:
        delete_directory(dl_model.model_path)
        create_folder_if_not_exits(dl_model.model_path)
        dl_model = run_download(dl_model)
    
    path = os.path.join(dl_model.output, dl_model.model_name, MODEL_STORE_DIR)
    create_folder_if_not_exits(path)
    dl_model.mar_output = path
    create_mar(dl_model)

    set_config(dl_model.model_name, dl_model.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download script')
    parser.add_argument('--model_name', type=str, default="", required=True,
                        metavar='mn', help='name of the model')
    parser.add_argument('--no_download', action='store_false',
                        help='flag to not download')
    parser.add_argument('--output', type=str, default="",
                        metavar='mx', help='absolute path of the output location in local nfs mount')
    parser.add_argument('--handler_path', type=str, default="",
                        metavar='hp', help='absolute path of handler')
    parser.add_argument('--hf_token', type=str, default=None,
                        metavar='hft', help='HuggingFace Hub token to download LLAMA(2) models')
    parser.add_argument('--debug', action='store_true',
                        help='flag to debug')
    args = parser.parse_args()
    run_script(args)