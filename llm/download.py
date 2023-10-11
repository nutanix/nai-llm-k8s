import os
import argparse
import json
import sys
import re
from collections import Counter
from huggingface_hub import snapshot_download, HfApi
import utils.marsgen as mg
from utils.system_utils import (
    check_if_path_exists,
    create_folder_if_not_exists,
    delete_directory,
    copy_file
)

CONFIG_DIR = 'config'
CONFIG_FILE = 'config.properties'
MODEL_STORE_DIR = 'model-store'
MODEL_FILES_LOCATION = 'download'
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'model_config.json')
FILE_EXTENSIONS_TO_IGNORE = [".safetensors", ".safetensors.index.json"]

def get_ignore_pattern_list(extension_list):
    return ["*"+pattern for pattern in extension_list]

def compare_lists(list1, list2):
    return Counter(list1) == Counter(list2)

def filter_files_by_extension(filenames, extensions_to_remove):
    pattern = '|'.join([re.escape(suffix) + '$' for suffix in extensions_to_remove])
    # for the extensions in FILE_EXTENSIONS_TO_IGNORE
    # pattern will be '\.safetensors$|\.safetensors\.index\.json$'
    filtered_filenames = [filename for filename in filenames if not re.search(pattern, filename)]
    return filtered_filenames

class DownloadDataModel(object):
    model_name = str()
    download_model = bool()
    model_path=str()
    output = str()
    mar_output=str()
    repo_id = str()
    repo_version=str()
    handler_path = str()
    hf_token = str()
    debug = bool()


def set_values(args):
    dl_model = DownloadDataModel()
    dl_model.model_name = args.model_name
    dl_model.download_model = args.no_download
    dl_model.output = args.output
    dl_model.handler_path = args.handler_path
    dl_model.repo_version = args.repo_version
    dl_model.hf_token = args.hf_token
    dl_model.debug = args.debug
    get_repo_id_version_and_handler(dl_model)
    check_if_path_exists(dl_model.output, "output")
    dl_model.model_path = os.path.join(dl_model.output, dl_model.model_name,
                                       dl_model.repo_version, MODEL_FILES_LOCATION)
    dl_model.mar_output = os.path.join(dl_model.output, dl_model.model_name,
                                       dl_model.repo_version, MODEL_STORE_DIR)
    return dl_model


def set_config(dl_model):
    model_spec_path = os.path.join(dl_model.output, dl_model.model_name, dl_model.repo_version)
    config_folder_path = os.path.join(model_spec_path, CONFIG_DIR)
    create_folder_if_not_exists(config_folder_path)

    source_config_file = os.path.join(os.path.dirname(__file__), CONFIG_FILE)
    config_file_path = os.path.join(config_folder_path, CONFIG_FILE)
    copy_file(source_config_file, config_file_path)

    check_if_path_exists(config_file_path, 'Config')
    mar_filename = f"{dl_model.model_name}.mar"
    check_if_path_exists(os.path.join(model_spec_path, MODEL_STORE_DIR, mar_filename),
                         'Model store') # Check if mar file exists

    config_info = ['\ninstall_py_dep_per_model=true\n',
                   'model_store=/mnt/models/model-store\n',
                   f'model_snapshot={{"name":"startup.cfg","modelCount":1,"models":{{"{dl_model.model_name}":{{"1.0":{{"defaultVersion":true,"marName":"{dl_model.model_name}.mar","minWorkers":1,"maxWorkers":1,"batchSize":1,"maxBatchDelay":500,"responseTimeout":60}}}}}}}}']

    with open(config_file_path, "a") as config_file:
        config_file.writelines(config_info)


def check_if_model_files_exist(dl_model):
    extra_files_list = os.listdir(dl_model.model_path)
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files(repo_id=dl_model.repo_id, 
                                        revision=dl_model.repo_version, 
                                        token=dl_model.hf_token)
    repo_files = filter_files_by_extension(repo_files, FILE_EXTENSIONS_TO_IGNORE)
    return compare_lists(extra_files_list, repo_files)


def check_if_mar_file_exist(dl_model):
    mar_filename = f"{dl_model.model_name}.mar"
    if os.path.exists(dl_model.mar_output):
        directory_contents = os.listdir(dl_model.mar_output)
        return len(directory_contents) == 1 and directory_contents[0]==mar_filename
    else:
        return False


def get_repo_id_version_and_handler(dl_model):
    check_if_path_exists(MODEL_CONFIG_PATH)
    with open(MODEL_CONFIG_PATH) as f:
        models = json.loads(f.read())
        if dl_model.model_name in models:
            try: 
                # validation to check if model repo commit id is valid or not
                dl_model.repo_id = models[dl_model.model_name]['repo_id']
                if dl_model.repo_id.startswith("meta-llama") and dl_model.hf_token is None: # Make sure there is HF hub token for LLAMA(2)
                    print(("HuggingFace Hub token is required for llama download. Please specify it "
                            "using --hf_token=<your token>. Refer "
                            "https://huggingface.co/docs/hub/security-tokens"))
                    sys.exit(1)

                if  dl_model.repo_version == "":
                    dl_model.repo_version = models[dl_model.model_name]['repo_version']

                hf_api = HfApi()
                hf_api.list_repo_commits(repo_id=dl_model.repo_id, 
                                         revision=dl_model.repo_version, 
                                         token=dl_model.hf_token)
                
                if dl_model.handler_path == "":
                    dl_model.handler_path = os.path.join(os.path.dirname(__file__), models[dl_model.model_name]["handler"])
            except Exception as ex:
                print(f"## Error: Please check either repo_id or repo_version is not correct")
                sys.exit(1)
        else:
            print("## Please check your model name, it should be one of the following : ")
            print(list(models.keys()))
            sys.exit(1)


def run_download(dl_model):
    if os.path.exists(dl_model.model_path) and check_if_model_files_exist(dl_model):
        print("## Skipping downloading as model files of the needed repo version are already present\n")
        return dl_model
    else:
        print("## Starting model files download\n")
        delete_directory(dl_model.model_path)
        create_folder_if_not_exists(dl_model.model_path)
        snapshot_download(repo_id=dl_model.repo_id,
                        revision=dl_model.repo_version,
                        local_dir=dl_model.model_path,
                        local_dir_use_symlinks=False,
                        token=dl_model.hf_token,
                        ignore_patterns=get_ignore_pattern_list(FILE_EXTENSIONS_TO_IGNORE))
        print("## Successfully downloaded model_files\n")
        return dl_model


def create_mar(dl_model):
    if check_if_mar_file_exist(dl_model):
        print("## Skipping generation of model archive file as it is present\n")
    else:
        check_if_path_exists(dl_model.model_path, "model_path")
        if not check_if_model_files_exist(dl_model):  #checking if local model files are same the repository files
            print("## Model files do not match HuggingFace repository Files")
            sys.exit(1)

        create_folder_if_not_exists(dl_model.mar_output)

        mg.generate_mars(dl_model=dl_model,
                        model_config=MODEL_CONFIG_PATH,
                        model_store_dir=dl_model.mar_output,
                        debug=dl_model.debug)


def run_script(args):
    dl_model = set_values(args)
    if dl_model.download_model:
        dl_model = run_download(dl_model)

    create_mar(dl_model)
    set_config(dl_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download script')
    parser.add_argument('--model_name', type=str, default="", required=True,
                        metavar='mn', help='name of the model')
    parser.add_argument('--no_download', action='store_false',
                        help='flag to not download')
    parser.add_argument('--output', type=str, default="", required=True,
                        metavar='mx', help='absolute path of the output location in local nfs mount')
    parser.add_argument('--handler_path', type=str, default="",
                        metavar='hp', help='absolute path of handler')
    parser.add_argument('--repo_version', type=str, default="",
                        metavar='rv', help='commit id of the HuggingFace Repo')
    parser.add_argument('--hf_token', type=str, default=None,
                        metavar='hft', help='HuggingFace Hub token to download LLAMA(2) models')
    parser.add_argument('--debug', action='store_true',
                        help='flag to debug')
    args = parser.parse_args()
    run_script(args)
