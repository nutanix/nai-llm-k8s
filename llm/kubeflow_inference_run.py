import argparse
import sys
import os
import time
import json
import utils.tsutils as ts
from utils.system_utils import check_if_path_exists
from kubernetes import client, config
from kserve import KServeClient, constants, V1beta1PredictorSpec, V1beta1TorchServeSpec, V1beta1InferenceServiceSpec, V1beta1InferenceService

CONFIG_DIR = 'config'
CONFIG_FILE = 'config.properties'
MODEL_STORE_DIR = 'model-store'

kubMemUnits = ['Ei', 'Pi', 'Ti', 'Gi', 'Mi', 'Ki']

def get_inputs_from_folder(input_path):
    return [os.path.join(input_path, item) for item in os.listdir(input_path)] if input_path else []

def get_repo_version(model_name, mount_path):
    model_config_path = os.path.join(os.path.dirname(__file__), 'model_config.json')
    check_if_path_exists(model_config_path)
    with open(model_config_path) as f:
        models = json.loads(f.read())
        if model_name in models:
            repo_version = models[model_name]['repo_version']
    check_if_valid_version(model_name, repo_version, mount_path)
    return repo_version

def check_if_valid_version(model_name, repo_version, mount_path):
    model_spec_path = os.path.join(mount_path, model_name, repo_version)
    if not os.path.exists(model_spec_path):
        print(f"## ERROR: The {model_name} model files for given commit ID are not downloaded")
        sys.exit(1)

def create_pv(core_api, deploy_name, storage, nfs_server, nfs_path):
    # Create Persistent Volume
    persistent_volume = client.V1PersistentVolume(
        api_version='v1',
        kind='PersistentVolume',
        metadata=client.V1ObjectMeta(
            name=deploy_name, 
            labels={
                "storage": "nfs"
            }
        ),
        spec=client.V1PersistentVolumeSpec(
            capacity={
                "storage": storage
            },
            access_modes=["ReadWriteMany"],
            persistent_volume_reclaim_policy='Retain',
            nfs=client.V1NFSVolumeSource(
                path=nfs_path,
                server=nfs_server
            )
        )
    )

    core_api.create_persistent_volume(body=persistent_volume)


def create_pvc(core_api, deploy_name, storage):
    # Create Persistent Volume Claim
    persistent_volume_claim = client.V1PersistentVolumeClaim(
        api_version='v1',
        kind='PersistentVolumeClaim',
        metadata=client.V1ObjectMeta(
            name=deploy_name
        ),
        spec=client.V1PersistentVolumeClaimSpec(
            storage_class_name="",
            access_modes=["ReadWriteMany"],
            resources=client.V1ResourceRequirements(
                requests={
                    "storage": storage
                }
            ),
            selector=client.V1LabelSelector(
                match_labels={
                    "storage": "nfs"
                }
            )
        )
    )

    core_api.create_namespaced_persistent_volume_claim(body=persistent_volume_claim, namespace='default')


def create_isvc(deploy_name, model_name, repo_version, cpus, memory, gpus, model_params):
    storageuri = f'pvc://{deploy_name}/{model_name}/{repo_version}'
    default_model_spec = V1beta1InferenceServiceSpec(
        predictor=V1beta1PredictorSpec(
            pytorch=V1beta1TorchServeSpec(
                protocol_version='v2',
                storage_uri=storageuri,
                env=[
                    client.V1EnvVar(
                        name='TS_SERVICE_ENVELOPE',
                        value='body'
                    ),
                    client.V1EnvVar(
                        name='TS_NUMBER_OF_GPU',
                        value=str(gpus)
                    ),
                    client.V1EnvVar(
                        name='NAI_TEMPERATURE',
                        value=str(model_params["temperature"])
                    ),
                    client.V1EnvVar(
                        name='NAI_REP_PENALTY',
                        value=str(model_params["repetition_penalty"])
                    ),
                    client.V1EnvVar(
                        name='NAI_TOP_P',
                        value=str(model_params["top_p"])
                    ),
                    client.V1EnvVar(
                        name='NAI_MAX_TOKENS',
                        value=str(model_params["max_new_tokens"])
                    )
                ],
                resources=client.V1ResourceRequirements(
                    limits={
                        "cpu": cpus,
                        "memory": memory,
                        "nvidia.com/gpu": gpus
                    },
                    requests={
                        "cpu": cpus,
                        "memory": memory,
                        "nvidia.com/gpu": gpus
                    }
                )
            )
        )
    )

    isvc = V1beta1InferenceService(api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(name=deploy_name, namespace='default'),
        spec=default_model_spec)


    kserve = KServeClient(client_configuration=config.load_kube_config())
    kserve.create(isvc, watch=True)

def execute_inference_on_inputs(model_inputs, model_name, deploy_name):
    for input in model_inputs:    
        host = os.environ.get('INGRESS_HOST')
        port = os.environ.get('INGRESS_PORT')

        kserve = KServeClient(client_configuration=config.load_kube_config())
        obj=kserve.get(name=deploy_name, namespace='default')
        service_hostname = obj['status']['url'].split('/')[2:][0]
        headers = {"Content-Type": "application/json; charset=utf-8", "Host": service_hostname}

        response = ts.run_inference_v2(model_name, input, protocol="http", host=host, port=port, headers=headers)
        if response and response.status_code == 200:
            print(f"## Successfully ran inference on {model_name} model. \n\n Output - {response.text}\n\n")
        else:
            print(f"## Failed to run inference on {model_name} - model \n")
            sys.exit(1)

def execute(args):
    if not any(unit in args.mem for unit in kubMemUnits):
        print("container memory unit has to be one of", kubMemUnits)
        sys.exit(1)

    gpus = args.gpu
    cpus = args.cpu
    memory = args.mem
    nfs_server, nfs_path = args.nfs.split(':')
    deploy_name = args.deploy_name
    model_name = args.model_name
    input_path = args.data
    repo_version = args.repo_version
    mount_path = args.mount_path
    if not nfs_path or not nfs_server:
        print("NFS server and share path was not provided in accepted format - <address>:<share_path>")
        sys.exit(1)

    storage = '100Gi'

    model_params = ts.get_model_params(model_name)
    if not repo_version:
       repo_version = get_repo_version(model_name, mount_path)
    else:
       check_if_valid_version(model_name, repo_version, mount_path)

    config.load_kube_config()
    core_api = client.CoreV1Api()

    create_pv(core_api, deploy_name, storage, nfs_server, nfs_path)
    create_pvc(core_api, deploy_name, storage)
    create_isvc(deploy_name, model_name, repo_version, cpus, memory, gpus, model_params)

    print("wait for model registration to complete, will take some time")
    time.sleep(240)

    if input_path:
        check_if_path_exists(input_path, 'Input')
        model_inputs = get_inputs_from_folder(input_path)
        execute_inference_on_inputs(model_inputs, model_name, deploy_name)

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to generate the yaml.')

    # Add arguments
    parser.add_argument('--nfs', type=str, help='nfs ip address with mount path')
    parser.add_argument('--gpu', type=int, help='number of gpus')
    parser.add_argument('--cpu', type=int, help='number of cpus')
    parser.add_argument('--mem', type=str, help='memory required by the container')
    parser.add_argument('--model_name', type=str, help='name of the model to deploy')
    parser.add_argument('--deploy_name', type=str, help='name of the deployment')
    parser.add_argument('--data', type=str, help='data folder for the deployment validation')
    parser.add_argument('--repo_version', type=str, default=None, help='commit id of the HuggingFace Repo')
    parser.add_argument('--mount_path', type=str, help='local path to the nfs mount location')
    # Parse the command-line arguments
    args = parser.parse_args()
    execute(args)