"""
Orchestrates the deployment and inference of a LLM 
in a Kubernetes cluster by performing tasks such as creating 
persistent storage, registering the model, and running inference.
"""
import argparse
import sys
import os
import time
import utils.tsutils as ts
import utils.hfutils as hf
from utils.system_utils import check_if_path_exists, get_all_files_in_directory
from kubernetes import client, config
from kserve import (
    KServeClient,
    constants,
    V1beta1PredictorSpec,
    V1beta1TorchServeSpec,
    V1beta1InferenceServiceSpec,
    V1beta1InferenceService,
)

CONFIG_DIR = "config"
CONFIG_FILE = "config.properties"
MODEL_STORE_DIR = "model-store"
PATH_TO_SAMPLE = "../data/qa/sample_text1.json"

kubMemUnits = ["Ei", "Pi", "Ti", "Gi", "Mi", "Ki"]


def get_inputs_from_folder(input_path):
    """
    Retrieve a list of file paths of inputs for inference within a specified directory.

    Args:
      input_path (str): The path to the directory containing the files.

    Returns:
      List[str]: A list of file paths within the input directory.
    """
    return (
        [
            os.path.join(input_path, item)
            for item in get_all_files_in_directory(input_path)
        ]
        if input_path
        else []
    )


def check_if_valid_version(model_info, mount_path):
    """
    Check if the model files for a specific commit ID exist in the given directory.

    Args:
      model_info(dict): A dictionary containing the following:
        model_name (str): The name of the model.
        repo_version (str): The commit ID of HuggingFace repo of the model.
        repo_id (str): The repo id.
        hf_token (str): Your HuggingFace token (Required only for LLAMA2 model).
      mount_path (str): The local file server mount path where the model files are expected.
    Raises:
        sys.exit(1): If the model files do not exist, the
                     function will terminate the program with an exit code of 1.
    """
    hf.hf_token_check(model_info["repo_id"], model_info["hf_token"])
    model_info["repo_version"] = hf.get_repo_commit_id(
        repo_id=model_info["repo_id"],
        revision=model_info["repo_version"],
        token=model_info["hf_token"],
    )
    print(model_info)
    model_spec_path = os.path.join(
        mount_path, model_info["model_name"], model_info["repo_version"]
    )
    if not os.path.exists(model_spec_path):
        print(
            f"## ERROR: The {model_info['model_name']} model files for given commit ID "
            "are not downloaded"
        )
        sys.exit(1)
    return model_info["repo_version"]


def create_pv(core_api, deploy_name, storage, nfs_server, nfs_path):
    """
    This function creates a Persistent Volume using the provided parameters.

    Args:
      core_api: The Kubernetes CoreV1Api instance.
      deploy_name (str): Name of the PV.
      storage (str): Storage capacity for the PV.
      nfs_server (str): Server address for the NFS storage.
      nfs_path (str): Path on the NFS server to mount.
    """
    # Create Persistent Volume
    persistent_volume = client.V1PersistentVolume(
        api_version="v1",
        kind="PersistentVolume",
        metadata=client.V1ObjectMeta(name=deploy_name, labels={"storage": "nfs"}),
        spec=client.V1PersistentVolumeSpec(
            capacity={"storage": storage},
            access_modes=["ReadWriteMany"],
            persistent_volume_reclaim_policy="Retain",
            nfs=client.V1NFSVolumeSource(path=nfs_path, server=nfs_server),
        ),
    )

    core_api.create_persistent_volume(body=persistent_volume)


def create_pvc(core_api, deploy_name, storage):
    """
    This function creates a Persistent Volume Claim using the provided parameters.

    Args:
      core_api: The Kubernetes CoreV1Api instance.
      deploy_name (str): Name of the PVC.
      storage (str): Storage capacity for the PV.
    """
    # Create Persistent Volume Claim
    persistent_volume_claim = client.V1PersistentVolumeClaim(
        api_version="v1",
        kind="PersistentVolumeClaim",
        metadata=client.V1ObjectMeta(name=deploy_name),
        spec=client.V1PersistentVolumeClaimSpec(
            storage_class_name="",
            access_modes=["ReadWriteMany"],
            resources=client.V1ResourceRequirements(requests={"storage": storage}),
            selector=client.V1LabelSelector(match_labels={"storage": "nfs"}),
        ),
    )

    core_api.create_namespaced_persistent_volume_claim(
        body=persistent_volume_claim, namespace="default"
    )


def create_isvc(deploy_name, model_info, deployment_resources, model_params):
    """
    This function creates a inference service a PyTorch Predictor that expose LLMs
    as RESTful APIs, allowing to make predictions using the deployed LLMs.
    The PyTorch Predictor specifies the model's storage URI, environment and resource
    requirements (CPU and memory limits and requests).

    Args:
      deploy_name (str): Name of the inference service.
      model_info(dict): A dictionary containing the following:
        model_name (str): The name of the model whose inference service is to be created.
        repo_version (str): The commit ID of HuggingFace repo of the model.
      deployment_resources (dict): Dictionary containing number of cpus,
                                   memory and number of gpus to be used
                                   for the inference service.
      model_params(dict): Dictionary containing parameters of the model
    """
    if model_params["is_custom"]:
        storageuri = f"pvc://{deploy_name}/{model_info['model_name']}"
    else:
        storageuri = f"pvc://{deploy_name}/{model_info['model_name']}/{model_info['repo_version']}"
    default_model_spec = V1beta1InferenceServiceSpec(
        predictor=V1beta1PredictorSpec(
            pytorch=V1beta1TorchServeSpec(
                protocol_version="v2",
                storage_uri=storageuri,
                env=[
                    client.V1EnvVar(name="TS_SERVICE_ENVELOPE", value="body"),
                    client.V1EnvVar(
                        name="TS_NUMBER_OF_GPU", value=str(deployment_resources["gpus"])
                    ),
                    client.V1EnvVar(
                        name="NAI_TEMPERATURE", value=str(model_params["temperature"])
                    ),
                    client.V1EnvVar(
                        name="NAI_REP_PENALTY",
                        value=str(model_params["repetition_penalty"]),
                    ),
                    client.V1EnvVar(name="NAI_TOP_P", value=str(model_params["top_p"])),
                    client.V1EnvVar(
                        name="NAI_MAX_TOKENS", value=str(model_params["max_new_tokens"])
                    ),
                ],
                resources=client.V1ResourceRequirements(
                    limits={
                        "cpu": deployment_resources["cpus"],
                        "memory": deployment_resources["memory"],
                        "nvidia.com/gpu": deployment_resources["gpus"],
                    },
                    requests={
                        "cpu": deployment_resources["cpus"],
                        "memory": deployment_resources["memory"],
                        "nvidia.com/gpu": deployment_resources["gpus"],
                    },
                ),
            )
        )
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(name=deploy_name, namespace="default"),
        spec=default_model_spec,
    )

    kserve = KServeClient(client_configuration=config.load_kube_config())
    kserve.create(isvc, watch=True)


def execute_inference_on_inputs(
    model_inputs, model_name, deploy_name, retry=False, debug=False
):
    """
    This function sends a list of model inputs to a specified model deployment using the KServe
    and gets the inference results. It is used to run inference on a LLM deployed in
    a Kubernetes cluster.
    Args:
        model_inputs (list): A list of input data path for the model.
        model_name (str): The name of the model to perform inference on.
        deploy_name (str): The name of the inference service where the model is deployed.
        retry (bool, optional): If True, the function will retry running inference on the model
                                in case of a failure. If False, the function will exit with
                                an error message on the first failure. Default is False.
        debug (bool, optional): If True, the function will print debug information during execution.
                                Default is False.

    Returns:
        bool: True if inference was successfully executed on all input data; False if at least
              one inference attempt failed.

    Raises:
        sys.exit(1): If the `retry` parameter is set to False, and an inference attempt fails, the
                     function will terminate the program with an exit code of 1.
    """
    for model_input in model_inputs:
        host = os.environ.get("INGRESS_HOST")
        port = os.environ.get("INGRESS_PORT")

        kserve = KServeClient(client_configuration=config.load_kube_config())
        obj = kserve.get(name=deploy_name, namespace="default")
        service_hostname = obj["status"]["url"].split("/")[2:][0]
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Host": service_hostname,
        }
        connection_params = {
            "protocol": "http",
            "host": host,
            "port": port,
            "headers": headers,
        }
        response = ts.run_inference_v2(
            model_name,
            model_input,
            connection_params,
            debug=debug,
        )
        if response and response.status_code == 200:
            if debug:
                print(
                    f"## Successfully ran inference on {model_name} model. \n\n"
                    f"Output - {response.text}\n\n"
                )

            is_success = True
        else:
            if not retry:
                if debug:
                    print(f"## Failed to run inference on {model_name} - model \n")
                sys.exit(1)
            is_success = False
    return is_success


def health_check(model_name, deploy_name, model_timeout):
    """
    This function checks if the model is resistered or not.

    Args:
      model_name (str): The name of the model that is being registered.
      deploy_name (str): The name of the inference service where the model is deployed.
      model_timeout (int): Maximum amount of time to wait for a response from inference service,

    Raises:
        sys.exit(1): If health check fails after multiple retries for model, the
                     function will terminate the program with an exit code of 1.
    """
    model_input = os.path.join(os.path.dirname(__file__), PATH_TO_SAMPLE)

    retry_count = 0
    sleep_time = 30
    success = False
    while not success and retry_count * sleep_time < model_timeout:
        success = execute_inference_on_inputs(
            [model_input], model_name, deploy_name, retry=True
        )

        if not success:
            time.sleep(sleep_time)
            retry_count += 1

    if success:
        print("## Health check passed. Model deployed.\n\n")
    else:
        print(
            f"## Failed health check after multiple retries for model - {model_name} \n"
        )
        sys.exit(1)


def execute(params):
    """
    This function orchestrates the deployment and inference of a LLM
    in a Kubernetes cluster by performing tasks such as creating
    persistent storage, registering the model, and running inference.

    Args:
        params (argparse.Namespace): An object containing command-line arguments and options.

    Returns:
        None
    """
    if not any(unit in params.mem for unit in kubMemUnits):
        print("container memory unit has to be one of", kubMemUnits)
        sys.exit(1)

    deployment_resources = {}
    deployment_resources["gpus"] = params.gpu
    deployment_resources["cpus"] = params.cpu
    deployment_resources["memory"] = params.mem

    nfs_server, nfs_path = params.nfs.split(":")
    deploy_name = params.deploy_name

    model_info = {}
    model_info["model_name"] = params.model_name
    model_info["repo_version"] = params.repo_version
    model_info["hf_token"] = params.hf_token

    input_path = params.data
    mount_path = params.mount_path
    model_timeout = params.model_timeout

    check_if_path_exists(mount_path, "local nfs mount", is_dir=True)
    if not nfs_path or not nfs_server:
        print(
            "NFS server and share path was not provided in accepted format - <address>:<share_path>"
        )
        sys.exit(1)

    storage = "100Gi"

    model_params = ts.get_model_params(model_info["model_name"])

    if not model_params["is_custom"]:
        if not model_info["repo_version"]:
            model_info["repo_version"] = model_params["repo_version"]
        model_info["repo_id"] = model_params["repo_id"]
        model_info["repo_version"] = check_if_valid_version(model_info, mount_path)

    config.load_kube_config()
    core_api = client.CoreV1Api()

    create_pv(core_api, deploy_name, storage, nfs_server, nfs_path)
    create_pvc(core_api, deploy_name, storage)
    create_isvc(deploy_name, model_info, deployment_resources, model_params)

    print("wait for model registration to complete, will take some time")
    health_check(model_info["model_name"], deploy_name, model_timeout)

    if input_path:
        check_if_path_exists(input_path, "Input", is_dir=True)
        model_inputs = get_inputs_from_folder(input_path)
        execute_inference_on_inputs(
            model_inputs, model_info["model_name"], deploy_name, debug=True
        )


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Script to generate the yaml.")

    # Add arguments
    parser.add_argument("--nfs", type=str, help="nfs ip address with mount path")
    parser.add_argument("--gpu", type=int, help="number of gpus")
    parser.add_argument("--cpu", type=int, help="number of cpus")
    parser.add_argument("--mem", type=str, help="memory required by the container")
    parser.add_argument("--model_name", type=str, help="name of the model to deploy")
    parser.add_argument("--deploy_name", type=str, help="name of the deployment")
    parser.add_argument(
        "--model_timeout",
        type=int,
        help="Max time in seconds before deployment health check is terminated",
    )
    parser.add_argument(
        "--data", type=str, help="data folder for the deployment validation"
    )
    parser.add_argument(
        "--repo_version",
        type=str,
        default=None,
        help="commit id of the HuggingFace Repo",
    )
    parser.add_argument(
        "--mount_path", type=str, help="local path to the nfs mount location"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace Hub token to download LLAMA(2) models",
    )
    # Parse the command-line arguments
    args = parser.parse_args()
    execute(args)
