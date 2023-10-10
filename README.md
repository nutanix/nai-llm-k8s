# NAI-LLM-K8S

## Setup

### Prerequisite
* [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
* [helm](https://helm.sh/docs/intro/install/)

Download and set up KubeConfig by following the steps outlined in “Downloading the Kubeconfig” on the Nutanix Support Portal.

Have a NFS mounted into your jump machine at a specific location. This mount location is required to be supplied as parameter to the execution scripts

Command to mount NFS to local folder
```
mount -t nfs -o <ip>:<share path> <NFS_LOCAL_MOUNT_LOCATION>
```

Configure Nvidia Driver in the cluster using helm commands:

```
helm repo add nvidia https://nvidia.github.io/gpu-operator && helm repo update
helm install --wait -n gpu-operator --create-namespace gpu-operator nvidia/gpu-operator --set toolkit.version=v1.13.0-centos7
```

### Kubeflow serving installation

```
curl -s "https://raw.githubusercontent.com/kserve/kserve/v0.11.0/hack/quick_install.sh" | bash
```

### Setup

Install pip3:
```
sudo apt-get install python3-pip
```

Set Working Directory
```
export WORK_DIR=/home/ubuntu/nai-llm-k8s
```

Install required packages:

```
pip install -r $WORK_DIR/llm/requirements.txt
```

### Scripts

#### Download model files and Generate MAR file
Run the following command for downloading model files and generating MAR file: 
```
python3 download.py [--no_download --repo_version <REPO_COMMIT_ID>] --model_name <MODEL_NAME> --output <NFS_LOCAL_MOUNT_LOCATION> --hf_token <Your_HuggingFace_Hub_Token>
```
- no_download:      Set flag to skip downloading the model files
- model_name:       Name of model
- output:           Mount path to your nfs server to be used in the kube PV where model files and model archive file be stored
- repo_version:     Commit id of model's repo from HuggingFace (optional, if not provided default set in model_config will be used)
- hf_token:         Your HuggingFace token. Needed to download LLAMA(2) models.

The available LLMs are mpt_7b, falcon_7b, llama2_7b

##### Examples

Download MPT-7B model files(13 GB) and generate model archive(9.83 GB) for it:
```
python3 $WORK_DIR/llm/download.py --model_name mpt_7b --output /mnt/llm --repo_version <repo_commit_id>
```
Download Falcon-7B model files(14 GB) and generate model archive(10.69 GB) for it:
```
python3 $WORK_DIR/llm/download.py --model_name falcon_7b --output /mnt/llm --repo_version <repo_commit_id>
```
Download Llama2-7B model files(26 GB) and generate model archive(9.66 GB) for it:
```
python3 $WORK_DIR/llm/download.py --model_name llama2_7b --output /mnt/llm --repo_version <repo_commit_id> --hf_token <token_value>
```

#### Start and run Kubeflow Serving

Run the following command for starting Kubeflow serving and running inference on the given input:
```
bash run.sh  -n <MODEL_NAME> -d <INPUT_PATH> -g <NUM_GPUS> -f <NFS_ADDRESS_WITH_SHARE_PATH> -m <NFS_LOCAL_MOUNT_LOCATION> -e <KUBE_DEPLOYMENT_NAME> [OPTIONAL -v <REPO_COMMIT_ID> -k]
```
- k:    Set flag to keep server alive
- n:    Name of model
- d:    Absolute path of input data folder
- g:    Number of gpus to be used to execute (Set 0 to use cpu)
- f:    NFS server address with share path information
- m:    Mount path to your nfs server to be used in the kube PV where model files and model archive file be stored
- e:    Name of the deployment metadata
- v:    Commit id of model's repo from HuggingFace (optional, if not provided default set in model_config will be used)

“-k” would keep the server alive and needs to stopped explicitly
For model names, we support MPT-7B, Falcon-7B and Llama2-7B.
Should print "Inference Run Successful" as a message at the end

##### Examples

For 1 GPU Inference with official MPT-7B model and keep inference server alive:
```
bash $WORK_DIR/llm/run.sh -n mpt_7b -d data/translate -g 1 -e llm-deploy -f '1.1.1.1:/llm' -m /mnt/llm -v <repo_commit_id> -k
```
For 1 GPU Inference with official Falcon-7B model and keep inference server alive:
```
bash $WORK_DIR/llm/run.sh -n falcon_7b -d data/qa -g 1 -e llm-deploy -f '1.1.1.1:/llm' -m /mnt/llm -v <repo_commit_id> -k
```
For 1 GPU Inference with official Llama2-7B model and keep inference server alive:
```
bash $WORK_DIR/llm/run.sh -n llama2_7b -d data/summarize -g 1 -e llm-deploy -f '1.1.1.1:/llm' -m /mnt/llm -v <repo_commit_id> -k
```

#### Inference Check

set HOST and PORT
```
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

set Service Host Name

SERVICE_HOSTNAME=$(kubectl get inferenceservice <DEPLOYMENT_NAME> -o jsonpath='{.status.url}' | cut -d "/" -f 3)

```
SERVICE_HOSTNAME=$(kubectl get inferenceservice llm-deploy -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer -d @data.json
Test input file can be found in the data folder.


For MPT-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/mpt_7b/infer -d @$WORK_DIR/data/qa/sample_test1.json
```
For Falcon-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/falcon_7b/infer -d @$WORK_DIR/data/summarize/sample_test1.json
```
For Llama2-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Content-Type: application/json" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/llama2_7b/infer -d @$WORK_DIR/data/translate/sample_test1.json
```

#### Cleanup Inference deployment

If keep alive flag was set in the bash script, then you can run the following command to stop the server and clean up temporary files

python3 $WORK_DIR/llm/utils/cleanup.py --deploy_name <DEPLOYMENT_NAME>

```
python3 $WORK_DIR/llm/utils/cleanup.py --deploy_name llm-deploy
```
