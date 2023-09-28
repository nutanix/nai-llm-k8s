# NAI-LLM-K8S

## Setup

### Prerequisite
* [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)
* [kustomize](https://github.com/kubernetes-sigs/kustomize/releases/tag/kustomize%2Fv5.0.1)

### Kubeflow installation
Pass your company domain e.g. (ntnx.com) to `install.sh` script
```
bash install.sh -d=<company-domain>
```

### Setup

Install openjdk, pip3:
```
sudo apt-get install python3-pip
```

Install required packages:

```
pip install -r requirements.txt
```

Install NVIDIA Drivers:

Reference: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#runfile
Download the latest Datacenter Nvidia drivers for the GPU type from  https://www.nvidia.com/download/index.aspx

For Nvidia A100, Select A100 in Datacenter Tesla for Linux 64 bit with cuda toolkit 11.7, latest driver is 515.105.01

```
curl -fSsl -O https://us.download.nvidia.com/tesla/515.105.01/NVIDIA-Linux-x86_64-515.105.01.run
sudo sh NVIDIA-Linux-x86_64-515.105.01.run -s
```

Note: We don’t need to install CUDA toolkit separately as it is bundled with PyTorch installation. Just Nvidia driver installation is enough. 


### Scripts

#### Download model files and Generate MAR file
Run the following command for downloading model files and/or generating MAR file: 
```
python3 download_script.py [--no_download] [--no_generate] --model_name <MODEL_NAME> --model_path <MODEL_PATH> --mar_output <MAR_EXPORT_PATH>  --hf_token <Your_HuggingFace_Hub_Token>
```
- no_download:      Set flag to skip downloading the model files
- no_generate:      Set flag to skip generating MAR file
- model_name:       Name of model
- model_path:       Absolute path of model files
- mar_output:       Mount path to your nfs server to be used in the kube PV
- hf_token:         Your HuggingFace token. Needed to download LLAMA(2) models.

The available LLMs are mpt_7b, falcon_7b, llama2_7b

##### Examples

Download MPT-7B model files(13 GB) and generate model archive(9.83 GB) for it:
```
python3 llm/download.py --model_name mpt_7b --model_path /home/ubuntu/models/mpt_7b/model_files --mar_output /mnt/llm
```
Download Falcon-7B model files(14 GB) and generate model archive(10.69 GB) for it:
```
python3 llm/download.py --model_name falcon_7b --model_path /home/ubuntu/models/falcon_7b/model_files --mar_output /mnt/llm
```
Download Llama2-7B model files(26 GB) and generate model archive(9.66 GB) for it:
```
python3 llm/download.py --model_name llama2_7b --model_path /home/ubuntu/models/llama2_7b/model_files --mar_output /mnt/llm --hf_token <token_value>
```

#### Start Torchserve and run inference

Run the following command for starting Torchserve and running inference on the given input:
```
bash run.sh  -n <MODEL_NAME> -d <INPUT_PATH> -g <NUM_GPUS> -m <NFS_LOCAL_MOUNT_LOCATION> -f <NFS_ADDRESS_WITH_SHARE_PATH> -e <KUBE_DEPLOYMENT_NAME> [OPTIONAL -k]
```
- k:    Set flag to keep server alive
- n:    Name of model
- d:    Absolute path of input data folder
- g:    Number of gpus to be used to execute (Set 0 to use cpu)
- a:    Absolute path to the MAR file (.mar)
- m:    Absolute path to the NFS local mount location
- f:    NFS server address with share path information
- e:    Name of the deployment metadata

“-k” would keep the server alive and needs to stopped explicitly
For model names, we support MPT-7B, Falcon-7b and Llama2-7B.
Should print "Inference Run Successful" as a message at the end

##### Examples

For 1 GPU Inference with official MPT-7B model and keep torchserve alive:
```
bash llm/run.sh -n mpt_7b -d data/translate -m /mnt/llm -g 1 -e mpt_deploy -f '1.1.1.1:/llm' -k
```
For 1 GPU Inference with official Falcon-7B model and keep torchserve alive:
```
bash llm/run.sh -n falcon_7b -d data/qa -m /mnt/llm -g 1 -e falcon_deploy -f '1.1.1.1:/llm' -k
```
For 1 GPU Inference with official Llama2-7B model and keep torchserve alive:
```
bash llm/run.sh -n llama2_7b -d data/summarize -m /mnt/llm -g 1 -e llama2_deploy -f '1.1.1.1:/llm' -k
```

#### Inference Check

set HOST and PORT
```
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

set Service Host Name
```
SERVICE_HOSTNAME=$(kubectl get inferenceservice <deployment_name> -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer -d @./data.json
Test input file can be found in the data folder.


For MPT-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/mpt_7b/infer -d @./data/qa/sample_test1.json
```
For Falcon-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/falcon_7b/infer -d @./data/summarize/sample_test1.json
```
For Llama2-7B model
```
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/llama2_7b/infer -d @./data/translate/sample_test1.json
```

#### Stop Torchserve and Cleanup

If keep alive flag was set in the bash script, then you can run the following command to stop the server and clean up temporary files
```
python3 llm/utils/cleanup.py --deploy_name <deployment_name>
```
