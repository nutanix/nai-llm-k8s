#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

CPU_POD="8"
MEM_POD="32Gi"
MODEL_TIMEOUT_IN_SEC="1500"

function helpFunction()
{
    echo "Usage: $0 -n <MODEL_NAME>  -g <NUM_OF_GPUS> -f <NFS_ADDRESS_WITH_SHARE_PATH> -m <NFS_LOCAL_MOUNT_LOCATION> -e <KUBE_DEPLOYMENT_NAME> [OPTIONAL -d <INPUT_DATA_ABSOLUTE_PATH> -v <REPO_COMMIT_ID> -t <Your_HuggingFace_Hub_Token> -q <QUANTIZE_BITS>]"
    echo -e "\t-f NFS server address with share path information"
    echo -e "\t-m Absolute path to the NFS local mount location"
    echo -e "\t-e Name of the deployment metadata"
    echo -e "\t-o Choice of compute infra to be run on"
    echo -e "\t-n Name of the Model"
    echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
    echo -e "\t-g Number of gpus to be used to execute. Set 0 to use cpu"
    echo -e "\t-v Commit id of the HuggingFace Repo."
    echo -e "\t-t Your HuggingFace token (Required only for LLAMA2 model)."
    echo -e "\t-q BitsAndBytes Quantization Precision (4 or 8)"
    exit 1 # Exit script after printing help
}

function inference_exec_kubernetes()
{   
    echo "Config $KUBECONFIG"

    if [ -z "$KUBECONFIG" ]; then
        echo "Kube config environment variable is not set - KUBECONFIG"
        exit 1 
    fi

    if [ -z "$gpus"  ] || [ "$gpus" -eq 0 ] 
    then
        gpus="0"
    fi
    
    if [ -z $mount_path ] ; then
        echo "Local mount path not provided"
        helpFunction
    fi

    if [ -z $nfs ] ; then
        echo "NFS info not provided"
        helpFunction
    fi

    if [ -z $deploy_name ] ; then
        echo "deployment metadata name not provided"
        helpFunction
    fi
    
    export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
    export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')

    exec_cmd="python3 $wdir/kubeflow_inference_run.py --gpu $gpus --cpu $CPU_POD --mem $MEM_POD --model_name $model_name --nfs $nfs --mount_path $mount_path --deploy_name $deploy_name --model_timeout $MODEL_TIMEOUT_IN_SEC"

    if [ ! -z $data ] ; then
        exec_cmd+=" --data $data"
    fi

    if [ ! -z $repo_version ] ; then
        exec_cmd+=" --repo_version $repo_version"
    fi

    if [ ! -z $hf_token ] ; then
        exec_cmd+=" --hf_token $hf_token"
    fi

    if [ ! -z $quantize_bits ] ; then
        exec_cmd+=" --quantize_bits $quantize_bits"
    fi

    echo "Running the Inference script";
    $exec_cmd
}

# Entry Point
while getopts ":n:v:m:t:d:g:f:e:q:" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        f ) nfs="$OPTARG" ;;
        e ) deploy_name="$OPTARG" ;;
        v ) repo_version="$OPTARG" ;;
        m ) mount_path="$OPTARG" ;;
        t ) hf_token="$OPTARG" ;;
        q ) quantize_bits="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

inference_exec_kubernetes
