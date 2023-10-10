#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

CPU_POD="8"
MEM_POD="32Gi"
MODEL_TIMEOUT_IN_SEC="600"

function helpFunction()
{
    echo "Usage: $0 -n <MODEL_NAME> -g <NUM_OF_GPUS> -f <NFS_ADDRESS_WITH_SHARE_PATH> -e <KUBE_DEPLOYMENT_NAME> [OPTIONAL -d <INPUT_DATA_ABSOLUTE_PATH>]"
    echo -e "\t-f NFS server address with share path information"
    echo -e "\t-e Name of the deployment metadata"
    echo -e "\t-o Choice of compute infra to be run on"
    echo -e "\t-n Name of the Model"
    echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
    echo -e "\t-g Number of gpus to be used to execute. Set 0 to use cpu"
    
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

    exec_cmd="python3 $wdir/kubeflow_inference_run.py --gpu $gpus --cpu $CPU_POD --mem $MEM_POD --model_name $model_name --nfs $nfs --deploy_name $deploy_name --model_timeout $MODEL_TIMEOUT_IN_SEC"

    if [ ! -z $data ] ; then
        exec_cmd+=" --data $data"
    fi

    echo "Running the Inference script";
    $exec_cmd
}

# Entry Point
while getopts ":n:d:g:f:e:" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        f ) nfs="$OPTARG" ;;
        e ) deploy_name="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

inference_exec_kubernetes
