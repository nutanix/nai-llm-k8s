#!/usr/bin/env bash
SCRIPT=$(realpath "$0")
wdir=$(dirname "$SCRIPT")

CPU_pod="8"
MEM_pod="32Gi"

function helpFunction()
{
    echo "Usage: $0 -n <MODEL_NAME> -d <INPUT_DATA_ABSOLUTE_PATH>  -g <NUM_OF_GPUS> -m <NFS_LOCAL_MOUNT_LOCATION> -f <NFS_ADDRESS_WITH_SHARE_PATH> -e <KUBE_DEPLOYMENT_NAME> -s <KUBE_DEPLOYMENT_NAMEPSACE> [OPTIONAL -k]"
    echo -e "\t-m Absolute path to the NFS local mount location"
    echo -e "\t-f NFS server address with share path information"
    echo -e "\t-e Name of the deployment metadata"
    echo -e "\t-s Namespace for the deployment"
    echo -e "\t-n Name of the Model"
    echo -e "\t-d Absolute path to the inputs folder that contains data to be predicted."
    echo -e "\t-g Number of gpus to be used to execute. Set 0 to use cpu"
    echo -e "\t-k Keep the torchserve server alive after run completion. Default stops the server if not set"
    
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

    if [ -z $namespace ] ; then
        namespace="kubeflow-user-example-com"
    fi

    mkdir $mount_path/$model_name/config
    cp $wdir/config.properties $mount_path/$model_name/config/

    export INGRESS_HOST=$(kubectl get po -l istio=ingressgateway -n istio-system -o jsonpath='{.items[0].status.hostIP}')
    export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].nodePort}')

    echo "Running the Inference script";
    python3 $wdir/kubeflow_inference_run.py --gpu $gpus --cpu $CPU_pod --mem $MEM_pod --model_name $model_name --mount_path $mount_path --nfs $nfs --deploy_name $deploy_name --namespace $namespace --data $data

    if [ -z $stop_server ] ; then
        python3 $wdir/utils/cleanup.py --deploy_name $deploy_name
    fi
}

# Entry Point
while getopts ":n:d:g:m:f:e:s:k" opt;
do
   case "$opt" in
        n ) model_name="$OPTARG" ;;
        d ) data="$OPTARG" ;;
        g ) gpus="$OPTARG" ;;
        m ) mount_path="$OPTARG" ;;
        f ) nfs="$OPTARG" ;;
        e ) deploy_name="$OPTARG" ;;
        s ) namespace="$OPTARG" ;;
        k ) stop_server=0 ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

inference_exec_kubernetes
