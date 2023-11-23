#!/usr/bin/env bash

KF_VERSION=release-v1.8

if [ -z "$WORK_DIR" ]; then
   echo "Working directory env variable not set"
   exit 1
fi

helpFunction()
{
   echo ""
   echo "Usage: $0 -d <COMPANY_DOMAIN>"
   echo "  -d provide company domain e.g. ntnx.com"
   exit 1 # Exit script after printing help
}
while getopts ":d:" opt;
do
   case "$opt" in
        d ) company_domain="$OPTARG" ;;
        ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

if [ -z "$company_domain"  ] 
then
    echo "Company domain not provided"
    helpFunction
fi

# Download kubeflow manifests and install
cd $WORK_DIR/setup && git clone -b $KF_VERSION https://github.com/nutanix/kubeflow-manifests.git
cp $WORK_DIR/setup/object-store-secrets.env $WORK_DIR/setup/kubeflow-manifests/kubeflow/overlays/pipeline/ntnx/object-store-secrets.env
cp $WORK_DIR/setup/pipeline-install-config.env $WORK_DIR/setup/kubeflow-manifests/kubeflow/overlays/pipeline/ntnx/pipeline-install-config.env
cd $WORK_DIR/setup/kubeflow-manifests && make install-nke-kubeflow

# Apply patches
kubectl patch cm config-domain -p "{\"data\": {\"$company_domain\": \"\"}}" -n knative-serving
kubectl patch service istio-ingressgateway -p '{"spec": {"type": "LoadBalancer"}}' -n istio-system

# Remove kubeflow manifests
rm -rf $WORK_DIR/setup/kubeflow-manifests