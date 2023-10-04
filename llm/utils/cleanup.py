import argparse
import sys

from kubernetes import client, config
from kserve import KServeClient

DEFAULT_NAMESPACE="kubeflow-user-example-com"

def kubernetes(deploy_name, namespace):
    print("Clean up triggered for all the deployments under -", deploy_name)
    kube_config = config.load_kube_config()
    kserve = KServeClient(client_configuration=kube_config)
    try:
        kserve.delete(name=deploy_name, namespace='default')
    except:
        print("Deployment pod delete triggered")

    core_api = client.CoreV1Api()
    try:
        core_api.delete_namespaced_persistent_volume_claim(name=deploy_name, namespace='default')
    except:
        print("PVC delete triggered")

    try:
        core_api.delete_persistent_volume(name=deploy_name)
    except:
        print("PV delete triggered")
    
    try:
        core_api.delete_namespace(name=namespace)
    except:
        print("Namespace delete triggered")
    
    


if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Script to cleanup existing deployment.')

    # Add arguments
    parser.add_argument('--deploy_name', type=str, help='name of the deployment')
    parser.add_argument('--namespace', type=str, help='namespace provided for the deployment', default=DEFAULT_NAMESPACE, required=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    if not args.deploy_name:
        print("Deployment name not provided")
        sys.exit(1)

    kubernetes(args.deploy_name, args.namespace)