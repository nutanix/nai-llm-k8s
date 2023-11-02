"""
Clean up Kubernetes resources associated with a deployment.
"""
import argparse
import sys
import requests
from kubernetes import client, config
from kserve import KServeClient


def kubernetes(deploy_name: str) -> None:
    """
    This function cleans up various Kubernetes resources,
    including deleting the deployment, persistent volume claims (PVCs), and
    persistent volumes (PVs) associated with the specified deployment name.
    Args:
        deploy_name (str): The name of the deployment to clean up.
    Returns:
        None
    Raises:
        Exception: If any error occurs during resource cleanup.
    """
    print("Clean up triggered for all the deployments under -", deploy_name)
    kserve = KServeClient(client_configuration=config.load_kube_config())  # noqa: F841
    try:
        kserve.delete(name=deploy_name, namespace="default")
    except requests.exceptions.RequestException:
        print("Deployment pod delete triggered")

    core_api = client.CoreV1Api()
    try:
        core_api.delete_namespaced_persistent_volume_claim(
            name=deploy_name, namespace="default"
        )
    except requests.exceptions.RequestException:
        print("PVC delete triggered")

    try:
        core_api.delete_persistent_volume(name=deploy_name)
    except requests.exceptions.RequestException:
        print("PV delete triggered")


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script to cleanup existing deployment."
    )

    # Add arguments
    parser.add_argument("--deploy_name", type=str, help="name of the deployment")

    # Parse the command-line arguments
    args = parser.parse_args()

    if not args.deploy_name:
        print("Deployment name not provided")
        sys.exit(1)

    kubernetes(args.deploy_name)
