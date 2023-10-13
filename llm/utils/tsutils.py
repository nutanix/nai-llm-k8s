"""
TSUtils
utility functions for running inference and getiing model parameters
"""
import os
import json
import collections
import requests


def run_inference_v2(model_name,
                    file_name,
                    connection_params,
                    timeout=120,
                    debug=False):
    """
    This function runs inference using a specified model via a REST API
    Args:
        model_name (str): The name of the model to run inference.
        file_name (str): The file containing input data in JSON format.
        connection_params(dict): Dictionary containing the following:
            protocol (str, optional): The communication protocol (default is "http").
            host (str, optional): The host where the inference 
                                service is running (default is "localhost").
            port (int, optional): The port to connect to (default is 8080).
            headers (dict, optional): Additional HTTP headers to include in the request.
        timeout (int, optional): The maximum time to wait for the 
                                 response in seconds (default is 120).
        debug (bool, optional): Whether to enable debugging output (default is False).
    Returns:
    dict: The response from the inference service, typically containing the inference results.

    Example usage:
    response = run_inference_v2("falcon_7b", "input_data.json", 
                                protocol="https", host="localhost", 
                                port=8080, timeout=120, debug=True)
    """
    if debug:
        print(f'## Running inference on {model_name} model \n')

    url = (
        f"{connection_params['protocol']}://{connection_params['host']}:"
        f"{connection_params['port']}/v2/models/{model_name}/infer"
    )

    if debug:
        print("Url", url)
    with open(file_name, "r", encoding='utf-8') as f:
        data = json.load(f)
        if debug:
            print("Data", data, "\n")

    response = requests.post(url, json=data, headers=connection_params['headers'], timeout=timeout)
    if debug:
        print(response, "\n")
    return response


def get_model_params(model_name):
    """
    This function reads the model parameters from model_config.json and stores then in a dict.
    Args:
        model_name (str): The name of the model whose parameters are needed.
    Returns:
    dict: contains the repo_version, temperature, 
          repetition penalty, top_p and max new tokens of the model.

    Example usage:
    model_params = get_model_params("falcon_7b")
    """
    model_params = collections.defaultdict(str)

    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, "../model_config.json"), "r",  encoding='utf-8') as file:
        model_config = json.loads(file.read())
        if model_name in model_config:
            model_params["repo_version"] = model_config[model_name]["repo_version"]
        if model_name in model_config and "model_params" in model_config[model_name]:
            param_config = model_config[model_name]["model_params"]
            if "temperature" in param_config:
                model_params["temperature"] = param_config["temperature"]

            if "repetition_penalty" in param_config:
                model_params["repetition_penalty"] = param_config["repetition_penalty"]

            if "top_p" in param_config:
                model_params["top_p"] = param_config["top_p"]

            if "max_new_tokens" in param_config:
                model_params["max_new_tokens"] = param_config["max_new_tokens"]

    return model_params
