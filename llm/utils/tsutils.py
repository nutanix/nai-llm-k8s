import os
import json
import requests
import collections

def run_inference_v2(model_name, file_name, protocol="http", 
                  host="localhost", port="8080", timeout=120, headers=None):
    print(f"## Running inference on {model_name} model \n")

    url = f"{protocol}://{host}:{port}/v2/models/{model_name}/infer"

    print("Url", url)
    with open(file_name, 'r') as f:
        data = json.load(f)
        print("Data", data, "\n")

    response = requests.post(url, json=data, headers=headers, timeout=timeout)
    print(response, "\n")
    return response

def get_model_params(model_name):
    model_params = collections.defaultdict(str)

    dirpath = os.path.dirname(__file__)    
    with open(os.path.join(dirpath, '../model_config.json'), 'r') as file:
        model_config = json.loads(file.read())
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
            