import json
import requests

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
