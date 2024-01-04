# Chatbot demo

This is a real time chatbot demo which talks to the deployed model endpoint over the REST API. 

## Install Python requirements

    pip install -r requirements.txt

## Deploy models

Download and deploy the following models on K8s cluster as per instructions provided in the [docs](https://opendocs.nutanix.com/gpt-in-a-box/overview/). 

    lama2-7b-chat
    
    codellama-7b-python

## Run Chatbot app
>**NOTE:**   
> When running the Chatbot from a machine separate from the one hosting the Language Model (LLM), it is required to:
>* Install [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl).
>* Download the KubeConfig of the same K8s cluster as the machine where LLM is deployed. Download and set up KubeConfig by following the steps outlined in [Downloading the Kubeconfig](https://portal.nutanix.com/page/documents/details?targetId=Nutanix-Kubernetes-Engine-v2_5:top-download-kubeconfig-t.html) on the Nutanix Support Portal.  

Once the inference server is up, run

    streamlit run chat.py
