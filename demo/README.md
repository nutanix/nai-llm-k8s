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
> Before deploying the Chatbot app, ensure that you have the necessary prerequisites. This includes having **kubectl** installed and a valid **KubeConfig** file for the Kubernetes (K8s) cluster where the Language Model (LLM) is deployed. If prerequisites are not present, follow the steps below:
>* Install [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl).
>* Download and set up KubeConfig by following the steps outlined in [Downloading the Kubeconfig](https://portal.nutanix.com/page/documents/details?targetId=Nutanix-Kubernetes-Engine-v2_5:top-download-kubeconfig-t.html) on the Nutanix Support Portal.  

Once the inference server is up, run

    streamlit run chat.py
