"""
This module stores the dataclasses GenerateDataModel, MarUtils, RepoInfo.
"""
import dataclasses


@dataclasses.dataclass
class MarUtils:
    """
    A class for representing information about a Model Archive (MAR).

    Attributes:
        mar_output (str): The path to the MAR output directory.
        model_path (str): The path to the model directory.
        handler_path (str): The path to the model handler script.
    """

    mar_output = str()
    model_path = str()
    handler_path = str()


@dataclasses.dataclass
class RepoInfo:
    """
    A class for specifying details related to a model repository in HuggingFace.

    Attributes:
        repo_id (str): The identifier of the model repository.
        repo_version (str): The version of the model in the repository.
        hf_token (str): The Hugging Face token for authentication.
    """

    repo_id = str()
    repo_version = str()
    hf_token = str()


@dataclasses.dataclass
class GenerateDataModel:
    """
    A class representing a model download configuration for data retrieval.

    Attributes:
    - model_name (str): The name of the model to be downloaded.
    - download_model (bool): A boolean indicating whether to download
                             the model (True) or not (False).
    - model_path (str): The path where the downloaded model will be stored locally.
    - output (str): Mount path to the nfs server to be used in the kube
                    PV where model files and model archive file be stored.
    - mar_output (str): The output directory specifically for the Model Archive (MAR) files.
    - repository_info (dict): Dictionary that will contain the unique identifier
                              or name of the model repository, the version of the model
                              within the repository and the path to the model's handler
                              script for custom processing
    - hf_token (str): The Hugging Face API token for authentication when
                      accessing models from the Hugging Face Model Hub.
    - debug (bool): A boolean indicating whether to enable debugging mode for
                    the download process (True) or not (False).
    """

    model_name = str()
    download_model = bool()
    output = str()
    mar_utils = MarUtils()
    repo_info = RepoInfo()
    is_custom = bool()
    debug = bool()
