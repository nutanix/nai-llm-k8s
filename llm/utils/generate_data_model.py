"""
This module stores the dataclasses GenerateDataModel, MarUtils, RepoInfo,
function set_values that sets the GenerateDataModel attributes and
function set_model_files_and_mar that sets model path and mar output values.
"""
import os
import dataclasses
import argparse

MODEL_STORE_DIR = "model-store"
MODEL_FILES_LOCATION = "download"


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
    extra_files = str()
    requirements_file = str()


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

    def __init__(self, params: argparse.Namespace) -> None:
        """
        This is the init function that calls set_values method.

        Args:
            params (argparse.Namespace): An argparse.Namespace object
                                         containing command-line arguments.
        """
        self.set_values(params)

    def set_values(self, params: argparse.Namespace) -> None:
        """
        Set values for the GenerateDataModel object based on the command-line arguments.
        Args:
            params (argparse.Namespace): An argparse.Namespace object
                                        containing command-line arguments.
        Returns:
            GenerateDataModel: An instance of the GenerateDataModel
                            class with values set based on the arguments.
        """
        self.model_name = params.model_name
        self.download_model = params.skip_download
        self.output = params.output
        self.is_custom = False

        self.mar_utils.handler_path = params.handler_path

        self.repo_info.repo_id = params.repo_id
        self.repo_info.repo_version = params.repo_version
        self.repo_info.hf_token = params.hf_token

        self.debug = params.debug

    def set_model_files_and_mar(self, params: argparse.Namespace) -> None:
        """
        This function sets model path and mar output values.
        Args:
            gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                        class with relevant information.
            params (argparse.Namespace): An argparse.Namespace object
                                        containing command-line arguments.
        Returns:
            None
        """
        if self.is_custom:
            self.mar_utils.model_path = params.model_path
            self.mar_utils.mar_output = os.path.join(
                self.output,
                self.model_name,
                MODEL_STORE_DIR,
            )
        else:
            self.mar_utils.model_path = os.path.join(
                self.output,
                self.model_name,
                self.repo_info.repo_version,
                MODEL_FILES_LOCATION,
            )
            self.mar_utils.mar_output = os.path.join(
                self.output,
                self.model_name,
                self.repo_info.repo_version,
                MODEL_STORE_DIR,
            )
