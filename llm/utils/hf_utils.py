"""
Utility functions for using HuggingFace Api
"""
import sys
from typing import List
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    RevisionNotFoundError,
    HfHubHTTPError,
    HFValidationError,
)
from utils.generate_data_model import GenerateDataModel


def get_repo_files_list(gen_model: GenerateDataModel) -> List[str]:
    """
    This function returns a list of all files in the HuggingFace repo of
    the model.
    Args:
        gen_model (GenerateDataModel): An instance of the GenerateDataModel
                                      class with relevant information.
    Returns:
        repo_files (list): all files in the HuggingFace repo of
                           the model
    Raises:
        sys.exit(1): If repo_id, repo_version or huggingface token
                     is not valid, the function will terminate
                     the program with an exit code of 1.
    """
    try:
        hf_api = HfApi()
        repo_files = hf_api.list_repo_files(
            repo_id=gen_model.repo_info.repo_id,
            revision=gen_model.repo_info.repo_version,
            token=gen_model.repo_info.hf_token,
        )
        return repo_files
    except (
        HfHubHTTPError,
        HFValidationError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
        KeyError,
    ):
        print(
            (
                "\n## Error: Please check either repo_id, repo_version "
                "or huggingface token is not correct\n"
            )
        )
        sys.exit(1)


def get_repo_commit_id(repo_id: str, revision: str, token: str) -> str:
    """
    This function returns the whole Commit ID from HuggingFace repo of
    the model.
    Args:
        revision (str): The commit ID of HuggingFace repo of the model.
        repo_id (str): The repo id.
        token (str): Your HuggingFace token (Required only for LLAMA2 model).
    Returns:
        commit id (str): The whole commit ID of HuggingFace repo of
                         the model.
    Raises:
        sys.exit(1): If repo_id, repo_version or huggingface token
                     is not valid, the function will terminate
                     the program with an exit code of 1.
    """
    try:
        hf_api = HfApi()
        commit_info = hf_api.list_repo_commits(
            repo_id=repo_id,
            revision=revision,
            token=token,
        )
        return commit_info[0].commit_id
    except (
        HfHubHTTPError,
        HFValidationError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    ):
        print(
            (
                "\n## Error: Please check either repo_id, repo_version "
                "or huggingface token is not correct\n"
            )
        )
        sys.exit(1)


def hf_token_check(repo_id: str, token: str) -> None:
    """
    This function checks if HuggingFace token is provided for
    Llama 2 model
    Args:
        repo_id (str): The repo id.
        token (str): Your HuggingFace token (Required only for LLAMA2 model).
    Returns:
        None
    Raises:
        sys.exit(1): If if HuggingFace token is not provided for
                     Llama 2 model, the function will terminate
                     the program with an exit code of 1.
    """
    if repo_id.startswith("meta-llama") and token is None:
        # Make sure there is HF hub token for LLAMA(2)
        print(
            (
                "HuggingFace Hub token is required for llama download. "
                "Please specify it using --hf_token=<your token> argument "
                "or set it as an environment variable 'HF_TOKEN'. Refer "
                "https://huggingface.co/docs/hub/security-tokens"
            )
        )
        sys.exit(1)
