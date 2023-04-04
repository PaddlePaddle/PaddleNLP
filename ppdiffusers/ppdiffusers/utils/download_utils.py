# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path
import re
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import quote

import requests
from filelock import FileLock
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import _chmod_and_replace, http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from huggingface_hub.utils import tqdm as hf_tqdm
from requests import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map

from .constants import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PPDIFFUSERS_CACHE,
    PPNLP_BOS_RESOLVE_ENDPOINT,
)
from .logging import get_logger


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


def _get_model_file(
    pretrained_model_name_or_path,
    *,
    weights_name,
    subfolder,
    cache_dir,
    force_download=False,
    revision=None,
    proxies=None,
    resume_download=False,
    local_files_only=None,
    use_auth_token=None,
    user_agent=None,
    from_hf_hub=False,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        return bos_hf_download(
            pretrained_model_name_or_path,
            filename=weights_name,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            revision=revision,
            from_hf_hub=from_hf_hub,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
        )


REPO_TYPES = ["model"]
DEFAULT_REVISION = "main"
# REPO_ID_SEPARATOR = "--"
REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
PPDIFFUSERS_BOS_URL_TEMPLATE = PPNLP_BOS_RESOLVE_ENDPOINT + "/{repo_type}/community/{repo_id}/{revision}/{filename}"

ALLOW_PATTERNS_MAPPING = {
    "scheduler": [
        "scheduler_config.json",
    ],
    "text_encoder": [
        "model_state.pdparams",
        "config.json",
        "model_config.json",
    ],
    "safety_checker": [
        "model_state.pdparams",
        "config.json",
        "model_config.json",
    ],
    "unet": [
        "model_state.pdparams",
        "config.json",
    ],
    "vae": [
        "model_state.pdparams",
        "config.json",
    ],
    "vqvae": [
        "model_state.pdparams",
        "config.json",
    ],
    "bert": [
        "model_state.pdparams",
        "config.json",
        "model_config.json",
    ],
    "tokenizer": [
        "tokenizer_config.json",
        "vocab.json",
        "added_tokens.json",
        "vocab.txt",
        "special_tokens_map.json",
        "spiece.model",
        "merges.txt",
        "sentencepiece.bpe.model",
    ],
    "feature_extractor": ["preprocessor_config.json"],
    "transformer": [
        "model_state.pdparams",
        "config.json",
    ],
    "mel": ["mel_config.json"],
    "others": [
        "model_state.pdparams",
        "model_config.json",
        "config.json",
        "model_config.json",
        "scheduler_config.json",
        "preprocessor_config.json",
        "pipeline.py",
    ],
}

logger = get_logger(__name__)


def ppdiffusers_bos_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = REPO_TYPES[0]
    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")
    if repo_type == "model":
        repo_type = "models"
    if revision is None:
        revision = DEFAULT_REVISION
    return PPDIFFUSERS_BOS_URL_TEMPLATE.format(
        repo_type=repo_type,
        repo_id=repo_id,
        revision=quote(revision, safe=""),
        filename=quote(filename),
    ).replace(f"/{DEFAULT_REVISION}/", "/")


def repo_folder_name(*, repo_id: str, repo_type: str) -> str:
    # """Return a serialized version of a hf.co repo name and type, safe for disk storage
    # as a single non-nested folder.
    # Example: models--julien-c--EsperBERTo-small
    # """
    # remove all `/` occurrences to correctly convert repo to directory name
    # parts = ["ppdiffusers", f"{repo_type}s", *repo_id.split("/")]
    # return REPO_ID_SEPARATOR.join(parts)
    return repo_id


def ppdiffusers_bos_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    force_download: bool = False,
    resume_download: bool = False,
    file_lock_timeout: int = -1,
):
    if cache_dir is None:
        cache_dir = PPDIFFUSERS_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = REPO_TYPES[0]

    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are:" f" {str(REPO_TYPES)}")
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))

    if REGEX_COMMIT_HASH.match(revision):
        pointer_path = os.path.join(storage_folder, revision, relative_filename)
    else:
        pointer_path = os.path.join(storage_folder, relative_filename)

    if os.path.exists(pointer_path) and not force_download:
        return pointer_path

    url_to_download = ppdiffusers_bos_url(repo_id, filename, repo_type=repo_type, revision=revision)

    blob_path = os.path.join(storage_folder, filename)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = blob_path + ".lock"

    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
        blob_path = "\\\\?\\" + os.path.abspath(blob_path)

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with FileLock(lock_path, timeout=file_lock_timeout):
        # If the download just completed while the lock was activated.
        if os.path.exists(pointer_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return pointer_path

        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=None,
                resume_size=resume_size,
                headers=None,
            )

        logger.info("storing %s in cache at %s", url_to_download, blob_path)
        _chmod_and_replace(temp_file.name, blob_path)
    try:
        os.remove(lock_path)
    except OSError:
        pass

    return pointer_path


def ppdiffusers_url_download(
    url_to_download: str,
    cache_dir: Union[str, Path, None] = None,
    filename: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = False,
    file_lock_timeout: int = -1,
):
    if cache_dir is None:
        cache_dir = PPDIFFUSERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if filename is None:
        filename = url_to_download.split("/")[-1]
    file_path = os.path.join(cache_dir, filename)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = file_path + ".lock"
    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(file_path)) > 255:
        file_path = "\\\\?\\" + os.path.abspath(file_path)

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with FileLock(lock_path, timeout=file_lock_timeout):
        # If the download just completed while the lock was activated.
        if os.path.exists(file_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return file_path

        if resume_download:
            incomplete_path = file_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=None,
                resume_size=resume_size,
                headers=None,
            )

        logger.info("storing %s in cache at %s", url_to_download, file_path)
        _chmod_and_replace(temp_file.name, file_path)
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return file_path


def bos_hf_download(
    pretrained_model_name_or_path,
    *,
    filename,
    subfolder,
    cache_dir,
    force_download=False,
    revision=None,
    from_hf_hub=False,
    proxies=None,
    resume_download=False,
    local_files_only=None,
    use_auth_token=None,
    user_agent=None,
    file_lock_timeout=-1,
):
    if from_hf_hub:
        try:
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
            )
            return model_file

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                "login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(f"{pretrained_model_name_or_path} does not appear to have a file named {filename}.")
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {filename} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {filename}"
            )
        except KeyboardInterrupt:
            raise EnvironmentError(
                "You have interrupted the download, if you want to continue the download, you can set `resume_download=True`!"
            )
    else:
        try:
            model_file = ppdiffusers_bos_download(
                pretrained_model_name_or_path,
                filename=filename,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                subfolder=subfolder,
                revision=revision,
                file_lock_timeout=file_lock_timeout,
            )
            return model_file
        except HTTPError as err:
            raise EnvironmentError(
                f"{err}!\n"
                f"There was a specific connection error when trying to load '{pretrained_model_name_or_path}'! "
                f"We couldn't connect to '{PPNLP_BOS_RESOLVE_ENDPOINT}' to load this model, couldn't find it "
                f"in the cached files and it looks like '{pretrained_model_name_or_path}' is not the path to a "
                f"directory containing a file named '{filename}'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                f"'{PPNLP_BOS_RESOLVE_ENDPOINT}', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named '{filename}'"
            )
        except KeyboardInterrupt:
            raise EnvironmentError(
                "You have interrupted the download, if you want to continue the download, you can set `resume_download=True`!"
            )


def url_file_exists(url: str) -> bool:
    """check whether the url file exists

        refer to: https://stackoverflow.com/questions/2486145/python-check-if-url-to-jpg-exists

    Args:
        url (str): the url of target file

    Returns:
        bool: whether the url file exists
    """
    is_url = url.startswith("http://") or url.startswith("https://")
    if not is_url:
        return False

    result = requests.head(url)
    return result.status_code == requests.codes.ok


def ppdiffusers_bos_dir_download(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    resume_download: bool = False,
    folder_names: Optional[Union[List[str], str]] = None,
    max_workers: int = 1,
    tqdm_class: Optional[base_tqdm] = None,
    variant: Optional[str] = None,
    is_fastdeploy_model: Optional[str] = False,
    file_lock_timeout: int = -1,
) -> str:
    # update repo id must end with @fastdeploy
    if is_fastdeploy_model and not repo_id.endswith("@fastdeploy"):
        repo_id = f"{repo_id}@fastdeploy"

    filtered_repo_files = [["model_index.json", None]]
    for subfolder in folder_names:
        allow_patterns = ALLOW_PATTERNS_MAPPING.get(subfolder, ALLOW_PATTERNS_MAPPING["others"])
        if is_fastdeploy_model:
            allow_patterns = [ap for ap in allow_patterns if "pdparams" not in ap]
            allow_patterns.extend(["inference.pdiparams", "inference.pdmodel"])
        for filename in allow_patterns:
            if "pdparams" in filename:
                filename = _add_variant(filename, variant)
            url = ppdiffusers_bos_url(
                repo_id,
                filename=filename,
                subfolder=subfolder,
            )
            if url_file_exists(url):
                filtered_repo_files.append(
                    [
                        filename,
                        subfolder,
                    ]
                )

    def _inner_ppdiffusers_bos_download(repo_file_list):
        filename, _subfolder = repo_file_list
        return ppdiffusers_bos_download(
            repo_id,
            filename=filename,
            subfolder=_subfolder,
            repo_type=repo_type,
            cache_dir=cache_dir,
            revision=revision,
            resume_download=resume_download,
            file_lock_timeout=file_lock_timeout,
        )

    thread_map(
        _inner_ppdiffusers_bos_download,
        filtered_repo_files,
        desc=f"Fetching {len(filtered_repo_files)} files",
        max_workers=max_workers,
        # User can use its own tqdm class or the default one from `huggingface_hub.utils`
        tqdm_class=tqdm_class or hf_tqdm,
    )
    return os.path.join(cache_dir, repo_id)
