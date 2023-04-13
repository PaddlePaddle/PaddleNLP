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
"""Utilities to dynamically load objects from the Hub."""

import importlib
import inspect
import json
import os
import re
import shutil
import sys
from distutils.version import StrictVersion
from pathlib import Path
from typing import Dict, Optional, Union
from urllib import request

from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info

from . import PPDIFFUSERS_DYNAMIC_MODULE_NAME, PPDIFFUSERS_MODULES_CACHE, logging

COMMUNITY_PIPELINES_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleNLP/{revision}/ppdiffusers/examples/community/{pipeline}.py"
)
GITEE_COMMUNITY_PIPELINES_URL = (
    "https://gitee.com/paddlepaddle/PaddleNLP/raw/{revision}/ppdiffusers/examples/community/{pipeline}.py"
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_ppdiffusers_versions():
    url = "https://pypi.org/pypi/ppdiffusers/json"
    releases = json.loads(request.urlopen(url).read())["releases"].keys()
    ignore = ["0.6.0.dev1"]
    releases = [r for r in releases if r not in ignore]
    return sorted(releases, key=StrictVersion)


def init_ppdiffusers_modules():
    """
    Creates the cache directory for modules with an init, and adds it to the Python path.
    """
    # This function has already been executed if PPDIFFUSERS_MODULES_CACHE already is in the Python path.
    if PPDIFFUSERS_MODULES_CACHE in sys.path:
        return

    sys.path.append(PPDIFFUSERS_MODULES_CACHE)
    os.makedirs(PPDIFFUSERS_MODULES_CACHE, exist_ok=True)
    init_path = Path(PPDIFFUSERS_MODULES_CACHE) / "__init__.py"
    if not init_path.exists():
        init_path.touch()


def create_dynamic_module(name: Union[str, os.PathLike]):
    """
    Creates a dynamic module in the cache directory for modules.
    """
    init_ppdiffusers_modules()
    dynamic_module_path = Path(PPDIFFUSERS_MODULES_CACHE) / name
    # If the parent module does not exist yet, recursively create it.
    if not dynamic_module_path.parent.exists():
        create_dynamic_module(dynamic_module_path.parent)
    os.makedirs(dynamic_module_path, exist_ok=True)
    init_path = dynamic_module_path / "__init__.py"
    if not init_path.exists():
        init_path.touch()


def get_relative_imports(module_file):
    """
    Get the list of modules that are relatively imported in a module file.

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.
    """
    with open(module_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import .xxx`
    relative_imports = re.findall("^\s*import\s+\.(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from .xxx import yyy`
    relative_imports += re.findall("^\s*from\s+\.(\S+)\s+import", content, flags=re.MULTILINE)
    # Unique-ify
    return list(set(relative_imports))


def get_relative_import_files(module_file):
    """
    Get the list of all files that are needed for a given module. Note that this function recurses through the relative
    imports (if a imports b and b imports c, it will return module files for b and c).

    Args:
        module_file (`str` or `os.PathLike`): The module file to inspect.
    """
    no_change = False
    files_to_check = [module_file]
    all_relative_imports = []

    # Let's recurse through all relative imports
    while not no_change:
        new_imports = []
        for f in files_to_check:
            new_imports.extend(get_relative_imports(f))

        module_path = Path(module_file).parent
        new_import_files = [str(module_path / m) for m in new_imports]
        new_import_files = [f for f in new_import_files if f not in all_relative_imports]
        files_to_check = [f"{f}.py" for f in new_import_files]

        no_change = len(new_import_files) == 0
        all_relative_imports.extend(files_to_check)

    return all_relative_imports


def check_imports(filename):
    """
    Check if the current Python environment contains all the libraries that are imported in a file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    # Imports of the form `import xxx`
    imports = re.findall("^\s*import\s+(\S+)\s*$", content, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall("^\s*from\s+(\S+)\s+import", content, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]

    # Unique-ify and test we got them all
    imports = list(set(imports))
    missing_packages = []
    for imp in imports:
        try:
            importlib.import_module(imp)
        except ImportError:
            missing_packages.append(imp)

    if len(missing_packages) > 0:
        raise ImportError(
            "This modeling file requires the following packages that were not found in your environment: "
            f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
        )

    return get_relative_imports(filename)


def get_class_in_module(class_name, module_path):
    """
    Import a module on the cache directory for modules and extract a class from it.
    """
    module_path = module_path.replace(os.path.sep, ".")
    module = importlib.import_module(module_path)

    if class_name is None:
        return find_pipeline_class(module)
    return getattr(module, class_name)


def find_pipeline_class(loaded_module):
    """
    Retrieve pipeline class that inherits from `DiffusionPipeline`. Note that there has to be exactly one class
    inheriting from `DiffusionPipeline`.
    """
    from ..pipelines import DiffusionPipeline

    cls_members = dict(inspect.getmembers(loaded_module, inspect.isclass))

    pipeline_class = None
    for cls_name, cls in cls_members.items():
        if (
            cls_name != DiffusionPipeline.__name__
            and issubclass(cls, DiffusionPipeline)
            and cls.__module__.split(".")[0] != "ppdiffusers"
        ):
            if pipeline_class is not None:
                raise ValueError(
                    f"Multiple classes that inherit from {DiffusionPipeline.__name__} have been found:"
                    f" {pipeline_class.__name__}, and {cls_name}. Please make sure to define only one in"
                    f" {loaded_module}."
                )
            pipeline_class = cls

    return pipeline_class


def get_cached_module_file(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
):
    """
    Prepares Downloads a module from a local folder or a distant repo and returns its path inside the cached
    Transformers module.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    You may pass a token in `use_auth_token` if you are not logged in (`huggingface-cli long`) and want to use private
    or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models).

    </Tip>

    Returns:
        `str`: The path to the module inside the cache.
    """
    # Download and cache module_file from the repo `pretrained_model_name_or_path` of grab it if it's a local file.
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)

    module_file_or_url = os.path.join(pretrained_model_name_or_path, module_file)

    if os.path.isfile(module_file_or_url):
        resolved_module_file = module_file_or_url
        submodule = "local"
    elif pretrained_model_name_or_path.count("/") == 0:
        # retrieve github version that matches
        if revision is None:
            revision = "main"
            logger.info(f"Defaulting to main: {revision}.")

        # community pipeline on GitHub
        github_url = COMMUNITY_PIPELINES_URL.format(revision=revision, pipeline=pretrained_model_name_or_path)
        try:
            resolved_module_file = cached_download(
                github_url,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=False,
            )
            submodule = "git"
            module_file = pretrained_model_name_or_path + ".py"
        except EnvironmentError:
            logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
            raise
    else:
        try:
            # Load from URL or cache if already cached
            resolved_module_file = hf_hub_download(
                pretrained_model_name_or_path,
                weights_name=module_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )
            submodule = os.path.join("local", "--".join(pretrained_model_name_or_path.split("/")))
        except EnvironmentError:
            logger.error(f"Could not locate the {module_file} inside {pretrained_model_name_or_path}.")
            raise

    # Check we have all the requirements in our environment
    modules_needed = check_imports(resolved_module_file)

    # Now we move the module inside our cached dynamic modules.
    full_submodule = PPDIFFUSERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
    create_dynamic_module(full_submodule)
    submodule_path = Path(PPDIFFUSERS_MODULES_CACHE) / full_submodule
    if submodule == "local" or submodule == "git":
        # We always copy local files (we could hash the file to see if there was a change, and give them the name of
        # that hash, to only copy when there is a modification but it seems overkill for now).
        # The only reason we do the copy is to avoid putting too many folders in sys.path.
        shutil.copy(resolved_module_file, submodule_path / module_file)
        for module_needed in modules_needed:
            module_needed = f"{module_needed}.py"
            shutil.copy(os.path.join(pretrained_model_name_or_path, module_needed), submodule_path / module_needed)
    else:
        # Get the commit hash
        # TODO: we will get this info in the etag soon, so retrieve it from there and not here.
        if isinstance(use_auth_token, str):
            token = use_auth_token
        elif use_auth_token is True:
            token = HfFolder.get_token()
        else:
            token = None

        commit_hash = model_info(pretrained_model_name_or_path, revision=revision, token=token).sha

        # The module file will end up being placed in a subfolder with the git hash of the repo. This way we get the
        # benefit of versioning.
        submodule_path = submodule_path / commit_hash
        full_submodule = full_submodule + os.path.sep + commit_hash
        create_dynamic_module(full_submodule)

        if not (submodule_path / module_file).exists():
            shutil.copy(resolved_module_file, submodule_path / module_file)
        # Make sure we also have every file with relative
        for module_needed in modules_needed:
            if not (submodule_path / module_needed).exists():
                get_cached_module_file(
                    pretrained_model_name_or_path,
                    f"{module_needed}.py",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
    return os.path.join(full_submodule, module_file)


def get_class_from_dynamic_module(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    module_file: str,
    class_name: Optional[str] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Extracts a class from a module file, present in the local folder or repository of a model.

    <Tip warning={true}>

    Calling this function will execute the code in the module file found locally or downloaded from the Hub. It should
    therefore only be called on trusted repos.

    </Tip>

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced
              under a user or organization name, like `dbmdz/bert-base-german-cased`.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        module_file (`str`):
            The name of the module file containing the class to look for.
        class_name (`str`):
            The name of the class to import in the module.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or `bool`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    You may pass a token in `use_auth_token` if you are not logged in (`huggingface-cli long`) and want to use private
    or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models).

    </Tip>

    Returns:
        `type`: The class, dynamically imported from the module.

    Examples:

    ```python
    # Download module `modeling.py` from huggingface.co and cache then extract the class `MyBertModel` from this
    # module.
    cls = get_class_from_dynamic_module("sgugger/my-bert-model", "modeling.py", "MyBertModel")
    ```"""
    # And lastly we get the class inside our newly created module
    final_module = get_cached_module_file(
        pretrained_model_name_or_path,
        module_file,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        use_auth_token=use_auth_token,
        revision=revision,
        local_files_only=local_files_only,
    )
    return get_class_in_module(class_name, final_module.replace(".py", ""))
