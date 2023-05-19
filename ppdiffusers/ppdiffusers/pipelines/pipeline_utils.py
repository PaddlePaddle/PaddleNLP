# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_url,
    model_info,
    repo_type_and_id_from_hf_id,
    snapshot_download,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from ..utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    TORCH_SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    get_class_from_dynamic_module,
    http_user_agent,
    is_paddle_available,
    is_paddlenlp_available,
    is_safetensors_available,
    logging,
    ppdiffusers_bos_dir_download,
    ppdiffusers_url_download,
)
from ..version import VERSION as __version__

if is_paddle_available():
    import paddle
    import paddle.nn as nn

if is_paddlenlp_available():
    from paddlenlp.transformers import PretrainedModel

from .fastdeploy_utils import FastDeployRuntimeModel

TRANSFORMERS_SAFE_WEIGHTS_NAME = "model.safetensors"
TRANSFORMERS_WEIGHTS_NAME = "pytorch_model.bin"

TORCH_INDEX_FILE = "diffusion_pytorch_model.bin"
PADDLE_INDEX_FILE = "model_state.pdparams"

CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "ppdiffusers.utils"
PADDLENLP_DUMMY_MODULES_FOLDER = "paddlenlp.transformers.utils"

logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "ppdiffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "FastDeployRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "paddlenlp.transformers": {
        "PretrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PretrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class TextPipelineOutput(BaseOutput):
    """
    Output class for text pipelines.
    Args:
        prompt (`List[str]` or `str`)
            List of denoised texts.
    """

    texts: Union[List[str], str]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised samples of shape `(batch_size, num_channels, sample_rate)`. Numpy array present the
            denoised audio samples of the diffusion pipeline.
    """

    audios: np.ndarray


def is_safetensors_compatible(filenames, variant=None) -> bool:
    """
    Checking for safetensors compatibility:
    - By default, all models are saved with the default pytorch serialization, so we use the list of default pytorch
      files to know which safetensors files are needed.
    - The model is safetensors compatible only if there is a matching safetensors file for every default pytorch file.
    Converting default pytorch serialized filenames to safetensors serialized filenames:
    - For models from the diffusers library, just replace the ".bin" extension with ".safetensors"
    - For models from the transformers library, the filename changes from "pytorch_model" to "model", and the ".bin"
      extension is replaced with ".safetensors"
    """
    pt_filenames = []

    sf_filenames = set()

    for filename in filenames:
        _, extension = os.path.splitext(filename)

        if extension == ".bin":
            pt_filenames.append(filename)
        elif extension == ".safetensors":
            sf_filenames.add(filename)

    for filename in pt_filenames:
        #  filename = 'foo/bar/baz.bam' -> path = 'foo/bar', filename = 'baz', extention = '.bam'
        path, filename = os.path.split(filename)
        filename, extension = os.path.splitext(filename)

        if filename == "pytorch_model":
            filename = "model"
        elif filename == f"pytorch_model.{variant}":
            filename = f"model.{variant}"
        else:
            filename = filename

        expected_sf_filename = os.path.join(path, filename)
        expected_sf_filename = f"{expected_sf_filename}.safetensors"

        if expected_sf_filename not in sf_filenames:
            logger.warning(f"{expected_sf_filename} not found")
            return False

    return True


def variant_compatible_siblings(info, variant=None) -> Union[List[os.PathLike], str]:
    filenames = set(sibling.rfilename for sibling in info.siblings)
    weight_names = [
        TORCH_WEIGHTS_NAME,
        TORCH_SAFETENSORS_WEIGHTS_NAME,
        TRANSFORMERS_WEIGHTS_NAME,
        TRANSFORMERS_SAFE_WEIGHTS_NAME,
    ]
    # model_pytorch, diffusion_model_pytorch, ...
    weight_prefixes = [w.split(".")[0] for w in weight_names]
    # .bin, .safetensors, ...
    weight_suffixs = [w.split(".")[-1] for w in weight_names]

    variant_file_regex = (
        re.compile(f"({'|'.join(weight_prefixes)})(.{variant}.)({'|'.join(weight_suffixs)})")
        if variant is not None
        else None
    )
    non_variant_file_regex = re.compile(f"{'|'.join(weight_names)}")

    if variant is not None:
        variant_filenames = set(f for f in filenames if variant_file_regex.match(f.split("/")[-1]) is not None)
    else:
        variant_filenames = set()

    non_variant_filenames = set(f for f in filenames if non_variant_file_regex.match(f.split("/")[-1]) is not None)

    usable_filenames = set(variant_filenames)
    for f in non_variant_filenames:
        variant_filename = f"{f.split('.')[0]}.{variant}.{f.split('.')[1]}"
        if variant_filename not in usable_filenames:
            usable_filenames.add(f)

    return usable_filenames, variant_filenames


class DiffusionPipeline(ConfigMixin):
    r"""
    Base class for all models.

    [`DiffusionPipeline`] takes care of storing all components (models, schedulers, processors) for diffusion pipelines
    and handles methods for loading, downloading and saving models as well as a few methods common to all pipelines to:

        - move all PyTorch modules to the device of your choice
        - enabling/disabling the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- name of the config file that will store the class and module names of all
          components of the diffusion pipeline.
        - **_optional_components** (List[`str`]) -- list of all components that are optional so they don't have to be
          passed for the pipeline to function (should be overridden by subclasses).
    """
    config_name = "model_index.json"
    _optional_components = []

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from ppdiffusers import pipelines

        for name, module in kwargs.items():
            # retrieve library
            if module is None:
                register_dict = {name: (None, None)}
            else:
                # TODO (junnyu) support paddlenlp.transformers
                if "paddlenlp" in module.__module__.split(".") or "ppnlp_patch_utils" in module.__module__.split("."):
                    library = "paddlenlp.transformers"
                else:
                    library = module.__module__.split(".")[0]

                # check if the module is a pipeline module
                pipeline_dir = module.__module__.split(".")[-2] if len(module.__module__.split(".")) > 2 else None
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if library not in LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        to_diffusers: bool = None,
    ):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        # TODO (junnyu) support old version
        model_index_dict.pop("_diffusers_paddle_version", None)
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_ppdiffusers_version", None)
        model_index_dict.pop("_module", None)

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_to_diffusers = "to_diffusers" in save_method_signature.parameters

            save_kwargs = {}
            # maybe we donot have torch so we use safe_serialization
            if to_diffusers:
                safe_serialization = True

            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_to_diffusers:
                save_kwargs["to_diffusers"] = to_diffusers

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        commit_message: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this pipeline to a new HuggingFace Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private
            commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
            revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
            create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False.
                If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch.
                If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

        Returns: The url of the commit of your model in the given repository.
        """
        repo_url = create_repo(repo_id, private=private, exist_ok=True)

        # Infer complete repo_id from repo_url
        # Can be different from the input `repo_id` if repo_owner was implicit
        _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)

        repo_id = f"{repo_owner}/{repo_name}"

        # Check if README file already exist in repo
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
            has_readme = True
        except EntryNotFoundError:
            has_readme = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save model
            self.save_pretrained(tmp_dir)
            # Add readme if does not exist
            logger.info("README.md not found, adding the default README.md")
            if not has_readme:
                with open(os.path.join(tmp_dir, "README.md"), "w") as f:
                    f.write(f"---\nlibrary_name: ppdiffusers\n---\n# {repo_id}")

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmp_dir,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
            )

    def to(self, paddle_device: Optional[str] = None, paddle_dtype: Optional[paddle.dtype] = None):
        if paddle_device is None and paddle_dtype is None:
            return self

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, nn.Layer):
                if paddle_device is not None and module.dtype == paddle.float16 and str(paddle_device) in ["cpu"]:
                    logger.warning(
                        "Pipelines loaded with `paddle_dtype=paddle.float16` cannot run with `cpu` device. It"
                        " is not recommended to move them to `cpu` as running them will fail. Please make"
                        " sure to use an accelerator to run the pipeline in inference, due to the lack of"
                        " support for`float16` operations on this device in Paddle. Please, remove the"
                        " `paddle_dtype=paddle.float16` argument, or use another device for inference."
                    )
                kwargs = {}
                if paddle_device is not None:
                    kwargs["device"] = paddle_device
                if paddle_dtype is not None:
                    kwargs["dtype"] = paddle_dtype
                module.to(**kwargs)
        return self

    @property
    def device(self):
        r"""
        Returns:
            `paddle.device`: The paddle device on which the pipeline is located.
        """
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, nn.Layer):
                return module.place
        return "cpu"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a Paddle diffusion pipeline from pre-trained pipeline weights.

        The pipeline is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated).

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* of a pretrained pipeline hosted inside a model repo on
                      https://huggingface.co/ Valid repo ids have to be located under a user or organization name, like
                      `CompVis/ldm-text2im-large-256`.
                    - A path to a *directory* containing pipeline weights saved using
                      [`~DiffusionPipeline.save_pretrained`], e.g., `./my_pipeline_directory/`.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                    This is an experimental feature and is likely to change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* of a custom pipeline hosted inside a model repo on
                      https://huggingface.co/. Valid repo ids have to be located under a user or organization name,
                      like `hf-internal-testing/diffusers-dummy-pipeline`.

                        <Tip>

                         It is required that the model repo has a file, called `pipeline.py` that defines the custom
                         pipeline.

                        </Tip>

                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      https://github.com/huggingface/diffusers/tree/main/examples/community. Valid file names have to
                      match exactly the file name without `.py` located under the above link, *e.g.*
                      `clip_guided_stable_diffusion`.

                        <Tip>

                         Community pipelines are always loaded from the current `main` branch of GitHub.

                        </Tip>

                    - A path to a *directory* containing a custom pipeline, e.g., `./my_pipeline_directory/`.

                        <Tip>

                         It is required that the directory has a file, called `pipeline.py` that defines the custom
                         pipeline.

                        </Tip>

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)

            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            custom_revision (`str`, *optional*, defaults to `"main"` when loading from the Hub and to local version of `diffusers` when loading from GitHub):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a diffusers version when loading a
                custom pipeline from GitHub.
            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information. specify the folder name here.
            return_cached_folder (`bool`, *optional*, defaults to `False`):
                If set to `True`, path to downloaded cached folder will be returned in addition to loaded pipeline.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load - and saveable variables - *i.e.* the pipeline components - of the
                specific pipeline class. The overwritten components are then directly passed to the pipelines
                `__init__` method. See example below for more information.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models), *e.g.* `"runwayml/stable-diffusion-v1-5"`

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        Examples:

        ```py
        >>> from ppdiffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from ppdiffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        runtime_options = kwargs.pop("runtime_options", None)
        return_cached_folder = kwargs.pop("return_cached_folder", False)
        variant = kwargs.pop("variant", None)
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        max_workers = int(kwargs.pop("max_workers", 1))

        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if not os.path.isdir(pretrained_model_name_or_path):
            is_local_dir = False
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                from_hf_hub=from_hf_hub,
            )

            # retrieve all folder_names that contain relevant files (we will ignore `None`` data)
            folder_names = []
            for k, v in config_dict.items():
                # if we pass specifc module, we won't donwload this
                if k in kwargs:
                    continue
                if isinstance(v, list):
                    if None in v:
                        continue
                    folder_names.append(k)

            if from_hf_hub:
                if not local_files_only:
                    info = model_info(
                        pretrained_model_name_or_path,
                        use_auth_token=use_auth_token,
                        revision=revision,
                    )
                    model_filenames, variant_filenames = variant_compatible_siblings(info, variant=variant)
                    model_folder_names = set([os.path.split(f)[0] for f in model_filenames])

                    if revision in DEPRECATED_REVISION_ARGS and version.parse(
                        version.parse(__version__).base_version
                    ) >= version.parse("0.15.0"):
                        info = model_info(
                            pretrained_model_name_or_path,
                            use_auth_token=use_auth_token,
                            revision=None,
                        )
                        comp_model_filenames, _ = variant_compatible_siblings(info, variant=revision)
                        comp_model_filenames = [
                            ".".join(f.split(".")[:1] + f.split(".")[2:]) for f in comp_model_filenames
                        ]

                        if set(comp_model_filenames) == set(model_filenames):
                            warnings.warn(
                                f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{variant}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
                                FutureWarning,
                            )
                        else:
                            warnings.warn(
                                f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added.",
                                FutureWarning,
                            )

                    # all filenames compatible with variant will be added
                    allow_patterns = list(model_filenames)

                    # allow all patterns from non-model folders
                    # this enables downloading schedulers, tokenizers, ...
                    allow_patterns += [os.path.join(k, "*") for k in folder_names if k not in model_folder_names]
                    # also allow downloading config.jsons with the model
                    allow_patterns += [os.path.join(k, "*.json") for k in model_folder_names]

                    allow_patterns += [
                        SCHEDULER_CONFIG_NAME,
                        CONFIG_NAME,
                        cls.config_name,
                        CUSTOM_PIPELINE_FILE_NAME,
                    ]

                    if is_safetensors_available() and is_safetensors_compatible(model_filenames, variant=variant):
                        ignore_patterns = ["*.bin", "*.msgpack", "*.onnx", "*.pb"]

                        safetensors_variant_filenames = set(
                            [f for f in variant_filenames if f.endswith(".safetensors")]
                        )
                        safetensors_model_filenames = set([f for f in model_filenames if f.endswith(".safetensors")])
                        if (
                            len(safetensors_variant_filenames) > 0
                            and safetensors_model_filenames != safetensors_variant_filenames
                        ):
                            logger.warn(
                                f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                            )

                    else:
                        ignore_patterns = ["*.safetensors", "*.msgpack", "*.onnx", "*.pb"]

                        bin_variant_filenames = set([f for f in variant_filenames if f.endswith(".bin")])
                        bin_model_filenames = set([f for f in model_filenames if f.endswith(".bin")])
                        if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
                            logger.warn(
                                f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                            )

                else:
                    # allow everything since it has to be downloaded anyways
                    ignore_patterns = allow_patterns = None

                if cls != DiffusionPipeline:
                    requested_pipeline_class = cls.__name__
                else:
                    requested_pipeline_class = config_dict.get("_class_name", cls.__name__)
                user_agent = {"pipeline_class": requested_pipeline_class}
                if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
                    user_agent["custom_pipeline"] = custom_pipeline

                user_agent = http_user_agent(user_agent)

                # download all allow_patterns
                cached_folder = snapshot_download(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                    user_agent=user_agent,
                    max_workers=max_workers,
                )
            else:
                if cls == DiffusionPipeline:
                    is_fastdeploy_model = "fastdeploy" in config_dict.get("_class_name", "").lower()
                else:
                    is_fastdeploy_model = "fastdeploy" in cls.__name__.lower()

                cached_folder = ppdiffusers_bos_dir_download(
                    pretrained_model_name_or_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    resume_download=resume_download,
                    folder_names=folder_names,
                    variant=variant,
                    max_workers=max_workers,
                    is_fastdeploy_model=is_fastdeploy_model,
                    local_files_only=local_files_only,
                )
        else:
            is_local_dir = True
            cached_folder = pretrained_model_name_or_path
            config_dict = cls.load_config(cached_folder)

        # retrieve which subfolders should load variants
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(path.split(".")[1] == variant for path in os.listdir(folder_path))
                if variant_exists:
                    model_variants[folder] = variant

        # 2. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        if custom_pipeline is not None:
            if custom_pipeline.endswith(".py"):
                path = Path(custom_pipeline)
                # decompose into folder & file
                file_name = path.name
                custom_pipeline = path.parent.absolute()
            else:
                file_name = CUSTOM_PIPELINE_FILE_NAME

            pipeline_class = get_class_from_dynamic_module(
                custom_pipeline, module_file=file_name, cache_dir=cache_dir, revision=custom_revision
            )
        elif cls != DiffusionPipeline:
            pipeline_class = cls
        else:
            ppdiffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            pipeline_class = getattr(ppdiffusers_module, config_dict["_class_name"])

        # To be removed in 1.0.0
        _ppdiffusers_version = (
            config_dict["_diffusers_paddle_version"]
            if "_diffusers_paddle_version" in config_dict
            else config_dict["_ppdiffusers_version"]
        )
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
            version.parse(_ppdiffusers_version).base_version
        ) <= version.parse("0.5.1"):
            from ppdiffusers import StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

            deprecation_message = (
                "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
                " {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
                " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
                " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
                f" checkpoint {pretrained_model_name_or_path} to the format of"
                " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
                " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
            )
            deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # # Special case: safety_checker must be loaded separately when using `from_diffusers`
        # if from_diffusers and "safety_checker" in init_dict and "safety_checker" not in passed_class_obj:
        #     raise NotImplementedError(
        #         "The safety checker cannot be automatically loaded when loading weights `from_diffusers`."
        #         " Please, pass `safety_checker=None` to `from_pretrained`, and load the safety checker"
        #         " separately if you need it."
        #     )

        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )
        # import it here to avoid circular import
        from ppdiffusers import ModelMixin, pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            # support old model_index.json and hf model_index.json
            if library_name in ["diffusers_paddle", "diffusers"]:
                library_name = "ppdiffusers"
            if library_name == "transformers":
                library_name = "paddlenlp.transformers"
            class_name = class_name.replace("Flax", "")

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                else:
                    logger.warning(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)

                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

            if loaded_sub_model is None:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                if load_method_name is None:
                    none_module = class_obj.__module__
                    is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
                        PADDLENLP_DUMMY_MODULES_FOLDER
                    )
                    if is_dummy_path and "dummy" in none_module:
                        # call class_obj for nice error message of missing requirements
                        class_obj()

                    raise ValueError(
                        f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
                        f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
                    )

                load_method = getattr(class_obj, load_method_name)
                loading_kwargs = {}

                if issubclass(class_obj, FastDeployRuntimeModel):
                    loading_kwargs["runtime_options"] = (
                        runtime_options.get(name, None) if isinstance(runtime_options, dict) else runtime_options
                    )

                if issubclass(class_obj, (PretrainedModel, ModelMixin)):
                    loading_kwargs["variant"] = model_variants.pop(name, None)
                    loading_kwargs["from_diffusers"] = from_diffusers
                    loading_kwargs["paddle_dtype"] = paddle_dtype

                loaded_sub_model = None
                try:
                    # check if the module is in a subdirectory
                    if os.path.isdir(os.path.join(cached_folder, name)):
                        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
                    else:
                        # else load from the root directory
                        loaded_sub_model = load_method(cached_folder, **loading_kwargs)
                except Exception as e:
                    # (TODO, junnyu)
                    # if we cant find this file, we will try to download this
                    if not local_files_only and not is_local_dir and not from_hf_hub:
                        loaded_sub_model = load_method(
                            pretrained_model_name_or_path + "/" + name, cache_dir=cache_dir, **loading_kwargs
                        )
                    if loaded_sub_model is None:
                        raise ValueError(
                            f"We cant load '{name}' from {pretrained_model_name_or_path} or {cached_folder}! \n {e} "
                        )
            # paddlenlp's model is in training mode not eval mode
            # if isinstance(loaded_sub_model, PretrainedModel):
            # if paddle_dtype is not None and next(loaded_sub_model.named_parameters())[1].dtype != paddle_dtype:
            #     loaded_sub_model = loaded_sub_model.to(dtype=paddle_dtype)
            # loaded_sub_model.eval()

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 5. (TODO, junnyu) make sure all modules are in eval mode and cast dtype
        for name, _module in init_kwargs.items():
            if isinstance(_module, nn.Layer):
                _module.eval()
                if paddle_dtype is not None and _module.dtype != paddle_dtype:
                    _module.to(dtype=paddle_dtype)
            elif isinstance(_module, (tuple, list)):
                if isinstance(_module[0], nn.Layer):
                    for _submodule in _module:
                        _submodule.eval()
                        if paddle_dtype is not None and _submodule.dtype != paddle_dtype:
                            _submodule.to(dtype=paddle_dtype)

        # 6. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        if return_cached_folder:
            return model, cached_folder
        return model

    @classmethod
    def from_pretrained_original_ckpt(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        from .stable_diffusion.convert_from_ckpt import (
            load_pipeline_from_original_stable_diffusion_ckpt,
        )

        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        cache_dir = kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        original_config_file = kwargs.pop("original_config_file", None)
        requires_safety_checker = kwargs.pop("requires_safety_checker", False)
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_path = pretrained_model_name_or_path
        elif pretrained_model_name_or_path.startswith("http://") or pretrained_model_name_or_path.startswith(
            "https://"
        ):
            checkpoint_path = ppdiffusers_url_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
            )
        else:
            raise EnvironmentError(f"Please check your {pretrained_model_name_or_path}.")
        pipeline = load_pipeline_from_original_stable_diffusion_ckpt(
            checkpoint_path=checkpoint_path,
            original_config_file=original_config_file,
            paddle_dtype=paddle_dtype,
            requires_safety_checker=requires_safety_checker,
            cls=cls,
            **kwargs,
        )

        return pipeline

    @staticmethod
    def _get_signature_keys(obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - set(["self"])
        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.

        Examples:

        ```py
        >>> from ppdiffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```

        Returns:
            A dictionary containing all the modules needed to initialize the pipeline.
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        r"""
        Enable memory efficient attention as implemented in xformers.

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.

        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import paddle
        >>> from ppdiffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", paddle_dtype=paddle.float16)
        >>> pipe.enable_xformers_memory_efficient_attention("cutlass")
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[str] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Layer):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for module_name in module_names:
            module = getattr(self, module_name)
            if isinstance(module, nn.Layer):
                fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for module_name in module_names:
            module = getattr(self, module_name)
            if isinstance(module, nn.Layer) and hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        if hasattr(self, "vae"):
            self.vae.enable_slicing()
        if hasattr(self, "vqvae"):
            self.vqvae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if hasattr(self, "vae"):
            self.vae.disable_slicing()
        if hasattr(self, "vqvae"):
            self.vqvae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        if hasattr(self, "vae"):
            self.vae.enable_tiling()
        if hasattr(self, "vqvae"):
            self.vqvae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if hasattr(self, "vae"):
            self.vae.disable_tiling()
        if hasattr(self, "vqvae"):
            self.vqvae.disable_tiling()
