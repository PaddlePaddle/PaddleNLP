# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .download_utils import ppdiffusers_bos_download

from .utils import FASTDEPLOY_WEIGHTS_NAME, FASTDEPLOY_MODEL_NAME, is_fastdeploy_available, logging

if is_fastdeploy_available():
    import fastdeploy as fd

logger = logging.get_logger(__name__)


class FastDeployRuntimeModel:

    def __init__(self, model=None, **kwargs):
        logger.info(
            "`ppdiffusers.FastDeployRuntimeModel` is experimental and might change in the future."
        )
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name",
                                            "inference.pdmodel")
        self.latest_params_name = kwargs.get("latest_params_name",
                                             "inference.pdiparams")

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.infer(inputs)

    @staticmethod
    def load_model(model_path: Union[str, Path],
                   params_path: Union[str, Path],
                   runtime_option=None):
        """
        Loads an FastDeploy Inference Model with fastdeploy.RuntimeOption

        Arguments:
            model_path (`str` or `Path`):
                Model path from which to load
            params_path (`str` or `Path`):
                Params path from which to load
            runtime_option(fd.RuntimeOption, *optional*):
                The RuntimeOption of fastdeploy to initialize the fastdeploy runtime. Default setting
                the device to cpu and the backend to paddle inference
        """
        option = runtime_option
        if option is None or not isinstance(runtime_option, fd.RuntimeOption):
            logger.info(
                "No fastdeploy.RuntimeOption specified, using CPU device and paddle inference backend."
            )
            option = fd.RuntimeOption()
            option.use_paddle_backend()
            option.use_cpu()
        option.set_model_path(model_path, params_path)
        return fd.Runtime(option)

    def _save_pretrained(self,
                         save_directory: Union[str, Path],
                         model_file_name: Optional[str] = None,
                         params_file_name: Optional[str] = None,
                         **kwargs):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] class method. It will always save the
        latest_model_name.

        Arguments:
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            model_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdmodel"` to `model_file_name`. This allows you to save the
                model with a different name.
            params_file_name(`str`, *optional*):
                Overwrites the default model file name from `"inference.pdiparams"` to `params_file_name`. This allows you to save the
                model with a different name.
        """

        model_file_name = model_file_name if model_file_name is not None else FASTDEPLOY_MODEL_NAME
        params_file_name = params_file_name if params_file_name is not None else FASTDEPLOY_WEIGHTS_NAME

        src_model_path = self.model_save_dir.joinpath(self.latest_model_name)
        dst_model_path = Path(save_directory).joinpath(model_file_name)

        src_params_path = self.model_save_dir.joinpath(self.latest_params_name)
        dst_params_path = Path(save_directory).joinpath(params_file_name)
        try:
            shutil.copyfile(src_model_path, dst_model_path)
            shutil.copyfile(src_params_path, dst_params_path)
        except shutil.SameFileError:
            pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Save a model to a directory, so that it can be re-loaded using the [`~OnnxModel.from_pretrained`] class
        method.:

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        cache_dir: Optional[str] = None,
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        runtime_option: Optional["fastdeploy.RuntimeOption"] = None,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.

        Arguments:
            pretrained_model_name_or_path (`str` or `Path`):
                Directory from which to load
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            model_file_name(`str`):
                Overwrites the default model file name from `"inference.pdmodel"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            params_file_name(`str`):
                Overwrites the default params file name from `"inference.pdiparams"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            runtime_option(`fastdeploy.RuntimeOption`, *optional*):
                The RuntimeOption of fastdeploy.
            kwargs (`Dict`, *optional*):
                kwargs will be passed to the model during initialization
        """
        model_file_name = model_file_name if model_file_name is not None else FASTDEPLOY_MODEL_NAME
        params_file_name = params_file_name if params_file_name is not None else FASTDEPLOY_WEIGHTS_NAME
        # load model from local directory
        if os.path.isdir(pretrained_model_name_or_path):
            model = FastDeployRuntimeModel.load_model(
                os.path.join(pretrained_model_name_or_path, model_file_name),
                os.path.join(pretrained_model_name_or_path, params_file_name),
                runtime_option=runtime_option,
            )
            kwargs["model_save_dir"] = Path(pretrained_model_name_or_path)
        # load model from hub
        else:
            # download model
            model_cache_path = ppdiffusers_bos_download(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                filename=model_file_name,
            )
            # download params
            params_cache_path = ppdiffusers_bos_download(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                filename=params_file_name,
            )
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name
            kwargs["latest_params_name"] = Path(params_cache_path).name
            model = FastDeployRuntimeModel.load_model(
                model_cache_path,
                params_cache_path,
                runtime_option=runtime_option)
        return cls(model=model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        force_download: bool = True,
        use_auth_token: Optional[str] = None,
        **model_kwargs,
    ):
        revision = None
        if len(str(model_id).split("@")) == 2:
            model_id, revision = model_id.split("@")

        return cls._from_pretrained(
            model_id=model_id,
            revision=revision,
            force_download=force_download,
            use_auth_token=use_auth_token,
            **model_kwargs,
        )
