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


import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from ..utils import (
    DIFFUSERS_CACHE,
    FASTDEPLOY_MODEL_NAME,
    FASTDEPLOY_WEIGHTS_NAME,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    _add_variant,
    _get_model_file,
    is_fastdeploy_available,
    is_paddle_available,
    logging,
)
from ..version import VERSION as __version__

__all__ = ["FastDeployRuntimeModel"]

if is_paddle_available():
    import paddle

if is_fastdeploy_available():
    import fastdeploy as fd

    def fdtensor2pdtensor(fdtensor: fd.C.FDTensor):
        dltensor = fdtensor.to_dlpack()
        pdtensor = paddle.utils.dlpack.from_dlpack(dltensor)
        return pdtensor

    def pdtensor2fdtensor(pdtensor: paddle.Tensor, name: str = "", share_with_raw_ptr=False):
        if not share_with_raw_ptr:
            dltensor = paddle.utils.dlpack.to_dlpack(pdtensor)
            return fd.C.FDTensor.from_dlpack(name, dltensor)
        else:
            return fd.C.FDTensor.from_external_data(
                name,
                pdtensor.data_ptr(),
                pdtensor.shape,
                pdtensor.dtype.name,
                str(pdtensor.place),
                int(pdtensor.place.gpu_device_id()),
            )


logger = logging.get_logger(__name__)


class FastDeployRuntimeModel:
    def __init__(self, model=None, **kwargs):
        logger.info("`ppdiffusers.FastDeployRuntimeModel` is experimental and might change in the future.")
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)
        self.latest_model_name = kwargs.get("latest_model_name", FASTDEPLOY_MODEL_NAME)
        self.latest_params_name = kwargs.get("latest_params_name", FASTDEPLOY_WEIGHTS_NAME)

    def zero_copy_infer(self, prebinded_inputs: dict, prebinded_outputs: dict, share_with_raw_ptr=True, **kwargs):
        """
        Execute inference without copying data from cpu to gpu.

        Arguments:
            kwargs (`dict(name, paddle.Tensor)`):
                An input map from name to tensor.
        Return:
            List of output tensor.
        """
        for inputs_name, inputs_tensor in prebinded_inputs.items():
            input_fdtensor = pdtensor2fdtensor(inputs_tensor, inputs_name, share_with_raw_ptr=share_with_raw_ptr)
            self.model.bind_input_tensor(inputs_name, input_fdtensor)

        for outputs_name, outputs_tensor in prebinded_outputs.items():
            output_fdtensor = pdtensor2fdtensor(outputs_tensor, outputs_name, share_with_raw_ptr=share_with_raw_ptr)
            self.model.bind_output_tensor(outputs_name, output_fdtensor)

            self.model.zero_copy_infer()

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.infer(inputs)

    @staticmethod
    def load_model(
        model_path: Union[str, Path],
        params_path: Union[str, Path],
        runtime_options: Optional["fd.RuntimeOption"] = None,
    ):
        """
        Loads an FastDeploy Inference Model with fastdeploy.RuntimeOption

        Arguments:
            model_path (`str` or `Path`):
                Model path from which to load
            params_path (`str` or `Path`):
                Params path from which to load
            runtime_options (fd.RuntimeOption, *optional*):
                The RuntimeOption of fastdeploy to initialize the fastdeploy runtime. Default setting
                the device to cpu and the backend to paddle inference
        """
        option = runtime_options
        if option is None or not isinstance(runtime_options, fd.RuntimeOption):
            logger.info("No fastdeploy.RuntimeOption specified, using CPU device and paddle inference backend.")
            option = fd.RuntimeOption()
            option.use_paddle_backend()
            option.use_cpu()
        option.set_model_path(model_path, params_path)
        return fd.Runtime(option)

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        **kwargs
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~FastDeployRuntimeModel.from_pretrained`] class method. It will always save the
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
        Save a model to a directory, so that it can be re-loaded using the [`~FastDeployRuntimeModel.from_pretrained`] class
        method.:

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # saving model weights/files
        self._save_pretrained(save_directory, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[str] = None,
        subfolder: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        runtime_options: Optional["fd.RuntimeOption"] = None,
        from_hf_hub: Optional[bool] = False,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        user_agent: Union[Dict, str, None] = None,
        **kwargs,
    ):
        """
        Load a model from a directory or the HF Hub.

        Arguments:
            pretrained_model_name_or_path (`str` or `Path`):
                Directory from which to load
            model_file_name (`str`):
                Overwrites the default model file name from `"inference.pdmodel"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            params_file_name (`str`):
                Overwrites the default params file name from `"inference.pdiparams"` to `file_name`. This allows you to load
                different model files from the same repository or directory.
            use_auth_token (`str` or `bool`):
                Is needed to load models from a private or gated repository
            revision (`str`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id
            cache_dir (`Union[str, Path]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            runtime_options (`fastdeploy.RuntimeOption`, *optional*):
                The RuntimeOption of fastdeploy.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
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
                runtime_options=runtime_options,
            )
            kwargs["model_save_dir"] = Path(pretrained_model_name_or_path)
        # load model from hub or paddle bos
        else:
            model_cache_path = _get_model_file(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weights_name=model_file_name,
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

            params_cache_path = _get_model_file(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                weights_name=params_file_name,
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
            kwargs["model_save_dir"] = Path(model_cache_path).parent
            kwargs["latest_model_name"] = Path(model_cache_path).name
            kwargs["latest_params_name"] = Path(params_cache_path).name
            model = FastDeployRuntimeModel.load_model(
                model_cache_path, params_cache_path, runtime_options=runtime_options
            )
        return cls(model=model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_file_name: Optional[str] = None,
        params_file_name: Optional[str] = None,
        runtime_options: Optional["fd.RuntimeOption"] = None,
        **kwargs,
    ):
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "fastdeploy",
        }

        return cls._from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_file_name=_add_variant(model_file_name, variant),
            params_file_name=_add_variant(params_file_name, variant),
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            force_download=force_download,
            cache_dir=cache_dir,
            runtime_options=runtime_options,
            from_hf_hub=from_hf_hub,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            user_agent=user_agent,
            **kwargs,
        )
