# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from functools import partial
from typing import Callable, Optional, Union

import paddle
import paddle.nn as nn

from ..utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PADDLE_WEIGHTS_NAME,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    TORCH_SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    is_safetensors_available,
    is_torch_available,
    logging,
    smart_load,
)
from ..version import VERSION as __version__
from .modeling_pytorch_paddle_utils import (
    convert_paddle_state_dict_to_pytorch,
    convert_pytorch_state_dict_to_paddle,
)

logger = logging.get_logger(__name__)


if is_safetensors_available():
    from safetensors.numpy import save_file as safetensors_numpy_save_file

    if is_torch_available():
        from safetensors.torch import save_file as safetensors_torch_save_file

if is_torch_available():
    import torch


def get_parameter_device(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].place
    except StopIteration:
        return paddle.get_device()


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    try:
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        return parameter._dtype


def convert_state_dict(state_dict, framework="torch"):
    if framework in ["torch", "pt"]:
        # support bfloat16
        newstate_dict = {}
        for k, v in state_dict.items():
            if v.dtype == paddle.bfloat16:
                v = v.cast("float32").cpu().numpy()
                newstate_dict[k] = torch.tensor(v).to(torch.bfloat16)
            else:
                newstate_dict[k] = torch.tensor(v.cpu().numpy())
        return newstate_dict
    elif framework in ["numpy", "np"]:
        state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
        return state_dict
    elif framework in ["paddle", "pd"]:
        state_dict = {k: paddle.to_tensor(v, place="cpu") for k, v in state_dict.items()}
        return state_dict
    else:
        raise NotImplementedError(f"Not Implemented {framework} framework!")


class ModelMixin(nn.Layer):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the configuration of the models and handles methods for loading, downloading
    and saving models.

        - **config_name** ([`str`]) -- A filename under which the model should be stored when calling
          [`~models.ModelMixin.save_pretrained`].
    """
    config_name = CONFIG_NAME
    _automatically_saved_args = ["_ppdiffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False

    def __init__(self):
        super().__init__()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(
            hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing
            for m in self.sublayers(include_self=True)
        )

    def enable_gradient_checkpointing(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[str] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Layer):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, nn.Layer):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        r"""
        Enable memory efficient attention as implemented in xformers.

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.

        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None`

        Examples:

        ```py
        >>> import paddle
        >>> from ppdiffusers import UNet2DConditionModel

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", paddle_dtype=paddle.float16
        ... )
        >>> model.enable_xformers_memory_efficient_attention()
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        to_diffusers: Optional[bool] = None,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `paddle.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            to_diffusers (`bool`, *optional*, defaults to `False`):
                If specified, weights are saved in the format of torch. eg. linear need transpose.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Only when `to_diffusers` is True, Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        # save ignore lora_weights
        fn = lambda k: ".lora_" in k or ".alpha" in k
        state_dict = {k: v for k, v in state_dict.items() if not fn(k)}

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        save_function = safetensors_torch_save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        save_function = safetensors_numpy_save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")
                    weights_name = _add_variant(TORCH_SAFETENSORS_WEIGHTS_NAME, variant)
                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    weights_name = _add_variant(TORCH_WEIGHTS_NAME, variant)
                    state_dict = convert_state_dict(state_dict, framework="torch")

                state_dict = convert_paddle_state_dict_to_pytorch(state_dict, model_to_save)
            else:
                save_function = paddle.save
                weights_name = _add_variant(PADDLE_WEIGHTS_NAME, variant)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weights_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
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
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            from_diffusers (`bool`, *optional*, defaults to `False`):
                Load the model weights from a torch checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin.
                model_state.<variant>.pdparams.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        """
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        ignore_keys = kwargs.pop("ignore_keys", None)
        variant = kwargs.pop("variant", None)

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "paddle",
        }

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path
        config, unused_kwargs = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            from_hf_hub=from_hf_hub,  # whether or not from_hf_hub
            **kwargs,
        )
        model = cls.from_config(config, **unused_kwargs)

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model
        model_file = None
        if from_diffusers:
            if is_safetensors_available():
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(TORCH_SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                except Exception:  # noqa: E722
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(TORCH_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
        else:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(PADDLE_WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                from_hf_hub=from_hf_hub,
            )
        assert model_file is not None

        # try load model_file with paddle / torch / safetensor
        state_dict = smart_load(model_file)

        # convert weights
        if from_diffusers:
            state_dict = convert_pytorch_state_dict_to_paddle(state_dict, model)

        # remove keys
        if ignore_keys is not None:
            keys = list(state_dict.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        logger.warning("Deleting key {} from state_dict.".format(k))
                        del state_dict[k]

        dtype = set(v.dtype for v in state_dict.values())
        if len(dtype) > 1 and paddle.float32 not in dtype:
            raise ValueError(
                f"The weights of the model file {model_file} have a mixture of incompatible dtypes {dtype}. Please"
                f" make sure that {model_file} weights have only one dtype."
            )
        elif len(dtype) > 1 and paddle.float32 in dtype:
            dtype = paddle.float32
        else:
            dtype = dtype.pop()
        model = model.to(dtype=dtype)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        if paddle_dtype is not None and not isinstance(paddle_dtype, paddle.dtype):
            raise ValueError(
                f"{paddle_dtype} needs to be of type `paddle.dtype`, e.g. `paddle.float16`, but is {type(paddle_dtype)}."
            )
        elif paddle_dtype is not None:
            model = model.to(dtype=paddle_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
    ):
        # Retrieve missing & unexpected_keys
        model_state_dict = model.state_dict()
        loaded_keys = [k for k in state_dict.keys()]

        expected_keys = list(model_state_dict.keys())

        original_loaded_keys = loaded_keys

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = model

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if model_key in model_state_dict and list(state_dict[checkpoint_key].shape) != list(
                        model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            # Whole checkpoint
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = ""
            model_to_load.load_dict(state_dict)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self):
        """
        `paddle.place`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> paddle.dtype:
        """
        `paddle.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_sublayers(include_self=True)
                if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if not p.stop_gradient or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if not p.stop_gradient or not only_trainable)


def unfreeze_params(params):
    for param in params:
        param.stop_gradient = False


def freeze_params(params):
    for param in params:
        param.stop_gradient = True


def unfreeze_model(model: nn.Layer):
    for param in model.parameters():
        param.stop_gradient = False


def freeze_model(model: nn.Layer):
    for param in model.parameters():
        param.stop_gradient = True


def unwrap_model(model: nn.Layer) -> nn.Layer:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`nn.Layer`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "_layers"):
        return unwrap_model(model._layers)
    else:
        return model
