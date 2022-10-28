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
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn

from .download_utils import ppdiffusers_bos_download
from requests import HTTPError

from .utils import CONFIG_NAME, PPDIFFUSERS_CACHE, DOWNLOAD_SERVER, WEIGHTS_NAME, logging

logger = logging.get_logger(__name__)


def freeze_params(params):
    for param in params:
        param.stop_gradient = True


# device
def get_parameter_device(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].place
    except StopIteration:
        return paddle.get_device()


def get_parameter_dtype(parameter: nn.Layer):
    try:
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        return paddle.get_default_dtype()


def load_dict(checkpoint_file: Union[str, os.PathLike],
              return_numpy: bool = True):
    """
    Reads a Paddle checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        return paddle.load(checkpoint_file, return_numpy=return_numpy)
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned.")
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from Paddle checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a Paddle model from a TF 2.0 checkpoint, please set from_tf=True."
            )


class ModelMixin(nn.Layer):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the configuration of the models and handles methods for loading, downloading
    and saving models.

        - **config_name** ([`str`]) -- A filename under which the model should be stored when calling
          [`~modeling_utils.ModelMixin.save_pretrained`].
    """
    config_name = CONFIG_NAME
    _automatically_saved_args = [
        "_ppdiffusers_version", "_class_name", "_name_or_path"
    ]
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
            for m in self.modules())

    def enable_gradient_checkpointing(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing."
            )
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Callable = paddle.save,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~modeling_utils.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `paddle.save` by another method.
        """
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory)

        # Save the model
        state_dict = model_to_save.state_dict()

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            if filename.startswith(WEIGHTS_NAME[:-4]) and os.path.isfile(
                    full_filename) and is_main_process:
                os.remove(full_filename)

        # Save the model
        save_function(state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        logger.info(
            f"Model weights saved in {os.path.join(save_directory, WEIGHTS_NAME)}"
        )

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]],
            **kwargs):
        r"""
        Instantiate a pretrained paddle model from a pre-trained model configuration.

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

            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        """
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        output_loading_info = kwargs.pop("output_loading_info", False)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        subfolder = kwargs.pop("subfolder", None)

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # Load model
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                # Load from a Paddle checkpoint
                model_file = os.path.join(pretrained_model_name_or_path,
                                          WEIGHTS_NAME)
            elif subfolder is not None and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder,
                                 WEIGHTS_NAME)):
                model_file = os.path.join(pretrained_model_name_or_path,
                                          subfolder, WEIGHTS_NAME)
            else:
                raise EnvironmentError(
                    f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                model_file = ppdiffusers_bos_download(
                    pretrained_model_name_or_path,
                    filename=WEIGHTS_NAME,
                    subfolder=subfolder,
                )
            except HTTPError as err:
                raise EnvironmentError(
                    "There was a specific connection error when trying to load"
                    f" {pretrained_model_name_or_path}:\n{err}")
            except ValueError:
                raise EnvironmentError(
                    f"We couldn't connect to '{DOWNLOAD_SERVER}' to load this model, couldn't find it"
                    f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                    f" directory containing a file named {WEIGHTS_NAME} or"
                    " \nCheckout your internet connection or see how to run the library in"
                    " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
                )
            except EnvironmentError:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                    "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                    f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {WEIGHTS_NAME}")

        model, unused_kwargs = cls.from_config(
            config_path,
            return_unused_kwargs=True,
            subfolder=subfolder,
            **kwargs,
        )
        state_dict = load_dict(model_file, return_numpy=True)

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

        if paddle_dtype is not None and not isinstance(paddle_dtype,
                                                       paddle.dtype):
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

                    if (model_key in model_state_dict
                            and list(state_dict[checkpoint_key].shape) !=
                            model_state_dict[model_key].shape):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape,
                             model_state_dict[model_key].shape))
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
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model).")
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}."
            )
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
                " without further training.")
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join([
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ])
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference.")

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

    def num_parameters(self,
                       only_trainable: bool = False,
                       exclude_embeddings: bool = False) -> int:
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
                parameter for name, parameter in self.named_parameters()
                if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters
                       if not p.stop_gradient or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters()
                       if not p.stop_gradient or not only_trainable)


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
