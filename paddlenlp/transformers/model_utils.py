# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import io
import json
import os
import six
import logging

import paddle
from paddle.nn import Layer
# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddlenlp.utils.downloader import get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger

from .generation_utils import GenerationMixin
from .utils import InitTrackerMeta, fn_args_to_dict

__all__ = [
    'PretrainedModel',
    'register_base_model',
]


def register_base_model(cls):
    """
    Add a `base_model_class` attribute for the base class of decorated class,
    representing the base model class in derived classes of the same architecture.
    Args:
        cls (class): the name of the model
    """
    base_cls = cls.__bases__[0]
    assert issubclass(
        base_cls, PretrainedModel
    ), "`register_base_model` should be used on subclasses of PretrainedModel."
    base_cls.base_model_class = cls
    return cls


@six.add_metaclass(InitTrackerMeta)
class PretrainedModel(Layer, GenerationMixin):
    """
    The base class for all pretrained models. It provides some attributes and
    common methods for all pretrained models, including attributes `init_config`,
    `config` for initialized arguments and methods for saving, loading.
    It also includes some class attributes (should be set by derived classes):
    - `model_config_file` (str): represents the file name for saving and loading
      model configuration, it's value is `model_config.json`.
    - `resource_files_names` (dict): use this to map resources to specific file
      names for saving and loading.
    - `pretrained_resource_files_map` (dict): The dict has the same keys as
      `resource_files_names`, the values are also dict mapping specific pretrained
      model name to URL linking to pretrained model.
    - `pretrained_init_configuration` (dict): The dict has pretrained model names
      as keys, and the values are also dict preserving corresponding configuration
      for model initialization.
    
    - `base_model_prefix` (str): represents the the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fileds as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {}
    base_model_prefix = ""

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the prtrained model instance.
        """
        init_dict = fn_args_to_dict(original_init, *((self, ) + args), **kwargs)
        self.config = init_dict

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        return list(self.pretrained_init_configuration.keys())

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None  # Overwrite for models with output embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Instantiate an instance of `PretrainedModel` from a predefined
        model specified by name or path.
        Args:
            pretrained_model_name_or_path (str): A name of or a file path to a
                pretrained model.
            *args (tuple): position arguments for `__init__`. If provide, use
                this as position argument values for model initialization.
            **kwargs (dict): keyword arguments for `__init__`. If provide, use
                this to update pre-defined keyword argument values for model
                initialization.
        Returns:
            PretrainedModel: An instance of PretrainedModel.
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[
                    pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(
                cls.pretrained_init_configuration[
                    pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path,
                                                  file_name)
                    resource_files[file_id] = full_file_name
                resource_files["model_config_file"] = os.path.join(
                    pretrained_model_name_or_path, cls.model_config_file)
            else:
                raise ValueError(
                    "Calling {}.from_pretrained() with a model identifier or the "
                    "path to a directory instead. The supported model "
                    "identifiers are as follows: {}".format(
                        cls.__name__, cls.pretrained_init_configuration.keys()))

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)

        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            path = os.path.join(default_root, file_path.split('/')[-1])
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
            elif os.path.exists(path):
                logger.info("Already cached %s" % path)
                resolved_resource_files[file_id] = path
            else:
                logger.info("Downloading %s and saved to %s" %
                            (file_path, default_root))
                resolved_resource_files[file_id] = get_path_from_url(
                    file_path, default_root)

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file",
                                                        None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class",
                                     cls.base_model_class.__name__)

        # Check if the loaded config matches the current model class's __init__
        # arguments. If not match, the loaded config is for the base model class.
        if init_class == cls.base_model_class.__name__:
            base_args = init_args
            base_kwargs = init_kwargs
            derived_args = ()
            derived_kwargs = {}
            base_arg_index = None
        else:  # extract config for base model
            derived_args = list(init_args)
            derived_kwargs = init_kwargs
            base_arg = None
            for i, arg in enumerate(init_args):
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = i
                    base_arg = arg
                    break
            for arg_name, arg in init_kwargs.items():
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop(
                        "init_class") == cls.base_model_class.__name__, (
                            "pretrained base model should be {}"
                        ).format(cls.base_model_class.__name__)
                    base_arg_index = arg_name
                    base_arg = arg
                    break
            base_args = base_arg.pop("init_args", ())
            base_kwargs = base_arg
        if cls == cls.base_model_class:
            # Update with newly provided args and kwargs for base model
            base_args = base_args if not args else args
            base_kwargs.update(kwargs)
            model = cls(*base_args, **base_kwargs)
        else:
            # Update with newly provided args and kwargs for derived model
            base_model = cls.base_model_class(*base_args, **base_kwargs)
            if base_arg_index is not None:
                derived_args[base_arg_index] = base_model
            else:
                derived_args = (base_model, )  # assume at the first position
            derived_args = derived_args if not args else args
            derived_kwargs.update(kwargs)
            model = cls(*derived_args, **derived_kwargs)

        # Maybe need more ways to load resources.
        weight_path = list(resolved_resource_files.values())[0]
        assert weight_path.endswith(
            ".pdparams"), "suffix of weight must be .pdparams"
        state_dict = paddle.load(weight_path)

        # Make sure we are able to load base models as well as derived models
        # (with heads)
        start_prefix = ""
        model_to_load = model
        state_to_load = state_dict
        unexpected_keys = []
        missing_keys = []
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # base model
            state_to_load = {}
            start_prefix = cls.base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(cls.base_model_prefix):
                    state_to_load[k[len(start_prefix):]] = v
                else:
                    unexpected_keys.append(k)
        if hasattr(model, cls.base_model_prefix) and not any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # derived model (base model with heads)
            model_to_load = getattr(model, cls.base_model_prefix)
            for k in model.state_dict().keys():
                if not k.startswith(cls.base_model_prefix):
                    missing_keys.append(k)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".
                format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".
                        format(model.__class__.__name__, unexpected_keys))
        model_to_load.set_state_dict(state_to_load)
        if paddle.in_dynamic_mode():
            return model
        return model, state_to_load

    def save_pretrained(self, save_directory):
        """
        Save model configuration and related resources (model state) to files
        under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving directory ({}) should be a directory".format(save_directory)
        # save model config
        model_config_file = os.path.join(save_directory, self.model_config_file)
        model_config = self.init_config
        # If init_config contains a Layer, use the layer's init_config to save
        for key, value in model_config.items():
            if key == "init_args":
                args = []
                for arg in value:
                    args.append(
                        arg.init_config
                        if isinstance(arg, PretrainedModel) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, PretrainedModel):
                model_config[key] = value.init_config
        with io.open(model_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_config, ensure_ascii=False))
        # save model
        file_name = os.path.join(save_directory,
                                 list(self.resource_files_names.values())[0])
        paddle.save(self.state_dict(), file_name)
