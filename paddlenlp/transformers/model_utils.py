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
from __future__ import annotations

import copy
import inspect
import io
import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import paddle
import paddle.nn as nn
import six
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    repo_type_and_id_from_hf_id,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError
from paddle import Tensor
from paddle.nn import Embedding, Layer

# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddle.utils.download import is_url

from paddlenlp import __version__
from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    download_check,
    get_path_from_url_with_filelock,
)
from paddlenlp.utils.env import (
    CONFIG_NAME,
    HF_CACHE_HOME,
    LEGACY_CONFIG_NAME,
    MODEL_HOME,
)
from paddlenlp.utils.log import logger

from .configuration_utils import PretrainedConfig
from .generation_utils import GenerationMixin
from .utils import (
    InitTrackerMeta,
    adapt_stale_fwd_patch,
    fn_args_to_dict,
    resolve_cache_dir,
)

__all__ = [
    "PretrainedModel",
    "register_base_model",
]


def unwrap_model(model, *args, **kwargs):
    raw_model = model._layers if isinstance(model, paddle.DataParallel) else model
    return raw_model


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    """get dtype of parameter which should be sub-class of nn.Layer

    Args:
        parameter (nn.Layer): the instance of layer

    Returns:
        paddle.dtype: the dtype of tensor
    """

    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    # TODO(wj-Mcat): get dtype of model when it's in DataParallel Mode.
    return last_dtype


def _find_weight_file_path(
    cache_dir: str, model_class: Type[PretrainedModel], resource_uri: Optional[str] = None
) -> str:
    """find the target weight file under the cache dir, because there are some conflicts about weight file names.

    Args:
        cache_dir (str): the cache dir of pretrained weighted files
        model_class (Type[PretrainedModel]): the class of pretrained model
        resource_uri (Optional[str], optional): the weight file resource file uri to help find the target file name. Defaults to None.
    """
    # 1. if the weight file is the name of resource_uri, eg: 'bert-base-uncased.pdparams'
    if resource_uri is not None:
        resouce_uri_file_name = os.path.split(resource_uri)[-1]
        weight_file_path = os.path.join(cache_dir, resouce_uri_file_name)
        if os.path.isfile(weight_file_path):
            return weight_file_path

    # 2. find the target weight file name under the `resource_files_names` attribute of `PretrainedModel`
    resource_weight_file_name = model_class.resource_files_names.get("model_state", None)
    weight_file_path = os.path.join(cache_dir, resource_weight_file_name)
    if os.path.isfile(weight_file_path):
        return weight_file_path

    # 3. find the target weight file if there is only one weight file
    weight_file_names = [file for file in os.listdir(cache_dir) if file.endswith(".pdparams")]
    if len(weight_file_names) == 1:
        logger.warning(
            f"there is no <{resource_weight_file_name}> which is the expected weight file name "
            f"under dir<{cache_dir}>, but the file<{weight_file_names[0]}> is found, and it will "
            f"be used to init model weights. We suggest that you rename it to <{resource_weight_file_name}>"
        )
        return os.path.join(cache_dir, weight_file_names[0])

    raise ValueError(
        'can"t find the target weight files<%s> or <%s> under the cache_dir<%s>',
        resouce_uri_file_name,
        resource_weight_file_name,
        cache_dir,
    )


def register_base_model(cls):
    """
    A decorator for `PretrainedModel` class. It first retrieves the parent class
    of the class being decorated, then sets the `base_model_class` attribute
    of that parent class to be the class being decorated. In summary, the decorator registers
    the decorated class as the base model class in all derived classes under the same architecture.

    Args:
        cls (PretrainedModel): The class (inherited from PretrainedModel) to be decorated .

    Returns:
        PretrainedModel: The input class `cls` after decorating.

    Example:
        .. code-block::

            from paddlenlp.transformers import BertModel, register_base_model

            BertModel = register_base_model(BertModel)
            assert BertModel.base_model_class == BertModel
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
    The base class for all pretrained models. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading
    and saving also rely on the following class attributes which should be overridden
    by derived classes accordingly:

    - **model_config_file** (str): Represents the file name of model configuration
      for configuration saving and loading in local file system. The value is
      `model_config.json`.
    - **resource_files_names** (dict): Name of local file where the model configuration
      can be saved and loaded locally. Currently, resources only include the model state,
      thus the dict only includes `'model_state'` as key with corresponding
      value `'model_state.pdparams'` for model weights saving and loading.
    - **pretrained_init_configuration** (dict): Provides the model configurations
      of built-in pretrained models (contrasts to models in local file system).
      It has pretrained model names as keys (such as `bert-base-uncased`), and
      the values are dict preserving corresponding configuration for model initialization.
    - **pretrained_resource_files_map** (dict): Provides resource URLs of built-in
      pretrained models (contrasts to models in local file system).
      It has the same key as resource_files_names (that is "model_state"),
      and the corresponding value is a dict with specific model name to model weights URL mapping
      (such as "bert-base-uncased" ->
      "https://bj.bcebos.com/paddlenlp/models/transformers/bert-base-uncased.pdparams").
    - **base_model_prefix** (str): Represents the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model. Note: A base model class is pretrained model class
      decorated by `register_base_model`, such as `BertModel`; A derived model
      class is a pretrained model class adding layers on top of the base model,
      and it has a base model as attribute, such as `BertForSequenceClassification`.

    Methods common to models for text generation are defined in `GenerationMixin`
    and also inherited here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedModel`,
    by which subclasses can track arguments for initialization automatically.
    """

    # Deprecated(wj-Mcat): after 2.6.* version
    # save the old-school `LEGACY_CONFIG_NAME`, and will be changed to `CONFIG_NAME` after 2.6.* version
    model_config_file = LEGACY_CONFIG_NAME

    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fields as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {}
    base_model_prefix = ""
    main_input_name = "input_ids"
    config_class = None

    # a list of `re` patterns of `state_dict` keys that should be removed from the list of missing
    # keys we find (keys inside the model but not in the checkpoint) and avoid unnecessary warnings.
    _keys_to_ignore_on_load_missing = None
    # a list of `re` patterns of `state_dict` keys that should be removed from the list of
    # unexpected keys we find (keys inside the checkpoint but not the model) and avoid unnecessary
    # warnings.
    _keys_to_ignore_on_load_unexpected = None
    # a list of `state_dict` keys to ignore when saving the model (useful for keys that aren't
    # trained, but which are either deterministic or tied variables)
    _keys_to_ignore_on_save = None

    def __init__(self, *args, **kwargs):
        super(PretrainedModel, self).__init__()

        if not self.constructed_from_pretrained_config():
            return

        # extract config from args
        config = None
        for arg in args:
            if isinstance(arg, PretrainedConfig):
                config = arg
                break
        if config is not None:
            self.config: PretrainedConfig = config
            self.model_config_file = CONFIG_NAME
            return

        # extract config from kwargs
        if "config" not in kwargs:
            raise ValueError(
                "PretrainedConfig instance not found in the arguments, you can set it as args or kwargs with config field"
            )

        config = kwargs["config"]
        if not isinstance(config, PretrainedConfig):
            raise TypeError("config parameter should be the instance of PretraiendConfig")

        self.config: PretrainedConfig = kwargs["config"]
        self.model_config_file = CONFIG_NAME
        self.warnings_issued = {}

    def _post_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the pretrained model instance.
        """
        if not self.constructed_from_pretrained_config():
            init_dict = fn_args_to_dict(original_init, *((self,) + args), **kwargs)
            self.config = init_dict

    @property
    def base_model(self):
        """
        PretrainedModel: The body of the same model architecture. It is the base
            model itself for base model or the base model attribute for derived
            model.
        """
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        """
        list: Contains all supported built-in pretrained model names of the
            current PretrainedModel class.
        """
        # Todo: return all model name
        return list(self.pretrained_init_configuration.keys())

    def get_input_embeddings(self) -> nn.Embedding:
        """get input embedding of model

        Returns:
            nn.Embedding: embedding of model
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()

        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def set_input_embeddings(self, value: Embedding):
        """set new input embedding for model

        Args:
            value (Embedding): the new embedding of model

        Raises:
            NotImplementedError: Model has not implement `set_input_embeddings` method
        """
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.set_input_embeddings(value)
        raise NotImplementedError(
            f"model of {type(base_model)} has not implemented the `get_input_embeddings`"
            " or `set_input_embeddings` method"
        )

    def get_output_embeddings(self) -> Optional[Embedding]:
        """To be overwrited for models with output embeddings

        Returns:
            Optional[Embedding]: the otuput embedding of model
        """
        return None

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """resize position embedding, this method should be overrited overwrited by downstream models

        Args:
            new_num_position_embeddings (int): the new position size

        Raises:
            NotImplementedError: when called and not be implemented
        """
        raise NotImplementedError(
            f"`resize_position_embeddings` is not implemented for {self.__class__}`. To implement it, you should "
            f"overwrite this method in the class {self.__class__} in `{self.__class__.__module__}.py`"
        )

    @classmethod
    def constructed_from_pretrained_config(cls, init_func=None) -> bool:
        """check if the model is constructed from `PretrainedConfig`

        Returns:
            bool: if the model is constructed from `PretrainedConfig`
        """
        return cls.config_class is not None and issubclass(cls.config_class, PretrainedConfig)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, from_hf_hub=False, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): whether to load from Huggingface Hub
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/')
        """
        if cls.constructed_from_pretrained_config():
            return cls.from_pretrained_v2(pretrained_model_name_or_path, from_hf_hub=from_hf_hub, *args, **kwargs)

        resource_files = {}
        init_configuration = {}
        load_state_as_np = kwargs.pop("load_state_as_np", False)
        track_download = True

        # From HF Hub
        if from_hf_hub:
            resource_files = cls.resource_files_names
            resource_files["model_config_file"] = cls.model_config_file

        # From built-in pretrained models
        elif pretrained_model_name_or_path in cls.pretrained_init_configuration:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                if pretrained_model_name_or_path not in map_list:
                    resource_files[file_id] = None
                else:
                    resource_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(cls.pretrained_init_configuration[pretrained_model_name_or_path])

        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            track_download = False
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                resource_files[file_id] = full_file_name
            resource_files["model_config_file"] = os.path.join(pretrained_model_name_or_path, cls.model_config_file)
        else:
            # Assuming from community-contributed pretrained models
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = "/".join([COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, file_name])
                resource_files[file_id] = full_file_name
            resource_files["model_config_file"] = "/".join(
                [COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.model_config_file]
            )

        default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
                continue
            # If from_hf_hub, let HF Hub takes care of the cache and the download process
            if from_hf_hub:
                resolved_resource_files[file_id] = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=file_path,
                    cache_dir=HF_CACHE_HOME,
                    library_name="PaddleNLP",
                    library_version=__version__,
                )
            else:
                path = os.path.join(default_root, file_path.split("/")[-1])
                if os.path.exists(path):
                    logger.info("Already cached %s" % path)
                    resolved_resource_files[file_id] = path
                else:
                    logger.info("Downloading %s and saved to %s" % (file_path, default_root))
                    try:
                        resolved_resource_files[file_id] = get_path_from_url_with_filelock(file_path, default_root)
                    except RuntimeError as err:
                        logger.error(err)
                        raise RuntimeError(
                            f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                            f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                            "- a correct model-identifier of built-in pretrained models,\n"
                            "- or a correct model-identifier of community-contributed pretrained models,\n"
                            "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                        )

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file", None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration

        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", cls.base_model_class.__name__)
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
                    assert arg.pop("init_class") == cls.base_model_class.__name__, (
                        "pretrained base model should be {}"
                    ).format(cls.base_model_class.__name__)
                    base_arg_index = i
                    base_arg = arg
                    break
            for arg_name, arg in init_kwargs.items():
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop("init_class") == cls.base_model_class.__name__, (
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
            base_parameters_dict = inspect.signature(cls.base_model_class.__init__).parameters
            for k, v in kwargs.items():
                if k in base_parameters_dict:
                    base_kwargs[k] = v
            base_model = cls.base_model_class(*base_args, **base_kwargs)
            if base_arg_index is not None:
                derived_args[base_arg_index] = base_model
            else:
                derived_args = (base_model,)  # assume at the first position
            derived_args = derived_args if not args else args
            derived_parameters_dict = inspect.signature(cls.__init__).parameters
            for k, v in kwargs.items():
                if k in derived_parameters_dict:
                    derived_kwargs[k] = v
            model = cls(*derived_args, **derived_kwargs)

        # save the model config file into cache dir
        model_config_file_path = os.path.join(default_root, cls.model_config_file)
        # check if there is model config file in cache directory
        if (
            pretrained_model_name_or_path in cls.pretrained_init_configuration
            and init_kwargs is not None
            and not os.path.exists(model_config_file_path)
        ):
            model.save_model_config(default_root)

        # Maybe need more ways to load resources.
        weight_path = resolved_resource_files["model_state"]
        if weight_path is None:
            logger.warning(
                "No model weight found for %s, return with random initialization !!!" % pretrained_model_name_or_path
            )
            return model

        assert weight_path.endswith(".pdparams"), "suffix of weight must be .pdparams"

        # NOTE: Allow to load partial model for model parallel.
        # TODO(guosheng): To make model loading for the model parallel automatic,
        # maybe we should make rank 0 worker load weights of the full model on
        # CPU, then split weights into multiple parts and pickle separately.
        # The other workers wait util pickle finish and then load the corresponding
        # partial weights. Also we can directly use separate weight files for
        # simplicity.
        state_dict = paddle.load(weight_path, return_numpy=load_state_as_np)

        # Make sure we are able to load base models as well as derived models
        # (with heads)
        start_prefix = ""
        model_to_load = model
        state_to_load = state_dict
        unexpected_keys = []
        missing_keys = []
        if not hasattr(model, cls.base_model_prefix) and any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            # base model
            state_to_load = {}
            start_prefix = cls.base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(cls.base_model_prefix):
                    state_to_load[k[len(start_prefix) :]] = v
                else:
                    unexpected_keys.append(k)
        if hasattr(model, cls.base_model_prefix) and not any(
            s.startswith(cls.base_model_prefix) for s in state_dict.keys()
        ):
            # derived model (base model with heads)
            model_to_load = getattr(model, cls.base_model_prefix)
            for k in model.state_dict().keys():
                if not k.startswith(cls.base_model_prefix):
                    missing_keys.append(k)
        if len(missing_keys) > 0:
            logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            logger.info(
                "Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys)
            )
        # Allow the float16 model to load float32 weights, which decreases memory
        # usage in model loading stage and is useful to big models.
        dtype_prefix_len = len("paddle.")  # paddle.float16

        for k, v in model_to_load.state_dict().items():
            if not isinstance(v, np.ndarray):
                dtype = str(v.dtype)[dtype_prefix_len:]
            # TODO(guosheng): add warnings for unmatched dtypes
            if k in state_to_load:
                if paddle.in_dynamic_mode():
                    if isinstance(state_to_load[k], np.ndarray):
                        state_to_load[k] = state_to_load[k].astype(dtype)
                    else:
                        state_to_load[k] = paddle.cast(state_to_load[k], dtype)
                else:
                    # there are some latent error when case dtype in static-mode, so let's:
                    # 1. convert fluid.*.Tensor -> numpy.ndarray
                    # 2. cast the dtype with numpy tools
                    # 3. paddle works well with ndarray state-dict
                    state_to_load[k] = np.array(state_to_load[k])
                    state_to_load[k] = state_to_load[k].astype(dtype)

        # For model parallel if FasterGeneration
        # To avoid recursive import temporarily.
        import paddlenlp.ops.faster_transformer.transformer.decoding as ft_decoding

        state_to_load = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_to_load)
        if paddle.in_dynamic_mode():
            model_to_load.set_state_dict(state_to_load)
            if track_download:
                download_check(pretrained_model_name_or_path, "from_pretrained")
            return model
        if track_download:
            download_check(pretrained_model_name_or_path, "from_pretrained")
        return model, state_to_load

    # NOTE: backward support for old models. Models with PretrainedConfig should be able to use .config
    def get_model_config(self):
        """Get model configuration.

        Returns:
            config: The config of the model.
        """

        # If init_config contains a Layer, use the layer's init_config to save
        def get_config(model):
            if model.config is not None and isinstance(model.config, PretrainedConfig):
                return model.config
            model_config = model.init_config
            for key, value in model_config.items():
                if key == "init_args":
                    args = []
                    for arg in value:
                        args.append(get_config(arg) if isinstance(arg, PretrainedModel) else arg)
                    model_config[key] = tuple(args)
                elif isinstance(value, PretrainedModel):
                    model_config[key] = value.init_config
            return model_config

        model_config = get_config(self)
        return model_config

    def save_model_config(self, save_dir: str):
        """
        Saves model configuration to a file named "config.json" under `save_dir`.

        Args:
            save_dir (str): Directory to save model_config file into.
        """
        # Save model config
        model_config = self.get_model_config()
        if isinstance(model_config, PretrainedConfig):
            model_config.save_pretrained(save_dir)
        else:
            model_config_file = os.path.join(save_dir, self.model_config_file)
            with io.open(model_config_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(model_config, ensure_ascii=False, indent=2))

    def save_pretrained(self, save_dir: str):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """
        if self.constructed_from_pretrained_config():
            return self.save_pretrained_v2(save_dir)

        assert not os.path.isfile(save_dir), "Saving directory ({}) should be a directory, not a file".format(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        # Save model config
        self.save_model_config(save_dir)
        # Save model
        if paddle.in_dynamic_mode():
            file_name = os.path.join(save_dir, list(self.resource_files_names.values())[0])
            paddle.save(self.state_dict(), file_name)
        else:
            logger.warning("Save pretrained model only supported dygraph mode for now!")

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        commit_message: Optional[bool] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this model to a new HuggingFace Hub repository.
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
                    f.write(f"---\nlibrary_name: paddlenlp\n---\n# {repo_id}")

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

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the model according to new_num_tokens.

        Args:
            new_num_tokens (Optional[int]):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or None, just
                returns a pointer to the input tokens embedding module of the model without doing anything.

        Returns:
            paddle.nn.Embedding: The input tokens Embeddings Module of the model.
        """
        old_embeddings: nn.Embedding = self.get_input_embeddings()
        if not new_num_tokens or new_num_tokens == old_embeddings.weight.shape[0]:
            return old_embeddings

        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # 2. Update vocab_size
        self.base_model.config["vocab_size"] = new_num_tokens
        self.vocab_size = new_num_tokens

        # update init_config
        self._update_init_config(self.init_config, "vocab_size", new_num_tokens)

        # TODO(westfish@126.com): add tie_weight.
        # TODO(westfish) Add tie_weight to tie the weights between the input embeddings and the output embeddings if needed.

        return new_embeddings

    def _update_init_config(self, init_config: dict, key: str, value: Any):
        """update init_config by <key, value> pair

        Args:
            init_config (dict): the init_config instance
            key (str): the key field
            value (Any): the new value of instance
        """
        if key in init_config:
            init_config[key] = value
            return

        for arg in init_config.get("init_args", []):
            if not isinstance(arg, PretrainedModel):
                continue
            self._update_init_config(arg.init_config, key, value)

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (nn.Embedding):
                Old embeddings to be resized.
            new_num_tokens (Optional[int]):
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end.

        Returns:
            paddle.nn.Embedding: The resized Embedding Module or the old Embedding Module if new_num_tokens is None.
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}. You"
                " should either use a different resize function or make sure that old_embeddings are an instance of"
                f" {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            new_embeddings.weight[:n, :] = old_embeddings.weight[:n, :]

        return new_embeddings

    def __setattr__(self, name, value):
        value = adapt_stale_fwd_patch(self, name, value)
        return super(PretrainedModel, self).__setattr__(name, value)

    @classmethod
    def _resolve_model_file_path(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
    ) -> str:
        """resolve model target file path from `` and `cache_dir`

        0. when it is file path:
            return the weight file

        1. when it is model-name:
            1.1 check default `MODEL_HOME` + `model-mame` + model_state.pdparams
            1.2 get the url from `pretrained_resource_files_map`, and set it to `pretrained_model_name_or_path`

        2. when it is url:
            fetch the resouce into the `cache_dir` (cache_dir or `MODEL_HOME` + `model-mame`)

        3. when it is local dir:
            check whether the file<local_dir + weight_file> exist

        Args:
            cls (Type[PretrainedModel]): the inherited PretrainedModel class
            pretrained_model_name_or_path (str): the model-name/url/local_dir/local_dir
            cache_dir (Optional[str], optional): cache_dir is used when name_or_path is model-name/url. Defaults to None.

        Returns:
            str: the model weight file path
        """
        # 0. when it is local file
        if os.path.isfile(pretrained_model_name_or_path):
            return pretrained_model_name_or_path

        # 1. when it's from HF
        if from_hf_hub:
            return hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=cls.resource_files_names["model_state"],
                cache_dir=HF_CACHE_HOME,
                library_name="PaddleNLP",
                library_version=__version__,
            )

        # 2. when it is model-name
        if pretrained_model_name_or_path in cls.pretrained_init_configuration:
            # check the cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

            # check the state_dict file
            weight_file_path = os.path.join(cache_dir, cls.resource_files_names["model_state"])
            if os.path.exists(weight_file_path):
                return weight_file_path

            # fetch the weight url from the `pretrained_resource_files_map`
            pretrained_model_name_or_path = cls.pretrained_resource_files_map["model_state"][
                pretrained_model_name_or_path
            ]

        # 3. when it is url
        if is_url(pretrained_model_name_or_path):
            weight_file_path = get_path_from_url_with_filelock(pretrained_model_name_or_path, cache_dir)
            # # check the downloaded weight file and registered weight file name

            # make sure that
            new_weight_file_path = os.path.join(
                os.path.split(weight_file_path)[0], cls.resource_files_names["model_state"]
            )

            # if the weight file name of url is: `bert-base-uncased.pdparams`, the downloaded file is also of it.
            # and we should convert it to the new weitht file: `model_state.pdparams`
            if weight_file_path != new_weight_file_path:

                # move the `model-name.pdparams` to `model_state.pdparams`
                # get more details from: https://github.com/PaddlePaddle/PaddleNLP/pull/3843
                shutil.move(weight_file_path, new_weight_file_path)

                weight_file_path = new_weight_file_path

            # find the weight file with the above two branch: `bert-base-uncased.pdparams`, `model_state.pdparams`
            weight_file_path = _find_weight_file_path(
                cache_dir=cache_dir, model_class=cls, resource_uri=pretrained_model_name_or_path
            )

            return weight_file_path

        # 4. when it is local dir
        if os.path.isdir(pretrained_model_name_or_path):
            # in-order to compatible with old style:
            # file name in pretrained_resouce_file_maps is https://path/to/bert-base-uncased.pdparams, but the registered model-state file name in `resouce_file_maps` is `model_state.pdparams`

            weight_file_path = _find_weight_file_path(cache_dir=pretrained_model_name_or_path, model_class=cls)

            if os.path.exists(weight_file_path):
                return weight_file_path
        else:
            # assume that the community-based models, name format: community/model-name
            pretrained_model_name_or_path = os.path.join(
                COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.resource_files_names["model_state"]
            )
            assert is_url(pretrained_model_name_or_path)
            return cls._resolve_model_file_path(pretrained_model_name_or_path, cache_dir=cache_dir)

        raise FileNotFoundError(
            "can't resolve the model_state file according to the <%s>", pretrained_model_name_or_path
        )

    @classmethod
    def _load_pretrained_model(
        cls,
        model: PretrainedModel,
        state_dict: Dict[str, Tensor],
        loaded_keys: List[str],
        ignore_mismatched_sizes=False,
        dtype=None,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * check the

        Args:
            model (PretrainedModel): the pretrained model instance
            state_dict (Dict[str, Tensor]): the model state dict data
            loaded_keys (List[str]):
            ignore_mismatched_sizes (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """

        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        if len(prefix) > 0:
            has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
            expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
        else:
            has_prefix_module = False
            expects_prefix_module = False

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix_from_model = not has_prefix_module and expects_prefix_module
        add_prefix_to_model = has_prefix_module and not expects_prefix_module

        if remove_prefix_from_model:
            expected_keys = [".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{checkpoint_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(checkpoint_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        # Whole checkpoint
        mismatched_keys = _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        )

        start_prefix = prefix + "."

        # `add_prefix_to_model` and `remove_prefix_from_model` are for different situation,
        # you can check the following matrix, which means:
        # the value of cell: (add_prefix_to_model, remove_prefix_from_model)
        # the load/Init-Base is the state-dict which don't contain `prefix`.
        # the load/Init-DownStream is the state-dict which contain the `prefix`
        #
        # |                 | load-Base | load-DownStream |
        # |-----------------|-----------|-----------------|
        # | Init-Base       | F,F       | T,F             |
        # | Init-DonwStream | F,T       | F,F             |
        #
        # the above value matrix will help you understand the following code.
        if add_prefix_to_model:
            for key in list(state_dict.keys()):
                if key.startswith(start_prefix):
                    state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

        if remove_prefix_from_model:
            for key in list(state_dict.keys()):
                state_dict[start_prefix + key] = state_dict.pop(key)

        # convert the dtype of state dict
        if dtype is not None:
            if isinstance(dtype, paddle.dtype):
                dtype = str(dtype)[7:]

            if dtype not in ["float32", "float16"]:
                raise ValueError(f"the value of `dtype` should be one of [`float32`, `float16`], but received {dtype}")
            for key in state_dict.keys():
                state_dict[key] = paddle.cast(state_dict[key], dtype=dtype)
        else:
            dtype_prefix_len = len("paddle.")
            for k, v in model_to_load.state_dict().items():
                if not isinstance(v, np.ndarray):
                    dtype = str(v.dtype)[dtype_prefix_len:]
                if k in state_dict:
                    if paddle.in_dynamic_mode():
                        if isinstance(state_dict[k], np.ndarray):
                            state_dict[k] = state_dict[k].astype(dtype)
                        else:
                            state_dict[k] = paddle.cast(state_dict[k], dtype)
                    else:
                        # there are some latent error when case dtype in static-mode, so let's:
                        # 1. convert fluid.*.Tensor -> numpy.ndarray
                        # 2. cast the dtype with numpy tools
                        # 3. paddle works well with ndarray state-dict
                        state_dict[k] = np.array(state_dict[k])
                        state_dict[k] = state_dict[k].astype(dtype)

        # For model parallel if FasterGeneration
        # To avoid recursive import temporarily.
        import paddlenlp.ops.faster_transformer.transformer.decoding as ft_decoding

        state_to_load = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_dict)
        if paddle.in_dynamic_mode():
            model_to_load.set_state_dict(state_to_load)

        return model_to_load, missing_keys, unexpected_keys, mismatched_keys

    @classmethod
    def from_pretrained_v2(cls, pretrained_model_name_or_path, from_hf_hub: bool = False, *args, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/'
        """
        load_state_as_np = kwargs.pop("load_state_as_np", False)
        config = kwargs.pop("config", None)
        force_download = kwargs.pop("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", None)
        dtype = kwargs.pop("dtype", None)
        cache_dir = kwargs.pop("cache_dir", None)

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)

        model_kwargs = kwargs
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                **kwargs,
            )
        config.save_pretrained(cache_dir)

        # 2. init the model
        init_args = config["init_args"] or ()
        model = cls(config, *init_args, **model_kwargs)

        # 3. resolve model_weight file
        model_weight_file = cls._resolve_model_file_path(
            pretrained_model_name_or_path, cache_dir=cache_dir, from_hf_hub=from_hf_hub
        )

        # 4. loading the state dict
        model_state_dict = paddle.load(model_weight_file, return_numpy=load_state_as_np)

        loaded_state_dict_keys = list(model_state_dict.keys())
        # TODO(wj-Mcat): load shard checkpoint weight file, refer to: https://github.com/huggingface/transformers/pull/16343
        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=model_state_dict,
            loaded_keys=loaded_state_dict_keys,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            dtype=dtype,
        )

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
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
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
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
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )
        if paddle.in_dynamic_mode():
            return model

        return model, model_state_dict

    def save_pretrained_v2(self, save_dir: str):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """
        assert not os.path.isfile(save_dir), "Saving directory ({}) should be a directory, not a file".format(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 1. retrieve the model related config

        # save the string version of dtype to the config, e.g. convert paddle.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        model_to_save = unwrap_model(self)
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        model_to_save.config.save_pretrained(save_dir)

        # Save model
        if paddle.in_dynamic_mode():
            file_name = os.path.join(save_dir, self.resource_files_names["model_state"])
            paddle.save(self.state_dict(), file_name)
        else:
            logger.warning("Save pretrained model only supported dygraph mode for now!")
