# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import inspect
import io
import json
import os
from shutil import copyfile

import numpy as np
import paddle
from paddle.framework import core

from paddlenlp.transformers import PretrainedModel
from paddlenlp.utils.download import resolve_file_path

# TODO(fangzeyang) Temporary fix and replace by paddle framework downloader later
from paddlenlp.utils.log import logger

__all__ = ["FasterPretrainedModel", "ActScalesLoader", "WeightScalesLoader", "PerTensorWeightScalesLoader"]


def load_vocabulary(filepath):
    token_to_idx = {}
    with io.open(filepath, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.rstrip("\n")
            token_to_idx[token] = int(index)
    return token_to_idx


class FasterPretrainedModel(PretrainedModel):
    def to_static(self, output_path):
        self.eval()

        # Convert to static graph with specific input description
        model = paddle.jit.to_static(
            self, input_spec=[paddle.static.InputSpec(shape=[None, None], dtype=core.VarDesc.VarType.STRINGS)]
        )
        paddle.jit.save(model, output_path)
        logger.info("Already save the static model to the path %s" % output_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.

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
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        subfolder = kwargs.pop("subfolder", "")

        # From built-in pretrained models
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(cls.pretrained_init_configuration[pretrained_model_name_or_path])
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            for file_id, file_name in cls.resource_files_names.items():
                full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                if os.path.isfile(full_file_name):
                    resource_files[file_id] = full_file_name
            resource_files["model_config_file"] = os.path.join(pretrained_model_name_or_path, cls.model_config_file)
        else:
            for file_id, file_name in cls.resource_files_names.items():
                resource_files[file_id] = file_name

        # default_root = os.path.join(MODEL_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
                continue
            resolved_resource_files[file_id] = resolve_file_path(
                pretrained_model_name_or_path,
                [file_path],
                subfolder,
                cache_dir=cache_dir,
                from_aistudio=from_aistudio,
                from_hf_hub=from_hf_hub,
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
            vocab_file = resolved_resource_files.pop("vocab_file", None)
            if vocab_file and base_kwargs.get("vocab_file", None) is None:
                base_kwargs["vocab_file"] = vocab_file
            assert base_kwargs.get("vocab_file", None) is not None, "The vocab "
            f"file is None. Please reload the class  {cls.__name__} with pretrained_name."

            model = cls(*base_args, **base_kwargs)
        else:
            # Update with newly provided args and kwargs for derived model
            base_parameters_dict = inspect.signature(cls.base_model_class.__init__).parameters
            for k, v in kwargs.items():
                if k in base_parameters_dict:
                    base_kwargs[k] = v

            vocab_file = resolved_resource_files.pop("vocab_file", None)
            if vocab_file and base_kwargs.get("vocab_file", None) is None:
                base_kwargs["vocab_file"] = vocab_file
            assert base_kwargs.get("vocab_file", None) is not None, "The vocab "
            f"file is None. Please reload the class  {cls.__name__} with pretrained_name."

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

        # Maybe need more ways to load resources.
        weight_path = resolved_resource_files["model_state"]
        assert weight_path.endswith(".pdparams"), "suffix of weight must be .pdparams"

        state_dict = paddle.load(weight_path)
        logger.info("Loaded parameters from %s" % weight_path)

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
        if paddle.in_dynamic_mode():
            model_to_load.set_state_dict(state_to_load)
            return model
        return model, state_to_load

    @staticmethod
    def load_vocabulary(filepath):
        token_to_idx = {}
        with io.open(filepath, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                token = line.rstrip("\n")
                token_to_idx[token] = int(index)
        return token_to_idx

    def save_pretrained(self, save_dir):
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
        # Save model config
        self.save_model_config(save_dir)
        # Save model
        if paddle.in_dynamic_mode():
            file_name = os.path.join(save_dir, list(self.resource_files_names.values())[0])
            paddle.save(self.state_dict(), file_name)
        else:
            logger.warning("Save pretrained model only supported dygraph mode for now!")
        # Save resources file
        self.save_resources(save_dir)

    def save_resources(self, save_directory):
        """
        Save tokenizer related resources to `resource_files_names` indicating
        files under `save_directory` by copying directly. Override it if necessary.

        Args:
            save_directory (str): Directory to save files into.
        """
        for name, file_name in self.resource_files_names.items():
            src_path = self.init_config["init_args"][0].get(name, None)
            dst_path = os.path.join(save_directory, file_name)
            if src_path and os.path.abspath(src_path) != os.path.abspath(dst_path):
                copyfile(src_path, dst_path)


class ActScalesLoader:
    def __init__(
        self,
        scale_json_file_path="act_scales.json",
        key_map_dict=None,
        num_of_layers=None,
    ):
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            self.scale[scale_type] = np.full([num_of_layers], fill_value=-1.0, dtype="float32")
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i] = 1 / self.scale_dict[key_template.replace("#", str(i))]


class WeightScalesLoader:
    def __init__(
        self,
        scale_json_file_path="weight_scales.json",
        key_map_dict=None,
        num_of_layers=None,
        concat_qkv=False,
        concat_ffn1=False,
    ):
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            no_skip_layer_list = []
            n = 1
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    no_skip_layer_list.append(key_template.replace("#", str(i)))
            if len(no_skip_layer_list) > 0:
                n = len(self.scale_dict[no_skip_layer_list[0]])
            self.scale[scale_type] = np.full([num_of_layers, n], fill_value=-1.0, dtype="float32")
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i, :] = self.scale_dict[key_template.replace("#", str(i))]

        # concat qkv and ffn1
        if concat_qkv:
            self.scale["qkv_weight_scale"] = []

        if concat_ffn1:
            self.scale["ffn1_weight_scale"] = []

        for i in range(num_of_layers):
            if concat_qkv:
                self.scale["qkv_weight_scale"].append(
                    np.concatenate(
                        [
                            self.scale["q_weight_scale"][i, :],
                            self.scale["k_weight_scale"][i, :],
                            self.scale["v_weight_scale"][i, :],
                        ]
                    )
                )

            if concat_ffn1:
                self.scale["ffn1_weight_scale"].append(
                    np.concatenate([self.scale["ffn1_1_weight_scale"][i, :], self.scale["ffn1_2_weight_scale"][i, :]])
                )


class PerTensorWeightScalesLoader:
    """
    Load Per Tensor Weight Scale from json file
    """

    def __init__(
        self,
        scale_json_file_path="weight_scales.json",
        key_map_dict=None,
        num_of_layers=None,
    ):
        """load weight scales from json file."""
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            no_skip_layer_list = []
            scale_shape = [1]
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    no_skip_layer_list.append(key_template.replace("#", str(i)))
            if len(no_skip_layer_list) > 0:
                scale_shape = np.array(self.scale_dict[no_skip_layer_list[0]]).shape

            self.scale[scale_type] = np.full(((num_of_layers,) + tuple(scale_shape)), fill_value=-1.0)
            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    self.scale[scale_type][i] = self.scale_dict[key_template.replace("#", str(i))]

        if "qkv_weight_scale" not in self.scale.keys():
            self.scale["qkv_weight_scale"] = np.full((num_of_layers), fill_value=-1.0).astype(np.float32)
            for i in range(num_of_layers):
                qkv_weight_scale = max(
                    self.scale["q_weight_scale"][i],
                    self.scale["k_weight_scale"][i],
                    self.scale["v_weight_scale"][i],
                )
                self.scale["qkv_weight_scale"][i] = qkv_weight_scale


class CacheScaleLoader:
    def __init__(
        self,
        scale_json_file_path="cache_scales.json",
        key_map_dict=None,
        num_of_layers=None,
        num_heads=None,
        num_key_value_heads=None,
    ):
        with open(scale_json_file_path) as json_file:
            self.scale_dict = json.load(json_file)
        self.key_map = key_map_dict
        self.scale = {}
        for scale_type, key_template in self.key_map.items():
            if "cache_k" in scale_type:
                scale_type_out = "cache_k_out_scale"
            else:
                scale_type_out = "cache_v_out_scale"
            self.scale[scale_type] = np.full([num_of_layers, num_key_value_heads], fill_value=-1.0)
            self.scale[scale_type_out] = np.full([num_of_layers, num_key_value_heads], fill_value=-1.0)

            for i in range(num_of_layers):
                if key_template.replace("#", str(i)) in self.scale_dict.keys():
                    if num_heads != num_key_value_heads:
                        self.scale[scale_type][i, :] = [
                            127.0 / self.scale_dict[key_template.replace("#", str(i))][j]
                            for j in range(0, num_heads, num_heads // num_key_value_heads)
                        ]
                    else:
                        self.scale[scale_type][i, :] = [
                            127.0 / self.scale_dict[key_template.replace("#", str(i))][j]
                            for j in range(0, num_key_value_heads)
                        ]
                    self.scale[scale_type_out][i, :] = [
                        1.0 / self.scale[scale_type][i, j] for j in range(0, num_key_value_heads)
                    ]
