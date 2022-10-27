# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from abc import ABC, abstractmethod
import json
import os
import re
from typing import Callable, Dict, Optional, List, Tuple, Union, TypeVar
from dataclasses import dataclass

import paddle
from paddle import Tensor
from paddle.nn import Layer
import numpy as np
from numpy import ndarray, transpose, allclose
from paddlenlp.utils.log import logger
from paddlenlp.utils.import_utils import is_package_available, is_torch_available, is_transformers_available

# the type hinting for pytorch model & layer & tensor
Module = TypeVar("Module")
PytorchTensor = TypeVar("PytorchTensor")


def tensor_summary(tensor: Union[str, Tensor, PytorchTensor, tuple, list,
                                 ndarray]):
    """get summary of values which can be some of different values

    Args:
        tensor (ndarray): the source data of tensor which can be: string, Paddle Tensor, Pytorch Tensor, tuple/list tensor, ndarray

    Returns:
        str: the summary info
    """
    if tensor is None:
        return "None"

    if isinstance(tensor, str):
        return tensor

    # Modeling Output from paddlenlp/transformers
    if isinstance(tensor, dict):
        tensor = list(tensor.values())

    if isinstance(tensor, (tuple, list)):
        infos = []
        for item in tensor:
            infos.append(tensor_summary(item))
        return '\n'.join(infos)

    # it must be Paddle Tensor or Pytorch Tensor
    if not isinstance(tensor, ndarray):
        tensor = tensor.detach().cpu().numpy()

    tensor = np.reshape(tensor, [-1])
    top_3_tensor = str(tensor[:3])
    return top_3_tensor


def _compare_model_weights(first_state_dict: Dict[str, ndarray],
                           second_state_dict: Dict[str, ndarray]) -> List[str]:
    """compare the values of two state_dict

    Args:
        first_state_dict (Dict[str, ndarray]): first state_dict
        second_state_dict (Dict[str, ndarray]): second state_dict

    Returns:
        mismatched keys (List[str]): the mismatched keys of state_dict because of some reason
    """
    mismatched_keys = []
    for key in first_state_dict.keys():
        is_close = np.allclose(first_state_dict[key],
                               second_state_dict[key],
                               atol=1e-4)
        if not is_close:
            mismatched_keys.append(key)
    return mismatched_keys


@dataclass
class StateDictNameMapping:
    """NameMapping of StateDict between two models"""
    source_name: str
    target_name: str

    action: Optional[
        str] = None  # the value can be: transpose, merge_last_two_dim
    index: Optional[int] = None

    def should_transpose(self) -> bool:
        return self.action == 'transpose'

    def should_merge_last_two_dim(self) -> bool:
        """check that wether merge last two dim"""
        return self.action == "merge_last_two_dim"

    def run(self, tensor: ndarray) -> ndarray:
        """run some custom operation on ndarray, eg: transpose, merge_last_two_dim

        Args:
            tensor (ndarray): the source of the tensor data

        Returns:
            ndarray: the final tensor
        """
        if self.action == "transpose":
            return transpose(tensor, [1, 0])
        if self.action == 'merge_last_two_dim':
            shape = tensor.shape
            assert len(shape) == 3
            return np.reshape(tensor, [shape[0], -1])


class TensorInfoSaver:

    def __init__(self) -> None:
        self.series = {}

    def add(self, state_dict_key: str, key: str,
            values: Union[float, ndarray, Tensor, PytorchTensor]):
        """add 

        Args:
            state_dict_key (str): _description_
            key (str): _description_
            values (Union[float, ndarray, Tensor]): _description_

        Raises:
            NotImplementedError: _description_
        """
        if state_dict_key not in self.series[state_dict_key]:
            self.series[state_dict_key]["state_dict_key"] = state_dict_key

        self.series[state_dict_key][key] = tensor_summary(values)

    def summary(self, output_path: Optional[str] = None):
        """output the summary info into different terminal

        Args:
            output_path (Optional[str], optional): the dir/file of sumamry file. Defaults to None.
        """
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, "tensor_summary.xlsx")

        self.summary_to_terminal()
        self.summary_to_excel(output_path)

    def summary_to_excel(self, file: str):
        if not is_package_available("pandas"):
            return False
        if not is_package_available("openpyxl"):
            logger.warning(
                "detect that pandas is installed, but openpyxl is not installed so can't save info into excel file. "
                "you can run command: `pip install openpyxl` to get the great feature"
            )
            return False

        import pandas as pd
        with pd.ExcelWriter(file, 'a', engine='openpyxl',
                            if_sheet_exists="new") as writer:
            pd.DataFrame(list(self.series.values())).to_excel(writer,
                                                              index=False)

    def summary_to_terminal(self):
        """print table info into terminal with tabulate"""
        from tabulate import tabulate
        headers = {key: key for key in self.series[0].keys()}
        print(
            tabulate(list(self.series.values()),
                     tablefmt='grid',
                     headers=headers))

    def clear(self):
        """clear the series data"""
        self.series.clear()


class LogitHooker:
    """hooks for pytorch model and paddle model, used to generate the logits of elment layers"""

    def __init__(self,
                 mappings: List[StateDictNameMapping],
                 tensor_info_saver: Optional[TensorInfoSaver] = None):
        """registe the logit hooks to compare the inputs * outputs model

        Args:
            mappings (List[StateDictNameMapping]): the mappings between paddle & pytorch model
            tensor_info_saver (Optional[TensorInfoSaver], optional): the saver for model logit. Defaults to None.
        """
        self.mappings = mappings
        self.tensor_info_saver = tensor_info_saver or TensorInfoSaver()

    def _paddle_hooks(self, layer: Layer, inputs: Tuple[Tensor],
                      outputs: Union[Tensor, Tuple[Tensor]]):
        """internal paddle hooks to save the logit of paddle layer

        Args:
            layer (Layer): the layer of paddle element
            inputs (Tuple[Tensor]): the inputs of paddle layer
            outputs (Union[Tensor, Tuple[Tensor]]): the outputs of paddle layer
        """
        state_dict_name = layer.__state_dict_name__

        self.tensor_info_saver.add(state_dict_name, "paddle-input", inputs)

        self.tensor_info_saver.add(state_dict_name, "paddle-outputs", outputs)

    def _pytorch_hooks(self, layer: Layer, inputs: Tuple[PytorchTensor],
                       outputs: Union[Dict[str, PytorchTensor],
                                      Tuple[PytorchTensor]]):
        """internal pytorch hooks to save the logit of pytorch module

        Args:
            layer (torch.nn.Module): the module of pytorch model
            inputs (Tuple[PytorchTensor]): the inputs of pytorch layer
            outputs (Union[Dict[str, PytorchTensor], Tuple[PytorchTensor]]): the outputs of pytorch layer
        """
        state_dict_name = layer.__state_dict_name__

        self.tensor_info_saver.add(
            state_dict_name,
            "pytorch-input",
            inputs,
        )

        self.tensor_info_saver.add(state_dict_name, "pytorch-outputs", outputs)

    def register_paddle_model_hooks(self, model: Layer):
        """regist post forward hook to save the inputs & outputs of paddle model

        Args:
            model (Layer): paddle model
        """

        # 1. register paddle model hook to save the logits of target layer
        def register_hook_by_name(model: Layer, mapping: StateDictNameMapping,
                                  hook: Callable[..., None]):
            """register hook by name of state_dict, eg: encoder.layers.0.linear1.bias

            Args:
                model (Layer): the source model
                mapping (StateDictNameMapping): the name mapping object
                hook (Callable[..., None]): the hook for paddle model
            """
            name = mapping.target_name
            attributes = name.split(".")
            last_layer: Layer = model
            for attribute in attributes:
                if getattr(model, attribute, None) is not None:
                    model = getattr(model, attribute)
                    if isinstance(model, Layer):
                        last_layer = model
            if hasattr(last_layer, 'register_forward_post_hook') and getattr(
                    last_layer, "__state_dict_name__", None) is None:
                last_layer.register_forward_post_hook(hook)
                # set state_dict key into layer as the private attribute
                last_layer.__state_dict_name__ = name

        for mapping in self.mappings:
            register_hook_by_name(model, mapping, self._paddle_hooks)

    def register_pytorch_model_hooks(self, model: Module):
        """regist hook for pytorch model to save the inputs & outputs of pytorch model

        Args:
            model (_type_): pytorch model 
        """
        from torch import nn

        # 1. register paddle model hook to save the logits of target layer
        def register_hook_by_name(model: Module, mapping: StateDictNameMapping,
                                  hook: Callable[..., None]):
            name = mapping.source_name
            attributes, index = name.split("."), 0
            last_layer: Module = model
            while index < len(attributes):
                attribute = attributes[index]
                if getattr(model, attribute, None) is not None:
                    if isinstance(model, nn.ModuleList) and attribute.isdigit():
                        model = model[int(attribute)]
                        last_layer = model
                    else:
                        model = getattr(model, attribute)
                        if isinstance(model, nn.Module):
                            last_layer = model
                index += 1
            if hasattr(last_layer, 'register_forward_hook') and getattr(
                    last_layer, "__state_dict_name__", None) is None:
                last_layer.register_forward_hook(hook)
                # set state_dict key into layer as the private attribute
                last_layer.__state_dict_name__ = mapping.target_name

        for mapping in self.mappings:
            register_hook_by_name(model, mapping, self._pytorch_hooks)

    def summary(self):
        """print the summary info to terminal/excel to analysis"""
        self.tensor_info_saver.summary()


class Converter(ABC):
    """Model Weight Converter for developer to convert pytorch/tensorflow/jax pretrained model weight to paddle.

        * you can convert model weight in online/offline mode.
        * you can convert weight and config file.
        * you can convert weight/config file in some customization ways.
    """

    _ignore_state_dict_keys = []
    num_layer_regex = "\.\d+\."

    num_layer_key: str = "num_hidden_layers"

    @classmethod
    def get_num_layer(
            cls, state_dict: Union[List[str], Dict[str,
                                                   ndarray]]) -> Optional[int]:
        """get num layer size from state_dict

        Args:
            state_dict (Union[List[str], Dict[str, ndarray]]): the state_dict data

        Returns:
            Optional[int]: the number of transformer layer
        """
        if isinstance(state_dict, dict):
            state_dict = list(state_dict.keys())

        numbers = set()
        for key in state_dict:
            spans = re.findall(cls.num_layer_regex, key)
            if spans and len(spans) == 1:
                numbers.add(int(spans[0][1:-1]))
        if len(numbers) == 0:
            return None

        return max(numbers) + 1

    @classmethod
    def resolve_num_layer(cls,
                          config_or_num_layers: Union[dict, int] = None) -> int:
        """resolve the number of transformer layer based on the key of model config, eg: `num_hidden_layers` in BertModel

        Args:
            config_or_num_layers (Union[dict, int], optional): the instance of config or num_layers. Defaults to None.

        Raises:
            ValueError: when `config_or_num_layers` is not dict/int, it will raise the error

        Returns:
            int: the number of transformer layer
        """
        if isinstance(config_or_num_layers, dict):
            num_layer = config_or_num_layers[cls.num_layer_key]
        elif isinstance(config_or_num_layers, int):
            num_layer = config_or_num_layers
        else:
            raise ValueError(
                f"the type of config_or_num_layers<{config_or_num_layers}> should be one of <dict, int>"
            )

        return num_layer

    def get_paddle_pytorch_model_classes(self) -> Tuple[object, object]:
        """return the [PaddleModelClass, PytorchModelClass] to
            1. generate paddle model automatically
            2. compare the logits from pytorch model and paddle model automatically

        Returns:
            Tuple[object, object]: [PaddleModelClass, PytorchModelClass]
        """
        raise NotImplementedError

    def get_inputs(self):
        """the numpy inputs for paddle & pytorch model"""
        input_ids = paddle.arange(600, 700)
        input_ids = paddle.unsqueeze(input_ids, axis=0).detach().cpu().numpy()
        return [input_ids]

    def resolve_paddle_output_logits(self, paddle_outputs: Tuple[Tensor]):
        """resolve the logit from paddle model which can be `last_hidden_state`"""
        output = None
        if isinstance(paddle_outputs, (tuple, list)):
            output = paddle_outputs[0]
        elif paddle.is_tensor(paddle_outputs):
            output = paddle_outputs

        if output is None:
            raise NotImplementedError("can't resolve paddle model outputs")

        return output.detach().cpu().reshape([-1]).numpy()

    def resolve_pytorch_output_logits(self, pytorch_outputs: Module):
        """resolve the logit from pytorch model which can be `last_hidden_state`"""
        output = pytorch_outputs[0]
        if output is None:
            raise NotImplementedError("can't resolve paddle model outputs")

        return output.detach().cpu().reshape([-1]).numpy()

    @staticmethod
    def get_model_state_dict(model: Union[Layer, Module]) -> Dict[str, ndarray]:
        """get the state_dict of pytorch/paddle model

        Args:
            model (Union[Layer, Module]): can be paddle/pytorch model

        Returns:
            Dict[str, ndarray]: the final state_dict data
        """
        from torch import nn
        assert isinstance(model, (Layer, nn.Module))
        return {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        }

    def compare_model_state_dicts(self, paddle_model: Layer,
                                  pytorch_model: Module,
                                  name_mappings: List[StateDictNameMapping]):
        """compare the pytorch and paddle mdoel state with name mappings

        Args:
            paddle_model (Layer): paddle model instance
            pytorch_model (Module): pytorch model instance
            name_mappings (List[StateDictNameMapping]): the name mappings
        """
        paddle_state_dict = {
            key: value.detach().cpu().numpy()
            for key, value in paddle_model.state_dict().items()
        }
        pytorch_state_dict = {
            key: value.detach().cpu().numpy()
            for key, value in pytorch_model.state_dict().items()
        }

        model_state_saver = TensorInfoSaver()
        for name_mapping in name_mappings:
            model_state_saver.add(name_mapping.target_name, "pytorch_key",
                                  name_mapping.source_name)
            model_state_saver.add(
                name_mapping.target_name, "paddle",
                paddle_state_dict.pop(name_mapping.target_name))
            model_state_saver.add(
                name_mapping.target_name, "pytorch",
                pytorch_state_dict.pop(name_mapping.source_name))

        model_state_saver.summary()

    def compare_logits(self, paddle_pretrained_dir: str,
                       pytorch_pretrained_dir: str) -> bool:
        """compare the logit of pytorch & paddle model

        Args:
            paddle_pretrained_dir (str): the pretrained_dir of paddle model which should contains pytorch_model.bin & config.json file
            pytorch_pretrained_dir (str): the pretrained_dir of pytorch model which should contains pytorch_model.bin & config.json file
        Returns:
            bool: if the logits is absolutly same
        """
        PaddleModel, PytorchModel = self.get_paddle_pytorch_model_classes()
        paddle_model = PaddleModel.from_pretrained(paddle_pretrained_dir)

        # 0. init the name_mapping & tensor_info_saver & logit_hooker
        name_mappings = self.get_name_mapping(paddle_model.config)
        tensor_info_saver = TensorInfoSaver()

        logit_hooker = LogitHooker(name_mappings, tensor_info_saver)
        inputs = self.get_inputs()

        # 1. get the logits of paddle model
        logit_hooker.register_paddle_model_hooks(paddle_model)
        paddle_inputs = [paddle.to_tensor(input_item) for input_item in inputs]
        paddle_model.eval()

        paddle_outputs = paddle_model(*paddle_inputs)
        paddle_logits = self.resolve_paddle_output_logits(paddle_outputs)

        logger.info(
            "===============the summary of paddle Model logits: ==============="
        )
        logger.info(tensor_summary(paddle_logits))

        # 2. get the logits of pytorch model
        import torch
        pytorch_model = PytorchModel.from_pretrained(pytorch_pretrained_dir)
        logit_hooker.register_pytorch_model_hooks(pytorch_model)

        pytorch_model.eval()
        pytorch_inputs = [torch.tensor(input_item) for input_item in inputs]
        torch_outputs = pytorch_model(*pytorch_inputs)
        pytorch_logits = self.resolve_pytorch_output_logits(torch_outputs)

        logger.info(
            "===============the summary of pytorch Model logits: ==============="
        )
        logger.info(tensor_summary(pytorch_logits))

        # 3. compare the logits
        result = allclose(paddle_logits[:3], pytorch_logits[:3], atol=1e-4)

        if not result:
            print(
                "============================== compare model state dict =============================="
            )
            self.compare_model_state_dicts(paddle_model, pytorch_model,
                                           name_mappings)
            del paddle_model, pytorch_model

            # 4. init two paddle model to check that if there are any weights not initialized correctly.
            first_paddle_model = PaddleModel.from_pretrained(
                paddle_pretrained_dir)
            first_state_dict = self.get_model_state_dict(first_paddle_model)

            second_paddle_model = PaddleModel.from_pretrained(
                paddle_pretrained_dir)
            second_state_dict = self.get_model_state_dict(second_paddle_model)

            keys = _compare_model_weights(first_state_dict, second_state_dict)
            if keys:
                for key in keys:
                    logger.error(f"the key<{key}> is not set with weight")
            else:
                # 5. compare the logits of paddle & pytorch model
                print(
                    "============================== compare model inputs & outputs =============================="
                )
                logit_hooker.summary()

        return result

    def convert_config(self, pytorch_config: dict) -> dict:
        """convert torch config to paddle config

        Args:
            pytorch_config (dict): the object of pytorch config file

        Returns:
            dict: the final converted config object
        """
        return pytorch_config

    @abstractmethod
    def get_name_mapping(
        self,
        config_or_num_layers: Union[dict, int] = None
    ) -> List[StateDictNameMapping]:
        """construct name-mapping object base on the `torch_state_dict` object. name-mapping is the configuration to convert torch state_dict to paddle state_dict

        Args:
            torch_state_dict (Dict[str, ndarray]): state_dict of pytorch
            config_or_num_layers (Union[dict, int]): config or num_layers info to help generate name-mapping info

        Raises:
            NotImplementedError: you should override `get_name_mapping` method to generate NameMapping info to convert weight file automaticlly

        Returns:
            List[StateDictNameMapping]: the configuration of name-mapping
        """
        raise NotImplementedError(
            "every Converter should everride `get_name_mapping` method to get the configuration between pytorch model and paddle model"
        )

    @staticmethod
    def detect_latent_transpose(name_mappings: List[StateDictNameMapping]):
        """detect the latent name-mapping rules

        Args:
            name_mappings (List[StateDictNameMapping]): the mappings object
        """
        linear_keys = ['dense', 'linear', 'fc', 'proj']
        for name_mapping in name_mappings:
            if not name_mapping.target_name.endswith(".weight"):
                continue
            if any([
                    linear_key in name_mapping.target_name
                    for linear_key in linear_keys
            ]):
                if not name_mapping.action:
                    logger.warning(
                        f"detect that the state_dict<{name_mapping.target_name}> is not supporting `transpose` or `merge_last_two_dim` action"
                    )

    def convert_state_dict(self,
                           torch_state_dict: Dict[str, ndarray],
                           config: Optional[dict] = None) -> Dict[str, ndarray]:
        """if you want to convert state_dict with your own script, you should override this method, which is the simplest way to reuse your script.

        Args:
            torch_state_dict (Dict[str, ndarray]): state_dict of pytorch
            config (Optional[dict], optional): model config object used to help generate name-mappings. Defaults to None.

        Returns:
            Dict[str, ndarray]: the final converted state_dict object
        """
        if config is None:
            config = self.get_num_layer(torch_state_dict)

        mappings = self.get_name_mapping(config_or_num_layers=config)
        self.detect_latent_transpose(mappings)

        state_dict = {}
        for mapping in mappings:
            if mapping.source_name not in torch_state_dict:
                logger.warning(
                    f"the key<{mapping.source_name}> is not in torch_state_dict"
                )
                continue

            value = torch_state_dict.pop(mapping.source_name)

            # run custom operation on tensor, eg: transpose, merge_last_two_dim
            value = mapping.run(value)

            state_dict[mapping.target_name] = value

        # remove the ignore keys
        for ignore_key in self._ignore_state_dict_keys:
            if ignore_key in torch_state_dict:
                logger.warning(
                    f"remove the ignored key<{ignore_key}> from torch state_dict"
                )
                torch_state_dict.pop(ignore_key)

        if len(torch_state_dict) > 0:
            keys = ",".join(list(torch_state_dict.keys()))
            logger.warning(f"there are some weights not converted: <{keys}>")

        return state_dict

    def load_torch_weight_file(self, model_file: str) -> Dict[str, ndarray]:
        """load torch weight file with torch which should be removed later.

        Args:
            model_file (str): the path of pytorch model file

        Returns:
            Dict[str, ndarray]: the state dict object of loaded pytorch state dict
        """
        # TODO(wj-Mcat): use torch to load model file
        import torch
        state_dict = torch.load(model_file)
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].numpy()
        return state_dict

    def convert(self, input_dir: str, output_dir: str):
        """the entry of converting config and converting model file

        Args:
            input_dir (str): the input dir which contains `pytorch_model.bin` and `config.json` file
            output_dir (str): the output dir
        """
        os.makedirs(output_dir, exist_ok=True)

        # detect the file path of
        model_config_file, torch_model_file = None, None
        if os.path.isfile(input_dir):
            if input_dir.endswith('.json'):
                model_config_file = input_dir
            elif input_dir.endswith('.bin'):
                torch_model_file = input_dir

        else:
            path = os.path.join(input_dir, 'pytorch_model.bin')
            if os.path.isfile(path):
                torch_model_file = path
            else:
                logger.warning(
                    f"pytorch_model.bin file not found under <{input_dir}>")

            path = os.path.join(input_dir, 'config.json')
            if os.path.isfile(path):
                model_config_file = path
            else:
                logger.warning(
                    f"config.json file not found under <{input_dir}>")

        config = None
        if model_config_file is not None:
            with open(model_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config = self.convert_config(config)

            # save config file
            config_file = os.path.join(output_dir, "model_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f)

        if torch_model_file:
            state_dict = self.load_torch_weight_file(torch_model_file)
            paddle_state_dict = self.convert_state_dict(state_dict, config)

            paddle.save(paddle_state_dict,
                        os.path.join(output_dir, 'model_state.pdparams'))

        if is_torch_available() and is_transformers_available():
            result = self.compare_logits(paddle_pretrained_dir=output_dir,
                                         pytorch_pretrained_dir=input_dir)
            if result is True:
                logger.info(
                    "the logits between pytorch model and paddle model is absolutly same"
                )
            else:
                logger.error(
                    "the logits between pytorch model and paddle model is not same, please check it out more carefully."
                )
        else:
            logger.warning(
                "you don't install `torch` and `transformers` package, so we can't compare the logits between paddle & pytorch model"
            )
