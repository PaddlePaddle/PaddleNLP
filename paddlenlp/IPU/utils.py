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

import paddle
from contextlib import ExitStack

_CONFIG_REGISTRY = {}
_MODEL_REGISTRY = {}
_Pipeline_REGISTRY = {}

def register(transformers_cls=None):
    def wrapper(cls):
        orig_cls = transformers_cls
        if not cls:
            raise ValueError(f"{orig_cls} needs a target class")
        else:
            if 'Config' in getattr(cls, '__name__'):
                _CONFIG_REGISTRY[orig_cls] = cls
            elif 'Model' in getattr(cls, '__name__'):
                _MODEL_REGISTRY[orig_cls] = cls
            elif 'Pipeline' in getattr(cls, '__name__'):
                _Pipeline_REGISTRY[orig_cls] = cls
            else:
                raise ValueError(f"{cls} is not in the list")
        return cls

    return wrapper


class DeviceScope(object):
    def __init__(self, index, stage, name_scope=None):
        self.index = index
        self.stage = stage
        self.name_scope = name_scope

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(
            paddle.static.ipu_shard_guard(
                index=self.index, stage=self.stage))
        if self.name_scope is not None:
            self.stack.enter_context(paddle.static.name_scope(self.name_scope))
        return self

    def __exit__(self, *exp):
        self.stack.close()
        return False


def AutoPipeline(args, config, model):
    """
    Import corresponding pipeline configuration for different models
    """
    pipeline_cls = _Pipeline_REGISTRY.get(args.model_name)
    modelPipeline = pipeline_cls(args, config, model)
    modelPipeline.to_pipelined()


def AutoConfig(args):
    """
    Import corresponding model configuration for different models
    """
    config_cls = _CONFIG_REGISTRY.get(args.model_name)
    config = config_cls(args)
    ipu_config = {}
    if args.num_hidden_layers == 12:
        if args.task == "PRETRAINING":
            config.get_layer_ipu_index()   
            ipu_config['embeddings_scope'] = DeviceScope(0, 0, "Embedding")
            ipu_config['mlm_scope'] = DeviceScope(0, args.num_ipus, "MLM")
            ipu_config['nsp_scope'] = DeviceScope(0, args.num_ipus, "NSP")
            config.from_dict(**ipu_config)
            return config        
        elif args.task == "SQUAD":
            config.get_layer_ipu_index()
            ipu_config['embeddings_scope'] = DeviceScope(0, 0, "Embedding")
            ipu_config['squad_scope'] = DeviceScope(args.num_ipus - 1, args.num_ipus - 1, "squad")
            config.from_dict(**ipu_config)
            return config
    else:
        raise Exception("Only support num_hidden_layers = 12")


def AutoModel(args, config, custom_ops):
    """
    Choose corresponding model class by args.model_name
    """
    model_cls = _MODEL_REGISTRY.get(args.model_name)
    model = model_cls(config, custom_ops)
    return model
