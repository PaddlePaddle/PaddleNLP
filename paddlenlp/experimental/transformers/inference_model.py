"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""


import os
from typing import Any, Dict

import paddle

from paddlenlp.transformers import AutoConfig, AutoInferenceModelForCausalLM
from paddlenlp.utils.log import logger


class InferenceModel:
    def __init__(self, predictor_args, model_args, nranks=1, rank=0, load_model_from_ipc=False, cold_start=False):
        """
        Initialize the Causal Language Model Loader.

        Args:
            predictor_args: Predictor arguments object
            model_args: Model arguments object
            nranks: Number of parallel ranks (default: None)
            rank: Current rank in parallel setup (default: None)
            load_model_from_ipc: Whether to load model from IPC (default: False)
        """
        self.predictor_args = predictor_args
        self.model_args = model_args
        self.nranks = nranks
        self.rank = rank
        self.load_model_from_ipc = load_model_from_ipc
        self.model = self._build_model()

        # (TODO:gaoziyuan)当前启动服务后直接加载参数，后续进行热启动
        if load_model_from_ipc and not cold_start:
            self.update_parameters()

    def _setup_environment(self):
        """Setup paddle device and default dtype."""
        paddle.set_device(self.predictor_args.device)
        paddle.set_default_dtype(self.predictor_args.dtype)

    def _load_config(self):
        """Load model configuration."""
        return AutoConfig.from_pretrained(self.predictor_args.model_name_or_path)

    def _build_model(self):
        """
        Load the causal language model with the configured parameters.

        Returns:
            The loaded model
        """
        self._setup_environment()
        self.config = self._load_config()

        self.model = AutoInferenceModelForCausalLM.from_pretrained(
            self.predictor_args.model_name_or_path,
            config=self.config,
            predictor_args=self.predictor_args,
            model_args=self.model_args,
            dtype=self.predictor_args.dtype,
            tensor_parallel_degree=self.nranks,
            tensor_parallel_rank=self.rank,
            load_model_from_ipc=self.load_model_from_ipc,
        )
        return self.model

    def clear_parameters(self) -> None:
        """Clear all model parameters."""
        for name, param in self.model.state_dict().items():
            logger.info(f"Clearing model parameter: {name}")
            param._clear_data()

        logger.info("Model parameters cleared successfully")

    def get_model(self) -> paddle.nn.Layer:
        """Get the underlying model instance."""
        return self.model

    def get_model_static_info(self) -> None:
        """get static info."""
        for k, v in self.model.state_dict().items():
            logger.info(f"model key name is :{k}, shape : {v.shape}, dtype : {v.dtype}")

    @staticmethod
    def load_tensor_from_ipc_meta(ipc_state_dict: Dict[str, Any]) -> Dict[str, paddle.Tensor]:
        """
        Convert ipc_meta to tensor while keeping keys unchanged.

        Args:
            state_dict: Dictionary containing ipc_meta objects

        Returns:
            Dictionary with ipc_meta objects converted to tensors
        """
        result = {}
        for k, v in ipc_state_dict.items():
            v[0] = v[0].encode("latin-1")
            tensor = paddle.base.core.LoDTensor._new_shared_cuda(tuple(v))
            result[k] = paddle.to_tensor(tensor)

        return result

    def update_parameters(
        self,
    ) -> None:
        """
        Update model parameters from IPC state dictionary.

        Args:
            ipc_state_dict: Dictionary containing new parameters in IPC format
        """
        model_path = "/shared_ipc_meta"
        current_device_id = int(os.getenv("FLAGS_selected_gpus"))
        ipc_state_dict_path = os.path.join(model_path, f"ipc_metas_{current_device_id}")
        ipc_state_dict = paddle.load(ipc_state_dict_path)
        state_dict = self.load_tensor_from_ipc_meta(ipc_state_dict)

        infer_model_state_dict = self.model.state_dict()

        for name, param in state_dict.items():
            if name in infer_model_state_dict:
                logger.info(f"Updating model parameter: {name}")
                update_param = infer_model_state_dict[name]
                assert (
                    update_param.dtype == param.dtype
                ), f"Type mismatch for {name}: {param.dtype} vs {update_param.dtype}"
                assert (
                    update_param.shape == param.shape
                ), f"Shape mismatch for {name}: {param.shape} vs {update_param.shape}"
                param._share_buffer_to(update_param)

        logger.info("Model parameters updated successfully")
