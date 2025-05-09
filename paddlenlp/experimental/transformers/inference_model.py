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
import time
import paddle
import numpy as np
from paddlenlp.transformers import AutoConfig, AutoInferenceModelForCausalLM
from paddlenlp.utils.log import logger
from multiprocessing.shared_memory import SharedMemory


class InferenceModel:
    def __init__(self, predictor_args, model_args, nranks=1, rank=0, load_model_from_ipc=False, hot_start=False):
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
        self.shared_buffer_to = False
        self.local_test = True
        self.first_load = True
        self.hot_start = hot_start

        # (TODO:gaoziyuan)当前启动服务后直接加载参数，后续进行热启动
        if load_model_from_ipc and not hot_start:
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
        self.model = AutoInferenceModelForCausalLM.from_config(
            config=self.config,
            predictor_args=self.predictor_args,
            model_args=self.model_args,
            dtype=self.predictor_args.dtype,
            tensor_parallel_degree=self.nranks,
            tensor_parallel_rank=self.rank,
            load_model_from_ipc=self.load_model_from_ipc,
        )
        # print("gaoziyuan test load from config:", self.model.state_dict())
        return self.model

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
    
    def _log_memory_usage(self, context: str = "") -> None:
        """Log current GPU memory usage."""
        max_alloc = paddle.device.cuda.max_memory_allocated() / (1024 ** 3)
        max_reserved = paddle.device.cuda.max_memory_reserved() / (1024 ** 3)
        curr_alloc = paddle.device.cuda.memory_allocated() / (1024 ** 3)
        curr_reserved = paddle.device.cuda.memory_reserved() / (1024 ** 3)
        
        logger.info(f"GPU memory usage {context}:")
        logger.warning(
            f"max_allocated: {max_alloc:.2f}GB\n"
            f"max_reserved: {max_reserved:.2f}GB\n"
            f"current_allocated: {curr_alloc:.2f}GB\n"
            f"current_reserved: {curr_reserved:.2f}GB"
        )

    def generate(self, **kwargs):
        self.model.generate(**kwargs)
    
    def _update_shared_status(self, pid: int, status: int) -> None:
        """Update shared memory status flag."""
        array = np.zeros([self.nranks], dtype=np.int32)
        shm = SharedMemory(create=False, size=array.nbytes, name=f"model_weights_status.{pid}")
        value = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        value[self.rank] = status

    def update_parameters(self, pid: int = 0) -> None:
        """Update model parameters from IPC state dictionary."""
        if self.verify_parameters_updated(False) and not self.first_load:
            logger.info("Parameters already updated.")
            self._update_shared_status(pid, 2)
            return

        paddle.device.cuda.empty_cache()
        self._log_memory_usage("start update parameters")
        
        if self.local_test:
            current_device_id = int(os.getenv("FLAGS_selected_gpus"))
            model_path = f"/shared_ipc_meta/model_state.tp0{current_device_id}.pdparams"
            logger.info(f"Loading model from: {model_path}")
            
            set_start = time.time()
            self.model.set_state_dict(paddle.load(model_path))
            logger.info(f"set_state_dict completed in {time.time() - set_start:.2f} seconds")
            
            self.verify_parameters_updated()
            self._log_memory_usage("update parameters end")
            
            if not self.first_load:
                logger.info("send update signal")
                self._update_shared_status(pid, 2)
            self.first_load = False
            return

        start_time = time.time()
        logger.info("Starting parameter update process...")
        
        current_device_id = int(os.getenv("FLAGS_selected_gpus"))
        ipc_state_dict_path = f"/shared_ipc_meta/ipc_metas_{current_device_id}"
        logger.info(f"Loading IPC state dict from: {ipc_state_dict_path}")
        
        convert_start = time.time()
        state_dict = self.load_tensor_from_ipc_meta(paddle.load(ipc_state_dict_path))
        logger.info(f"IPC meta converted to tensors in {time.time() - convert_start:.2f} seconds")

        if not self.shared_buffer_to:
            logger.info("Updating parameters via set_state_dict...")
            set_start = time.time()
            self.model.set_state_dict(state_dict)
            logger.info(f"set_state_dict completed in {time.time() - set_start:.2f} seconds")
        else:
            logger.info("Updating parameters via shared_buffer_to...")
            share_start = time.time()
            infer_model_state_dict = self.model.state_dict()
            
            for name, param in state_dict.items():
                if name in infer_model_state_dict:
                    logger.info(f"Updating model parameter: {name}")
                    update_param = infer_model_state_dict[name]
                    
                    if update_param.dtype != param.dtype:
                        raise TypeError(f"Type mismatch for {name}: {param.dtype} vs {update_param.dtype}")
                    if update_param.shape != param.shape:
                        raise ValueError(f"Shape mismatch for {name}: {param.shape} vs {update_param.shape}")
                    
                    param._share_buffer_to(update_param)
            
            logger.info(f"Parameter sharing completed in {time.time() - share_start:.2f} seconds")

        if not self.first_load:
            logger.info("send update signal")
            self._update_shared_status(pid, 2)
        
        self.first_load = False
        self.verify_parameters_updated()
        paddle.device.cuda.empty_cache()
        self._log_memory_usage("update parameters end")
    
    def clear_parameters(self, pid: int = 0) -> None:
        """Clear all model parameters."""
        if self.verify_parameters_cleared(False):
            logger.info("Parameters already cleared!")
            self._update_shared_status(pid, -2)
            return

        start_time = time.time()
        paddle.device.cuda.empty_cache()
        self._log_memory_usage("start clear parameters")
        
        for name, param in self.model.state_dict().items():
            logger.info(f"Clearing model parameter: {name}")
            param._clear_data()

        clear_time = time.time() - start_time
        logger.info(f"Parameter clearing completed in {clear_time:.2f} seconds")

        self.verify_parameters_cleared()
        logger.info("Model parameters cleared successfully")
        
        self._update_shared_status(pid, -2)
        paddle.device.cuda.empty_cache()
        self._log_memory_usage("clear parameters end")
        logger.info("send clear signal!")

    
    def verify_parameters_cleared(self, erro_log:bool = True) -> bool:
        """
        Verify that all model parameters have been cleared.
        
        Returns:
            bool: True if all parameters are cleared, False otherwise
        """
        logger.info("Verifying parameters are cleared...")
        all_cleared = True
        for name, param in self.model.state_dict().items():
            if param._is_initialized():
                if erro_log:
                    logger.error(f"Parameter {name} was not properly cleared!")
                all_cleared = False
        
        if all_cleared:
            logger.info("All parameters verified as cleared successfully")
        else:
            if erro_log:
                logger.error("Some parameters were not properly cleared!")
        
        return all_cleared

    def verify_parameters_updated(self, erro_log:bool = True) -> bool:
        """
        Verify that model parameters match the source state dictionary.
        
        Args:
            source_state_dict: Dictionary containing the expected parameters
            
        Returns:
            bool: True if all parameters match, False otherwise
        """
        logger.info("Verifying parameters are cleared...")
        all_update = True
        for name, param in self.model.state_dict().items():
            if not param._is_initialized():
                if erro_log:
                    logger.error(f"Parameter {name} was not properly cleared!")
                all_update = False
        
        if all_update:
            logger.info("All parameters verified as updated successfully")
        else:
            if erro_log:
                logger.error("Some parameters were not properly updated!")
        
        return all_update
