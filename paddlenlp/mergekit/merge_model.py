# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import json
import os
from multiprocessing import Process

import numpy as np
import paddle
from safetensors.numpy import save_file

from paddlenlp.utils.env import (
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
)
from paddlenlp.utils.log import logger
from paddlenlp.utils.safetensors import fast_safe_open

from .merge_method import MergeMethod
from .merge_utils import divide_positions
from .sparsify_method import SparsifyMethod

SPARSIFY_MERGE_MAPPING = {
    "linear": (None, "linear"),
    "ties": ("trim", "ties"),
    "slerp": (None, "slerp"),
    "della_linear": ("magprune", "linear"),
    "della": ("magprune", "ties"),
    "dare_linear": ("dare", "linear"),
    "dare_ties": ("dare", "ties"),
}


class MergeModel:
    def __init__(self, merge_config):
        self.reset_merge_model(merge_config=merge_config)

    def reset_merge_model(self, merge_config=None, merge_param_dict=None):
        if merge_config is not None:
            self.merge_config = merge_config
        elif merge_param_dict is not None:
            for k, v in merge_param_dict.items():
                setattr(self.merge_config, k, v)
            self.merge_config.config_check()

        # map sparsify & merge type
        if self.merge_config.merge_method is not None:
            self.merge_config.sparsify_type, self.merge_config.merge_type = SPARSIFY_MERGE_MAPPING[
                self.merge_config.merge_method
            ]
        if self.merge_config.merge_type is None:
            raise ValueError("Either merge_type or merge_method must be specified.")
        # init merge method
        sparsify_method = SparsifyMethod(self.merge_config)
        self.merge_method = MergeMethod(merge_config, sparsify_method)

    def merge_model(self):
        check_safetensor_with_index = []
        for model_path in self.merge_config.model_name_or_path_list:
            check_safetensor_with_index.append(self.check_model_path(model_path))
        if self.merge_config.base_model_name_or_path is not None:
            check_safetensor_with_index.append(self.check_model_path(self.merge_config.base_model_name_or_path))
        if all(check_safetensor_with_index):
            self.merge_safetensor_model()
        else:
            raise NotImplementedError("Not support non safetensors models.")

    def merge_safetensor_model(self):
        # load index
        index_list = []
        for model_path in self.merge_config.model_name_or_path_list:
            with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                index_list.append(json.load(f))
        if self.merge_config.base_model_name_or_path is not None:
            with open(
                os.path.join(self.merge_config.base_model_name_or_path, self.safe_index_name()), "r", encoding="utf-8"
            ) as f:
                index_list.append(json.load(f))
        # check index
        if not all(index_list[0]["metadata"]["total_size"] == index["metadata"]["total_size"] for index in index_list):
            raise ValueError("Weights total_size mismatch. Please make sure you load the correct weight file")
        if not all(index_list[0]["weight_map"].keys() == index["weight_map"].keys() for index in index_list):
            raise ValueError("Weights weight_map mismatch. Please make sure you load the correct weight file")

        # init new index
        index = {}
        index["metadata"] = index_list[0]["metadata"]
        index["weight_map"] = {}

        # Multi-process update
        key_list = list(index_list[0]["weight_map"].keys())
        positions = divide_positions(len(key_list), self.merge_config.n_process)
        threads = []
        if self.merge_config.tensor_type == "np":
            target = self.shard_merge_np
        else:
            target = self.shard_merge_pd
        for i in range(len(positions) - 1):
            shard_file = f"{self.merge_config.merge_preifx}-{i+1:05d}-of-{self.merge_config.n_process:05d}.safetensors"
            t = Process(
                target=target,
                args=(
                    key_list[positions[i] : positions[i + 1]],  # key_list
                    index_list,  # index_list
                    shard_file,  # shard_file name
                ),
            )
            threads.append(t)
            for k in key_list[positions[i] : positions[i + 1]]:
                index["weight_map"][k] = shard_file
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # save safe index file
        save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
        if save_index_file and not os.path.exists(self.merge_config.output_path):
            os.makedirs(self.merge_config.output_path)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)
        self.merge_config.save_pretrained(self.merge_config.output_path)

    def shard_merge_np(
        self,
        key_list,
        index_list,
        shard_file,
    ):
        merge_state_dict = {}
        for k in key_list:
            tensor_list = []

            for i, model_path in enumerate(self.merge_config.model_name_or_path_list):
                with fast_safe_open(os.path.join(model_path, index_list[i]["weight_map"][k]), framework="np") as w:
                    tensor = w.get_tensor(k)
                    dtype = tensor.dtype
                    # dtype==bfloat16: numpy(uint16) -> paddle(bfloat16) -> paddle(float32) -> numpy(float32)
                    if tensor.dtype == np.uint16:
                        tensor = paddle.to_tensor(tensor, dtype="bfloat16").astype("float32").numpy()
                    tensor_list.append(tensor)
            if self.merge_config.base_model_name_or_path is not None:
                with fast_safe_open(
                    os.path.join(self.merge_config.base_model_name_or_path, index_list[-1]["weight_map"][k]),
                    framework="np",
                ) as w:
                    base_tensor = w.get_tensor(k)
                    if base_tensor.dtype == np.uint16:
                        base_tensor = paddle.to_tensor(base_tensor, dtype="bfloat16").astype("float32").numpy()
                tensor_list = [tensor - base_tensor for tensor in tensor_list]
            merge_state_dict[k] = self.merge_method.merge(tensor_list)
            if self.merge_config.base_model_name_or_path is not None:
                merge_state_dict[k] += base_tensor
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if dtype == np.uint16:
                merge_state_dict[k] = paddle.to_tensor(merge_state_dict[k], dtype="float32").astype("bfloat16").numpy()
        save_file(
            merge_state_dict,
            os.path.join(self.merge_config.output_path, shard_file),
            metadata={"format": "np"},
        )

    def shard_merge_pd(
        self,
        key_list,
        index_list,
        shard_file,
    ):
        raise NotImplementedError("Not support paddle tensors.")

    def check_model_path(self, model_path):

        if os.path.exists(os.path.join(model_path, self.safe_index_name())):
            with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                index = json.load(f)
                safe_file_list = list(set(index["weight_map"][k] for k in index["weight_map"]))
                for i in range(len(safe_file_list)):
                    if os.path.exists(os.path.join(model_path, safe_file_list[i])):
                        continue
                    else:
                        ValueError(f"Not found {os.path.join(model_path, safe_file_list[i])}.")
            is_safetensor = True
        elif os.path.exists(os.path.join(model_path, self.weight_name)):
            is_safetensor = False
            if self.merge_config.n_process > 1:
                logger.info("Set `n_process`=1 when using non safetensors models.")
                self.merge_config.n_process = 1
            raise NotImplementedError("Not support non safetensors models.")
        else:
            raise ValueError(f"Please check path {model_path} is correct.")
        return is_safetensor

    def weight_name(self):
        if self.merge_config.merge_preifx == "model":
            return PADDLE_WEIGHTS_NAME
        else:
            return PADDLE_MASTER_WEIGHTS_NAME

    def safe_index_name(self):
        if self.merge_config.merge_preifx == "model":
            return SAFE_WEIGHTS_INDEX_NAME
        else:
            return SAFE_MASTER_WEIGHTS_INDEX_NAME
