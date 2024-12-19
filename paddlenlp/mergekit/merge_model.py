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
import gc
import json
import os
from multiprocessing import Process

import numpy as np
import paddle
from safetensors import safe_open
from safetensors.numpy import save_file

from paddlenlp.utils.env import (
    PADDLE_MASTER_WEIGHTS_NAME,
    PADDLE_WEIGHTS_NAME,
    SAFE_MASTER_WEIGHTS_INDEX_NAME,
    SAFE_MASTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
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
        self.numpy_dtype_map = {"float32": 4, "float16": 2, "uint16": 2}

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
        file_type_list = []
        for model_path in self.merge_config.model_path_list:
            file_type_list.append(self.check_model_path(model_path))
        if self.merge_config.base_model_path is not None:
            file_type_list.append(self.check_model_path(self.merge_config.base_model_path))
        if all(file_type == "safetensors" or file_type == "safetensors_without_index" for file_type in file_type_list):
            self.merge_safetensor_model(file_type_list)
        else:
            self.merge_mix_model(file_type_list)

    def merge_mix_model(self, file_type_list):
        state_dict_list = []
        for i, model_path in enumerate(self.merge_config.model_path_list):
            state_dict_list.append(self.get_model_state_dict(model_path, file_type_list[i]))
        if self.merge_config.base_model_path is not None:
            state_dict_list.append(self.get_model_state_dict(self.merge_config.base_model_path, file_type_list[-1]))
        if not all(state_dict_list[0].keys() == state_dict.keys() for state_dict in state_dict_list):
            raise ValueError("State dict keys mismatch. Please make sure you load the correct weight file")
        if self.merge_config.base_model_path is not None:
            base_state_dict = state_dict_list.pop()
            base_file_type = file_type_list.pop()
        merge_state_dict = {}
        total_size = 0
        weight_map = {}
        for key in state_dict_list[0].keys():
            is_bf16 = False
            tensor_list = []
            for state_dict, file_type in zip(state_dict_list, file_type_list):
                if file_type == "pdparams":
                    if str(state_dict[key].dtype) == "paddle.bfloat16":
                        is_bf16 = True
                        state_dict[key] = state_dict[key].astype("float32").numpy()
                    else:
                        state_dict[key] = state_dict[key].numpy()
                elif str(state_dict[key].dtype) == "uint16":
                    is_bf16 = True
                    state_dict[key] = paddle.to_tensor(state_dict[key], dtype="bfloat16").astype("float32").numpy()
                tensor_list.append(state_dict[key])
            if self.merge_config.base_model_path is not None:
                if base_file_type == "pdparams":
                    if str(base_state_dict[key].dtype) == "paddle.bfloat16":
                        base_state_dict[key] = base_state_dict[key].astype("float32").numpy()
                    else:
                        base_state_dict[key] = base_state_dict[key].numpy()
                elif str(base_state_dict[key].dtype) == "uint16":
                    base_state_dict[key] = (
                        paddle.to_tensor(base_state_dict[key], dtype="bfloat16").astype("float32").numpy()
                    )
                tensor_list = [tensor - base_state_dict[key] for tensor in tensor_list]
            merge_state_dict[key] = self.merge_method.merge(tensor_list)
            if self.merge_config.base_model_path is not None:
                merge_state_dict[key] += base_state_dict[key]
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if is_bf16:
                merge_state_dict[key] = (
                    paddle.to_tensor(merge_state_dict[key], dtype="float32").astype("bfloat16").numpy()
                )
            total_size += np.prod(merge_state_dict[key].shape) * self.numpy_dtype_map[str(merge_state_dict[key].dtype)]
            weight_map[key] = f"{self.merge_config.merge_preifx}-00001-of-00001.safetensors"
        # save safetensor file
        save_file(
            merge_state_dict,
            os.path.join(
                self.merge_config.output_path, f"{self.merge_config.merge_preifx}-00001-of-00001.safetensors"
            ),
            metadata={"format": "np"},
        )
        # save safe index file
        index = {"metadata": {"total_size": int(total_size)}, "weight_map": weight_map}
        save_index_file = os.path.join(self.merge_config.output_path, self.safe_index_name())
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)
        # save merge config file
        self.merge_config.save_pretrained(self.merge_config.output_path)
        del state_dict_list
        del merge_state_dict
        if self.merge_config.base_model_path is not None:
            del base_state_dict
        gc.collect()

    def get_model_state_dict(self, model_path, file_type):
        if file_type == "safetensors":
            state_dict = {}
            with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                index = json.load(f)
            for key in index["weight_map"].keys():
                with fast_safe_open(
                    os.path.join(model_path, index["weight_map"][key]),
                    framework="np",
                ) as f:
                    state_dict[key] = f.get_tensor(key)
        elif file_type == "safetensors_without_index":
            state_dict = {}
            with fast_safe_open(os.path.join(model_path, self.safe_weight_name()), framework="numpy") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        elif file_type == "pdparams":
            state_dict = paddle.load(os.path.join(model_path, self.weight_name()))
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        return state_dict

    def create_safetensor_index(self, model_path):
        weight_map = {}
        total_size = 0

        with safe_open(os.path.join(model_path, self.safe_weight_name()), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_size += np.prod(tensor.shape) * self.numpy_dtype_map[str(tensor.dtype)]
                weight_map[key] = self.safe_weight_name()
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        return index

    def merge_safetensor_model(self, file_type_list):
        # load index
        index_list = []
        model_path_list = self.merge_config.model_path_list
        if self.merge_config.base_model_path is not None:
            model_path_list += [self.merge_config.base_model_path]

        for model_path, file_type in zip(model_path_list, file_type_list):
            if file_type == "safetensors":
                with open(os.path.join(model_path, self.safe_index_name()), "r", encoding="utf-8") as f:
                    index_list.append(json.load(f))
            else:
                index = self.create_safetensor_index(model_path)
                index_list.append(index)
        # check index
        if not all(index_list[0]["metadata"]["total_size"] == index["metadata"]["total_size"] for index in index_list):
            raise ValueError("Weights total_size mismatch. Please make sure you load the correct weight file")
        if not all(index_list[0]["weight_map"].keys() == index["weight_map"].keys() for index in index_list):
            raise ValueError("Weights weight_map mismatch. Please make sure you load the correct weight file")
        # init new index
        index = {}
        index["metadata"] = index_list[0]["metadata"]
        index["metadata"]["total_size"] = int(index["metadata"]["total_size"])
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

            for i, model_path in enumerate(self.merge_config.model_path_list):
                with fast_safe_open(os.path.join(model_path, index_list[i]["weight_map"][k]), framework="np") as w:
                    tensor = w.get_tensor(k)
                    dtype = tensor.dtype
                    # dtype==bfloat16: numpy(uint16) -> paddle(bfloat16) -> paddle(float32) -> numpy(float32)
                    if tensor.dtype == np.uint16:
                        tensor = paddle.to_tensor(tensor, dtype="bfloat16").astype("float32").numpy()
                    tensor_list.append(tensor)
            if self.merge_config.base_model_path is not None:
                with fast_safe_open(
                    os.path.join(self.merge_config.base_model_path, index_list[-1]["weight_map"][k]),
                    framework="np",
                ) as w:
                    base_tensor = w.get_tensor(k)
                    if base_tensor.dtype == np.uint16:
                        base_tensor = paddle.to_tensor(base_tensor, dtype="bfloat16").astype("float32").numpy()
                tensor_list = [tensor - base_tensor for tensor in tensor_list]
            merge_state_dict[k] = self.merge_method.merge(tensor_list)
            if self.merge_config.base_model_path is not None:
                merge_state_dict[k] += base_tensor
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if dtype == np.uint16:
                merge_state_dict[k] = paddle.to_tensor(merge_state_dict[k], dtype="float32").astype("bfloat16").numpy()
            del tensor_list
            if self.merge_config.base_model_path is not None:

                del base_tensor
        save_file(
            merge_state_dict,
            os.path.join(self.merge_config.output_path, shard_file),
            metadata={"format": "np"},
        )
        del merge_state_dict
        gc.collect()

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
            file_type = "safetensors"
        elif os.path.exists(os.path.join(model_path, self.safe_weight_name())):
            file_type = "safetensors_without_index"
        elif os.path.exists(os.path.join(model_path, self.weight_name())):
            file_type = "pdparams"
        else:
            raise ValueError(
                f"Please check path {model_path} is correct. Support safetensors and pdparams in complete parameter format (not TP or PP format) only."
            )
        return file_type

    def weight_name(self):
        if self.merge_config.merge_preifx == "model":
            return PADDLE_WEIGHTS_NAME
        else:
            return PADDLE_MASTER_WEIGHTS_NAME

    def safe_weight_name(self):
        if self.merge_config.merge_preifx == "model":
            return SAFE_WEIGHTS_NAME
        else:
            return SAFE_MASTER_WEIGHTS_NAME

    def safe_index_name(self):
        if self.merge_config.merge_preifx == "model":
            return SAFE_WEIGHTS_INDEX_NAME
        else:
            return SAFE_MASTER_WEIGHTS_INDEX_NAME
