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

from .merge_dare import MergeDare
from .merge_della import MergeDella
from .merge_linear import MergeLinear
from .merge_slerp import MergeSlerp
from .merge_ties import MergeTies
from .merge_utils import divide_positions

MERGE_MAPIING = {"linear": MergeLinear, "slerp": MergeSlerp, "ties": MergeTies, "dare": MergeDare, "della": MergeDella}


class MergeModel:
    def __init__(self, merge_config):
        self.merge_config = merge_config
        self.merge_method = MERGE_MAPIING[self.merge_config.merge_type](self.merge_config)

    def merge_model(self, model_path0, model_path1, output_path):
        is_safetensor0 = self.check_model_path(model_path0)
        is_safetensor1 = self.check_model_path(model_path1)
        if is_safetensor0 and is_safetensor1:
            self.merge_safetensor_model(model_path0, model_path1, output_path)
        else:
            raise NotImplementedError("Not support non safetensors models.")

    def merge_safetensor_model(self, model_path0, model_path1, output_path):
        with open(os.path.join(model_path0, self.safe_index_name()), "r", encoding="utf-8") as f:
            index0 = json.load(f)
        with open(os.path.join(model_path1, self.safe_index_name()), "r", encoding="utf-8") as f:
            index1 = json.load(f)
        if index0["metadata"]["total_size"] != index1["metadata"]["total_size"]:
            raise ValueError("Weights total_size mismatch. " "Please make sure you load the correct weight file")
        if index0["weight_map"].keys() != index1["weight_map"].keys():
            raise ValueError("Weights weight_map mismatch. Please make sure you load the correct weight file")
        key_list = list(index0["weight_map"].keys())
        positions = divide_positions(len(key_list), self.merge_config.n_process)
        index = {}
        index["metadata"] = index0["metadata"]
        index["weight_map"] = {}
        # Multi-process update
        threads = []
        for i in range(len(positions) - 1):
            shard_file = f"{self.merge_config.merge_preifx}-{i+1:05d}-of-{self.merge_config.n_process:05d}.safetensors"
            t = Process(
                target=self.shard_merge,
                args=(
                    key_list[positions[i] : positions[i + 1]],  # key_list
                    index0["weight_map"],  # weight_map0
                    index1["weight_map"],  # weight_map1
                    model_path0,  # model_path0
                    model_path1,  # model_path1
                    shard_file,
                    output_path,
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
        save_index_file = os.path.join(output_path, self.safe_index_name())
        if save_index_file and not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)

    def merge_safetensor_model_with_base(self, model_path0, model_path1, model_path_base, output_path):
        with open(os.path.join(model_path0, self.safe_index_name()), "r", encoding="utf-8") as f:
            index0 = json.load(f)
        with open(os.path.join(model_path1, self.safe_index_name()), "r", encoding="utf-8") as f:
            index1 = json.load(f)
        with open(os.path.join(model_path_base, self.safe_index_name()), "r", encoding="utf-8") as f:
            index_base = json.load(f)

        if (
            index0["metadata"]["total_size"]
            != index1["metadata"]["total_size"] | index0["metadata"]["total_size"]
            != index_base["metadata"]["total_size"] | index1["metadata"]["total_size"]
            != index_base["metadata"]["total_size"]
        ):
            raise ValueError("Weights total_size mismatch. " "Please make sure you load the correct weight file")
        if (
            index0["weight_map"].keys()
            != index1["weight_map"].keys() | index0["weight_map"].keys()
            != index_base["weight_map"].keys() | index1["weight_map"].keys()
            != index_base["weight_map"].keys()
        ):
            raise ValueError("Weights weight_map mismatch. Please make sure you load the correct weight file")
        key_list = list(index0["weight_map"].keys())
        positions = divide_positions(len(key_list), self.merge_config.n_process)
        index = {}
        index["metadata"] = index0["metadata"]
        index["weight_map"] = {}
        # Multi-process update
        threads = []
        for i in range(len(positions) - 1):
            shard_file = f"{self.merge_config.merge_preifx}-{i+1:05d}-of-{self.merge_config.n_process:05d}.safetensors"
            t = Process(
                target=self.shard_merge_with_base,
                args=(
                    key_list[positions[i] : positions[i + 1]],  # key_list
                    index0["weight_map"],  # weight_map0
                    index1["weight_map"],  # weight_map1
                    index_base["weight_map"],
                    model_path0,  # model_path0
                    model_path1,  # model_path1
                    model_path_base,
                    shard_file,
                    output_path,
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
        save_index_file = os.path.join(output_path, self.safe_index_name())
        if save_index_file and not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2) + "\n"
            f.write(content)

    def shard_merge(self, key_list, weight_map0, weight_map1, model_path0, model_path1, shard_file, output_path):
        merge_state_dict = {}
        for k in key_list:
            with fast_safe_open(os.path.join(model_path0, weight_map0[k]), framework="np") as w:
                v0 = w.get_tensor(k)
                # print("v0:",v0)
                # print("v0的type",type(v0))
                # print("v0 shape:", v0.shape)
                # print("k的值",k)
            with fast_safe_open(os.path.join(model_path1, weight_map1[k]), framework="np") as w:
                v1 = w.get_tensor(k)
            # print("v1:",v1)
            # dtype==bfloat16: numpy(uint16) -> paddle(bfloat16) -> paddle(float32) -> numpy(float32)
            if v0.dtype == np.uint16:
                v0 = paddle.to_tensor(v0, dtype="bfloat16").astype("float32").numpy()
            if v1.dtype == np.uint16:
                v1 = paddle.to_tensor(v1, dtype="bfloat16").astype("float32").numpy()
            print("merge method:", self.merge_method)
            print("k的值:", k)
            print("v0", v0)
            print("v1", v1)
            merge_state_dict[k] = self.merge_method.merge_op(v0, v1)
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if self.merge_config.dtype == "bfloat16":
                merge_state_dict[k] = paddle.to_tensor(merge_state_dict[k], dtype="float32").astype("bfloat16").numpy()
        save_file(
            merge_state_dict,
            os.path.join(output_path, shard_file),
            metadata={"format": "np"},
        )

    # 这个是用于有base_model
    def shard_merge_with_base(
        self,
        key_list,
        weight_map0,
        weight_map1,
        weight_map_base,
        model_path0,
        model_path1,
        model_path_base,
        shard_file,
        output_path,
    ):
        merge_state_dict = {}
        for k in key_list:
            with fast_safe_open(os.path.join(model_path0, weight_map0[k]), framework="np") as w:
                v0 = w.get_tensor(k)
            with fast_safe_open(os.path.join(model_path1, weight_map1[k]), framework="np") as w:
                v1 = w.get_tensor(k)
            with fast_safe_open(os.path.join(model_path_base, weight_map_base[k]), framework="np") as w:
                vb = w.get_tensor(k)
            # dtype==bfloat16: numpy(uint16) -> paddle(bfloat16) -> paddle(float32) -> numpy(float32)
            if v0.dtype == np.uint16:
                v0 = paddle.to_tensor(v0, dtype="bfloat16").astype("float32").numpy()
            if v1.dtype == np.uint16:
                v1 = paddle.to_tensor(v1, dtype="bfloat16").astype("float32").numpy()
            if vb.dtype == np.uint16:
                vb = paddle.to_tensor(vb, dtype="bfloat16").astype("float32").numpy()
            merge_state_dict[k] = self.merge_method.merge_op(v0 - vb, v1 - vb) + vb
            # dtype==bfloat16: numpy(float32) -> paddle(float32) -> paddle(bfloat16) -> numpy(uint16)
            if self.merge_config.dtype == "bfloat16":
                merge_state_dict[k] = paddle.to_tensor(merge_state_dict[k], dtype="float32").astype("bfloat16").numpy()
        save_file(
            merge_state_dict,
            os.path.join(output_path, shard_file),
            metadata={"format": "np"},
        )

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
