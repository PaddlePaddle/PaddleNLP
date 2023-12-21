# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


def get_pretrain_arguments(pretrain_arguments):

    configs = {}

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 8
    train_args["pipeline_parallel_degree"] = 1
    train_args["pretrain_arguments_name"] = "TP8"
    configs["TP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 2
    train_args["pretrain_arguments_name"] = "TP4PP2"
    configs["TP4PP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    train_args["pretrain_arguments_name"] = "TP4DP2"
    configs["TP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 4
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    train_args["pretrain_arguments_name"] = "TP4Sharding2"
    configs["TP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 2
    train_args["pipeline_parallel_degree"] = 4
    train_args["pretrain_arguments_name"] = "TP2PP4"
    configs["TP2PP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 2
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 4
    train_args["pretrain_arguments_name"] = "TP2Sharding4"
    configs["TP2Sharding4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 8
    train_args["pretrain_arguments_name"] = "PP8"
    configs["PP8"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 4
    train_args["sharding"] = ""
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    train_args["pretrain_arguments_name"] = "PP4DP2"
    configs["PP4DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 2
    train_args["pretrain_arguments_name"] = "PP4Sharding2"
    configs["PP4Sharding2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding8S1"
    configs["Sharding8S1"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding8S2"
    configs["Sharding8S2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding4S1DP2"
    configs["Sharding4S1DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 4
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding4S2DP2"
    configs["Sharding4S2DP2"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage1"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding2S1DP4"
    configs["Sharding2S1DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 2
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "Sharding2S2DP4"
    configs["Sharding2S2DP4"] = train_args

    train_args = copy.deepcopy(pretrain_arguments)
    train_args["tensor_parallel_degree"] = 1
    train_args["pipeline_parallel_degree"] = 1
    train_args["sharding_parallel_degree"] = 1
    train_args["sharding"] = "stage2"
    train_args["gradient_accumulation_steps"] = train_args["gradient_accumulation_steps"] // 8
    train_args["pretrain_arguments_name"] = "DP8"
    configs["DP8"] = train_args

    return configs
