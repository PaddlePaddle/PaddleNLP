#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import itertools


def create_variants(
    scr_length: list, max_length: list, total_max_length: list, batch_size: list, decode_strategy: list
):
    return [
        dict(zip(["src_length", "max_length", "total_max_length", "batch_size", "decode_strategy"], v))
        for v in itertools.product(scr_length, max_length, total_max_length, batch_size, decode_strategy)
    ]


def add_testcase(
    dict_lst: dict,
    model_full_name,
    scr_length: list,
    max_length: list,
    total_max_length: list,
    batch_size: list,
    decode_strategy: list,
):
    case_dict = dict_lst.setdefault(model_full_name, {})
    case_dict["variants"] = {
        "inference": create_variants(scr_length, max_length, total_max_length, batch_size, decode_strategy)
    }
    case_dict["output_file"] = f"{model_full_name.lower().split('/')[-1]}.json"


test_case_lst = {}
skip_case_lst = {}

for i in ["bat", "pr", "sanity", "full"]:
    test_case_lst.setdefault(i, {})
    skip_case_lst.setdefault(i, {})
    add_testcase(
        test_case_lst[i],
        "meta-llama/Llama-2-7b-chat",
        ["128"],
        ["128"],
        ["256"],
        ["1", "16"],
        ["sampling", "greedy_search", "beam_search"],
    )

# testcase list + model name + scr_length(list) + max_length(list) + total_max_length (list)  + batch_size(list) + decode_strategy (list)
add_testcase(
    test_case_lst["sanity"],
    "meta-llama/Llama-2-13b-chat",
    ["128"],
    ["128"],
    ["256"],
    ["1", "16"],
    ["greedy_search", "greedy_search", "beam_search"],
)
add_testcase(
    test_case_lst["sanity"], "meta-llama/Llama-2-70b-chat", ["128"], ["128"], ["256"], ["1", "16"], ["greedy_search"]
)

# when filter passwdown 'stable' will load this list
# this list for the unstable test case to skip
skip_case_lst["sanity"]["stable"] = []

# when filter passwdown 'unstable' will load this list
# this list for the stable test case to skip
skip_case_lst["sanity"]["unstable"] = []
