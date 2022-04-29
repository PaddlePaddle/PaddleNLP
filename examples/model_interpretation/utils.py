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
"""This file contains some public functions
"""


def convert_tokenizer_res_to_old_version(tokenized_res):
    if isinstance(tokenized_res, list):
        return tokenized_res
    if isinstance(tokenized_res, dict):
        if len(tokenized_res['input_ids']) == 0 or not isinstance(
                tokenized_res['input_ids'][0], list):
            return tokenized_res
        else:
            res = []
            for idx in range(len(tokenized_res['input_ids'])):
                temp_dict = {}
                key_list = list(tokenized_res.keys())
                for key in key_list:
                    temp_dict[key] = tokenized_res[key][idx]
                res.append(temp_dict)
            return res
    else:
        raise ValueError('unsupported result type')
