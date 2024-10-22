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

import os
import sys
import unittest
from unittest import skip

import numpy as np
import paddle
from parameterized import parameterized_class

from paddlenlp.transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .testing_utils import LLMTest, argv_context_guard, load_test_config

# fmt:off
all_inputs = [
    # llama-7b
    [[1, 910, 3461, 8128, 3239, 2472, 322, 5626, 363, 11559, 373, 2473, 6360, 9580, 545, 358, 313, 17870, 29925, 29897, 14974, 6360, 9580, 545, 358, 313, 17870, 29925, 29897, 322, 2908, 15649, 8078, 292, 313, 29933, 5371, 29897, 526, 4266, 8078, 292, 7208, 12903, 393, 11559, 3635, 1169, 278, 10317, 310, 5282, 1947, 313, 3970, 29928, 29897, 304, ]],
    # qwen-7b
    [[1986, 1895, 5707, 4004, 1995, 323, 4714, 369, 7992, 389, 7299, 3157, 52578, 320, 44, 9954, 8, 323, 2504, 3695, 20358, 3157, 52578, 320, 44, 9954, 8, 323, 2504, 3695, 59406, 320, 66755, 8, 525, 3281, 59406, 23783, 429, 7992, 28690, 279, 5887, 315, 16373, 320, 35, 2069, 8, 311, 990, 369, 264, 7199, 1372, 315, 9055, 23390, ]],
    # chatglm3-6b
    [[64790, 64792, 666, 1284, 2736, 4467, 1097, 293, 2326, 332, 4168, 331, 5332, 2475, 23355, 359, 26594, 30947, 30945, 293, 15903, 2475, 23355, 359, 26594, 30947, 30945, 293, 3579, 2505, 26317, 359, 54223, 30945, 383, 1720, 26317, 11972, 343, 4168, 15125, 267, 2902, 290, 10196, 359, 30952, 3809, 30945, 289, 792, 332, 260, 3666, 1276, 290, 5735, 10625, ]],
    # chatglm-6b
    [[200, 647, 986, 1186, 320, 102, 953, 108, 2355, 111, 1297, 626, 26020, 19, 10806, 266, 14, 102, 130001, 130004, 6723, 626, 26020, 19, 10806, 266, 14, 102, 1204, 1784, 27817, 19, 27798, 14, 118, 972, 27817, 2055, 109, 2355, 9187, 100, 1334, 101, 7319, 19, 9220, 234, 14, 103, 179, 108, 104, 1132, 277, 101, 2576, 6225, ]],
    # bloom
    [[55, 75, 76, 86, 210, 85, 72, 83, 82, 85, 87, 210, 83, 85, 82, 89, 76, 71, 72, 86, 48, 88, 79, 87, 76, 92, 72, 68, 85, 210, 83, 85, 82, 70, 88, 85, 72, 80, 72, 81, 87, 210, 11, 48, 60, 51, 12, 210, 68, 81, 71, 210, 69, 79, 82, 70, 78, 210, ]],
]
all_position_ids = [
    # llama-7b
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]],
    # qwen07b
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]],
    # chatglm3-6b
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]],
    # chatglm-6b
    [
        [
            [18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, ],
        ]
    ],
    # bloom
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, ]],
]
all_attention_mask = [
    # llama
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]],
    # qwen
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]],
    # chatglm3-6b
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]],
    # chatglm-6b
    [
        [
            [
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, ],
                [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, ],
            ]
        ]
    ],
    # bloom
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ]],
]
all_labels = [
    # llama
    [[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 14974, 6360, 9580, 545, 358, 313, 17870, 29925, 29897, 322, 2908, 15649, 8078, 292, 313, 29933, 5371, 29897, 526, 4266, 8078, 292, 7208, 12903, 393, 11559, 3635, 1169, 278, 10317, 310, 5282, 1947, 313, 3970, 29928, 29897, 304, 671, ]],
    # qwen
    [[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 20358, 3157, 52578, 320, 44, 9954, 8, 323, 2504, 3695, 59406, 320, 66755, 8, 525, 3281, 59406, 23783, 429, 7992, 28690, 279, 5887, 315, 16373, 320, 35, 2069, 8, 311, 990, 369, 264, 7199, 1372, 315, 9055, 23390, 7468, ]],
    # chatglm3-6b
    [[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 15903, 2475, 23355, 359, 26594, 30947, 30945, 293, 3579, 2505, 26317, 359, 54223, 30945, 383, 1720, 26317, 11972, 343, 4168, 15125, 267, 2902, 290, 10196, 359, 30952, 3809, 30945, 289, 792, 332, 260, 3666, 1276, 290, 5735, 10625, 3181, ]],
    # chatglm-6b
    [[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 130004, 6723, 626, 26020, 19, 10806, 266, 14, 102, 1204, 1784, 27817, 19, 27798, 14, 118, 972, 27817, 2055, 109, 2355, 9187, 100, 1334, 101, 7319, 19, 9220, 234, 14, 103, 179, 108, 104, 1132, 277, 101, 2576, 6225, 1785, ]],
    # bloom
    [[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 48, 88, 79, 87, 76, 92, 72, 68, 85, 210, 83, 85, 82, 70, 88, 85, 72, 80, 72, 81, 87, 210, 11, 48, 60, 51, 12, 210, 68, 81, 71, 210, 69, 79, 82, 70, 78, 210, 69, ]],
]

all_ppl = [
    # llama
    31361.590644223128,
    31361.590644223128,
    31362.757106912533,
    31361.62055298091,
    # qwen
    155909.83795939674,
    155939.57823718787,
    155917.27249705535,
    155909.83795939674,
    # chatglm3-6b
    64415.31959719674,
    64454.8934643284,
    64416.60966606845,
    64420.172847651804,
    # chatglm-6b
    130540.64669131214,
    130573.01895270264,
    130539.15278071642,
    130538.4058318297,
    # llama-alibi
    31369.517462860927,
    # bloom-alibi
    251106.84487228873,
]
# fmt:on


@parameterized_class(
    [
        "model_name_or_path",
        "strategy_type",
        "strategy_name",
        "inputs",
        "positin_ids",
        "labels",
        "attention_mask",
        "ppl",
    ],
    [
        [
            "__internal_testing__/micro-random-llama",
            "embedding_strategies",
            "RotaryEmbedding",
            all_inputs[0],
            all_position_ids[0],
            all_labels[0],
            all_attention_mask[0],
            all_ppl[0],
        ],
        [
            "__internal_testing__/micro-random-llama",
            "embedding_strategies",
            "LinearScalingRotaryEmbedding",
            all_inputs[0],
            all_position_ids[0],
            all_labels[0],
            all_attention_mask[0],
            all_ppl[1],
        ],
        [
            "__internal_testing__/micro-random-llama",
            "embedding_strategies",
            "NTKScalingRotaryEmbedding",
            all_inputs[0],
            all_position_ids[0],
            all_labels[0],
            all_attention_mask[0],
            all_ppl[2],
        ],
        [
            "__internal_testing__/micro-random-llama",
            "embedding_strategies",
            "DynamicNTKScalingRotaryEmbedding",
            all_inputs[0],
            all_position_ids[0],
            all_labels[0],
            all_attention_mask[0],
            all_ppl[3],
        ],
        [
            "__internal_testing__/tiny-new-random-qwen-7b",
            "embedding_strategies",
            "RotaryEmbedding",
            all_inputs[1],
            all_position_ids[1],
            all_labels[1],
            all_attention_mask[1],
            all_ppl[4],
        ],
        [
            "__internal_testing__/tiny-new-random-qwen-7b",
            "embedding_strategies",
            "LinearScalingRotaryEmbedding",
            all_inputs[1],
            all_position_ids[1],
            all_labels[1],
            all_attention_mask[1],
            all_ppl[5],
        ],
        [
            "__internal_testing__/tiny-new-random-qwen-7b",
            "embedding_strategies",
            "NTKScalingRotaryEmbedding",
            all_inputs[1],
            all_position_ids[1],
            all_labels[1],
            all_attention_mask[1],
            all_ppl[6],
        ],
        [
            "__internal_testing__/tiny-new-random-qwen-7b",
            "embedding_strategies",
            "DynamicNTKScalingRotaryEmbedding",
            all_inputs[1],
            all_position_ids[1],
            all_labels[1],
            all_attention_mask[1],
            all_ppl[7],
        ],
        [
            "__internal_testing__/tiny-new-random-chatglm3-6b",
            "embedding_strategies",
            "RotaryEmbedding",
            all_inputs[2],
            all_position_ids[2],
            all_labels[2],
            all_attention_mask[2],
            all_ppl[8],
        ],
        [
            "__internal_testing__/tiny-new-random-chatglm3-6b",
            "embedding_strategies",
            "LinearScalingRotaryEmbedding",
            all_inputs[2],
            all_position_ids[2],
            all_labels[2],
            all_attention_mask[2],
            all_ppl[9],
        ],
        [
            "__internal_testing__/tiny-new-random-chatglm3-6b",
            "embedding_strategies",
            "NTKScalingRotaryEmbedding",
            all_inputs[2],
            all_position_ids[2],
            all_labels[2],
            all_attention_mask[2],
            all_ppl[10],
        ],
        [
            "__internal_testing__/tiny-new-random-chatglm3-6b",
            "embedding_strategies",
            "DynamicNTKScalingRotaryEmbedding",
            all_inputs[2],
            all_position_ids[2],
            all_labels[2],
            all_attention_mask[2],
            all_ppl[11],
        ],
        # fmt:off
        # [
        #     "__internal_testing__/tiny-new-random-chatglm-6b",
        #     "embedding_strategies",
        #     "RotaryEmbedding",
        #     all_inputs[3],
        #     all_position_ids[3],
        #     all_labels[3],
        #     all_attention_mask[3],
        #     all_ppl[12],
        # ],
        # [
        #     "__internal_testing__/tiny-new-random-chatglm-6b",
        #     "embedding_strategies",
        #     "LinearScalingRotaryEmbedding",
        #     all_inputs[3],
        #     all_position_ids[3],
        #     all_labels[3],
        #     all_attention_mask[3],
        #     all_ppl[13],
        # ],
        # [
        #     "__internal_testing__/tiny-new-random-chatglm-6b",
        #     "embedding_strategies",
        #     "NTKScalingRotaryEmbedding",
        #     all_inputs[3],
        #     all_position_ids[3],
        #     all_labels[3],
        #     all_attention_mask[3],
        #     all_ppl[14],
        # ],
        # [
        #     "__internal_testing__/tiny-new-random-chatglm-6b",
        #     "embedding_strategies",
        #     "DynamicNTKScalingRotaryEmbedding",
        #     all_inputs[3],
        #     all_position_ids[3],
        #     all_labels[3],
        #     all_attention_mask[3],
        #     all_ppl[15],
        # ],
        # fmt:on
        [
            "__internal_testing__/micro-random-llama",
            "attention_strategies",
            "AttentionWithLinearBias",
            all_inputs[0],
            all_position_ids[0],
            all_labels[0],
            all_attention_mask[0],
            all_ppl[16],
        ],
        [
            "__internal_testing__/tiny-random-bloom",
            "attention_strategies",
            "AttentionWithLinearBias",
            all_inputs[4],
            all_position_ids[4],
            all_labels[4],
            all_attention_mask[4],
            all_ppl[17],
        ],
    ],
)
class TestLongSequenceStrategiesTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    root_path = ""

    def setUp(self) -> None:
        super().setUp()
        sys.path.insert(0, "./llm")

    def disable_static(self):
        paddle.utils.unique_name.switch()
        paddle.disable_static()

    def get_model(self, model_name_or_path):
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        if self.strategy_type == "embedding_strategies":
            model_config.alibi = False
        else:
            model_config.alibi = True
        model_config.use_long_sequence_strategies = True
        model_config.long_sequence_strategy_type = self.strategy_type
        model_config.long_sequence_strategy_name = self.strategy_name
        max_position_embeddings = 10 if self.strategy_name == "DynamicNTKScalingRotaryEmbedding" else 2048
        model_config.long_sequence_init_args = {
            "dim": int(model_config.hidden_size / model_config.num_attention_heads),
            "max_position_embeddings": max_position_embeddings,
            "base": 10000,
            "scaling_factor": 4,
        }
        if "chatglm" in model_name_or_path:
            model_config.long_sequence_init_args["position_encoding_2d"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=model_config, dtype="float32")
        return model

    def test_long_sequence_strategies(self):
        input_ids = paddle.to_tensor(self.inputs, dtype=paddle.int64)
        position_ids = paddle.to_tensor(self.positin_ids, dtype=paddle.int64)
        attention_mask = paddle.to_tensor(self.attention_mask, dtype=paddle.int64)
        labels = paddle.to_tensor(self.labels, dtype=paddle.int64)
        ppl = self.ppl
        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        model = self.get_model(self.model_name_or_path)

        output = model(**inputs)
        self.assertTrue(
            np.allclose(
                np.exp(output[0].item()),
                ppl,
                rtol=1e-2,
            )
        )

    @skip("Skip and wait to fix.")
    def test_dynamic_to_static_inference(self):

        if (
            "qwen" not in self.model_name_or_path
            and "chatglm-6b" not in self.model_name_or_path
            and "bloom" not in self.model_name_or_path
        ):
            model = self.get_model(self.model_name_or_path)
            save_path = os.path.join(self.output_dir, self.model_name_or_path)
            model.save_pretrained(save_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            if "llama" in self.model_name_or_path:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.save_pretrained(save_path)

            self.disable_static()
            config = load_test_config(self.config_path, "inference-to-static")
            config["output_path"] = self.inference_output_dir
            config["model_name_or_path"] = save_path

            with argv_context_guard(config):
                from predict.export_model import main

                main()
