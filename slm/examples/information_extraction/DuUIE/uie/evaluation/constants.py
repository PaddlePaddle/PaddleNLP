#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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

from dataclasses import dataclass

spot_prompt = "<spot>"
asoc_prompt = "<asoc>"

type_start = "<extra_id_0>"
type_end = "<extra_id_1>"
text_start = "<extra_id_2>"
span_start = "<extra_id_5>"
null_span = "<extra_id_6>"
null_lÍabel = "<extra_id_7>"

offset_map_strategy = {
    "closest_en": {
        "map_strategy": "closest",
        "de_duplicate": True,
        "span_to_token": "space",
    },
    "closest_zh": {
        "map_strategy": "closest",
        "de_duplicate": True,
        "span_to_token": "list",
    },
    "fisrt_en": {
        "map_strategy": "first",
        "de_duplicate": True,
        "span_to_token": "space",
    },
    "first_zh": {
        "map_strategy": "first",
        "de_duplicate": True,
        "span_to_token": "list",
    },
    "longer_first_zh": {
        "map_strategy": "longer_first",
        "de_duplicate": True,
        "span_to_token": "list",
    },
}


@dataclass
class BaseStructureMarker:
    sent_start = "<extra_id_0>"
    sent_end = "<extra_id_1>"
    record_start = "<extra_id_0>"
    record_end = "<extra_id_1>"
    span_start = "<extra_id_0>"
    span_end = "<extra_id_1>"
    text_start = "<extra_id_2>"
    source_span_start = "<extra_id_3>"
    source_span_end = "<extra_id_4>"
    target_span_start = "<extra_id_5>"
    null_span = "<extra_id_6>"
    null_label = "<extra_id_7>"
