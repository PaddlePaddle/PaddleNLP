# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


def read_semval2016_phone_dataset(file_path):
    """Reads data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            _, label, text_a, text_b = line.strip().split("\t")
            yield {"text": text_a, "text_pair": text_b, "label": label}


def read_cote_dp_dataset(file_path):
    """Reads data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            text, label = line.strip().split("\t")
            tokens = text.split("\x02")
            label = label.split("\x02")
            yield {"tokens": tokens, "labels": label}
