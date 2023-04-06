# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import cv2
import numpy as np
import paddle
from paddle.io import Dataset


class Fill50kDataset(Dataset):
    def __init__(self, tokenizer, file_path="./fill50k"):
        self.data = []
        self.file_path = file_path
        with open(os.path.join(file_path, "prompt.json"), "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.text_processing = lambda caption: tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="np",
        ).input_ids[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(os.path.join(self.file_path, source_filename))
        target = cv2.imread(os.path.join(self.file_path, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        input_ids = self.text_processing(prompt)

        return dict(
            input_ids=paddle.to_tensor(input_ids, dtype=paddle.int64),
            pixel_values=paddle.to_tensor(target.transpose([2, 0, 1]), dtype=paddle.float32),
            controlnet_cond=paddle.to_tensor(source.transpose([2, 0, 1]), dtype=paddle.float32),
        )
