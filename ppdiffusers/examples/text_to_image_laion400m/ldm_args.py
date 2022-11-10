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

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="CompVis/ldm_laion400M_pretrain",
        metadata={"help": "Path to pretrained model or model"})
    num_inference_steps: Optional[int] = field(
        default=200, metadata={"help": "num_inference_steps"})
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased",
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as model_name"
        })
    model_max_length: Optional[int] = field(
        default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    pretrained_text_encoder_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained text encoder name"})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """
    file_list: Optional[str] = field(
        default="./data/filelist/train.filelist.list",
        metadata={"help": "The name of the file_list."})
    resolution: Optional[str] = field(
        default=256,
        metadata={
            "help":
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        })
    num_records: Optional[str] = field(default=10000000,
                                       metadata={"help": "num_records"})
    buffer_size: int = field(
        default=100,
        metadata={"help": "Buffer size"},
    )
    shuffle_every_n_samples: int = field(
        default=5,
        metadata={"help": "shuffle_every_n_samples."},
    )
