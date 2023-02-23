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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # use pretrained vae kl-8.ckpt (CompVis/stable-diffusion-v1-4/vae)
    vae_name_or_path: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/vae", metadata={"help": "pretrained_vae_name_or_path"}
    )
    text_encoder_name_or_path: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/text_encoder", metadata={"help": "text_encoder_name_or_path"}
    )
    unet_name_or_path: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/unet", metadata={"help": "unet_encoder_name_or_path"}
    )
    tokenizer_name: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/tokenizer",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_max_length: Optional[int] = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: Optional[int] = field(default=200, metadata={"help": "num_inference_steps"})
    use_ema: Optional[bool] = field(default=False, metadata={"help": "Whether or not use ema"})
    pretrained_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model, when we want to resume training."}
    )
    image_logging_steps: Optional[int] = field(default=1000, metadata={"help": "Log image every X steps."})
    sd_locked: Optional[bool] = field(default=True, metadata={"help": "lock unet output_blocks and out."})

    controlnet_config_file: Optional[str] = field(
        default="./controlnet.json", metadata={"help": "controlnet_config_file"}
    )
    # TODO, not support this
    only_mid_control: Optional[bool] = field(default=False, metadata={"help": "only_mid_control."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    file_list: str = field(
        default="./data/filelist/train.filelist.list", metadata={"help": "The name of the file_list."}
    )
    resolution: int = field(
        default=512,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    num_records: int = field(default=10000000, metadata={"help": "num_records"})
    buffer_size: int = field(
        default=100,
        metadata={"help": "Buffer size"},
    )
    shuffle_every_n_samples: int = field(
        default=5,
        metadata={"help": "shuffle_every_n_samples."},
    )
    file_path: str = field(default="./fill50k", metadata={"help": "The path to of the fill50k."})
