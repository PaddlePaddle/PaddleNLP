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

import math
from dataclasses import dataclass, field
from typing import List, Optional

from paddlenlp.trainer import TrainingArguments

__all__ = [
    "SDTrainingArguments",
    "SDModelArguments",
    "SDDataArguments",
]
import os


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Unsupported value encountered.")


@dataclass
class SDTrainingArguments(TrainingArguments):
    image_logging_steps: int = field(default=1000, metadata={"help": "Log image every X steps."})
    to_static: bool = field(default=False, metadata={"help": "Whether or not to_static"})
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["custom_visualdl"],
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    resolution: int = field(
        default=512,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    use_ema: bool = field(default=False, metadata={"help": "Whether or not use ema"})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "enable_xformers_memory_efficient_attention."}
    )
    only_save_updated_model: bool = field(
        default=True, metadata={"help": "Whether or not save only_save_updated_model"}
    )
    unet_learning_rate: float = field(default=None, metadata={"help": "The initial learning rate for Unet Model."})
    text_encoder_learning_rate: float = field(
        default=None, metadata={"help": "The initial learning rate for Text Encoder Model."}
    )

    def __post_init__(self):
        super().__post_init__()
        self.image_logging_steps = (
            (math.ceil(self.image_logging_steps / self.logging_steps) * self.logging_steps)
            if self.image_logging_steps > 0
            else -1
        )
        self.use_ema = str2bool(os.getenv("FLAG_USE_EMA", "False")) or self.use_ema
        self.enable_xformers_memory_efficient_attention = (
            str2bool(os.getenv("FLAG_XFORMERS", "False")) or self.enable_xformers_memory_efficient_attention
        )
        self.recompute = str2bool(os.getenv("FLAG_RECOMPUTE", "False")) or self.recompute
        self.benchmark = str2bool(os.getenv("FLAG_BENCHMARK", "False")) or self.benchmark
        self.to_static = str2bool(os.getenv("FLAG_TO_STATIC", "False")) or self.to_static

        if self.text_encoder_learning_rate is None:
            self.text_encoder_learning_rate = self.learning_rate
        if self.unet_learning_rate is None:
            self.unet_learning_rate = self.learning_rate

        # set default learning rate
        self.learning_rate = self.unet_learning_rate

        if self.to_static:
            self.use_ema = False
            self.enable_xformers_memory_efficient_attention = False
            self.recompute = False


@dataclass
class SDModelArguments:
    vae_name_or_path: Optional[str] = field(default=None, metadata={"help": "vae_name_or_path"})
    text_encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "text_encoder_name_or_path"})
    unet_name_or_path: Optional[str] = field(default=None, metadata={"help": "unet_name_or_path"})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as pretrained_model_name_or_path"},
    )
    pretrained_model_name_or_path: str = field(
        default="CompVis/stable-diffusion-v1-4",
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    model_max_length: int = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    prediction_type: str = field(
        default="epsilon",
        metadata={
            "help": "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        },
    )
    num_inference_steps: int = field(default=50, metadata={"help": "num_inference_steps"})
    train_text_encoder: bool = field(default=False, metadata={"help": "Whether or not train text encoder"})

    noise_offset: float = field(default=0, metadata={"help": "The scale of noise offset."})
    snr_gamma: Optional[float] = field(
        default=None,
        metadata={
            "help": "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."
        },
    )
    input_perturbation: Optional[float] = field(
        default=0, metadata={"help": "The scale of input perturbation. Recommended 0.1."}
    )


@dataclass
class SDDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    file_list: str = field(
        default="./data/filelist/train.filelist.list", metadata={"help": "The name of the file_list."}
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
    interpolation: str = field(
        default="lanczos",
        metadata={"help": "interpolation method"},
    )
