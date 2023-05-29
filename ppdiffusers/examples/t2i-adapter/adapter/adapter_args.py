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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    adapter_config_file: Optional[str] = field(
        default="./config/openpose_adapter.json", metadata={"help": "adapter_config_file"}
    )
    vae_name_or_path: Optional[str] = field(default=None, metadata={"help": "pretrained_vae_name_or_path"})
    text_encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "text_encoder_name_or_path"})
    unet_name_or_path: Optional[str] = field(default=None, metadata={"help": "unet_encoder_name_or_path"})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_max_length: Optional[int] = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: Optional[int] = field(default=50, metadata={"help": "num_inference_steps"})
    use_ema: bool = field(default=False, metadata={"help": "Whether or not use ema"})
    pretrained_model_name_or_path: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    pretrained_adapter_name_or_path: str = field(
        default=None,
        metadata={
            "help": "The pretrained weight of adapter, which is used to facilitate loading the same initialization for training."
        },
    )
    image_logging_steps: Optional[int] = field(default=1000, metadata={"help": "Log image every X steps."})
    use_paddle_conv_init: bool = field(default=False, metadata={"help": "Whether or not use paddle conv2d init."})
    is_ldmbert: bool = field(default=False, metadata={"help": "Whether to use ldmbert."})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "enable_xformers_memory_efficient_attention."}
    )
    control_type: Optional[str] = field(default="canny", metadata={"help": "The type of control"})
    latents_path: str = field(
        default=None,
        metadata={"help": "Path to latents, used for alignment."},
    )
    random_alignment: bool = field(default=False, metadata={"help": "Whether to align random."})
    timestep_sample_schedule: Optional[str] = field(
        default="linear",
        metadata={
            "help": "The type of timestep-sampling schedule during training, select from ['linear', 'cosine', 'cubic']."
        },
    )


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
    data_format: str = field(
        default="default",
        metadata={
            "help": "The data format, must be 'default' or 'img2img'.  The img2img format directly provides control image."
        },
    )


@dataclass
class GenerateArguments:
    """
    Arguments pertaining to specify the model generation settings.
    """

    use_controlnet: bool = field(default=False, metadata={"help": "Whether or not use text condition"})
    adapter_model_name_or_path: str = field(default=None, metadata={"help": "adapter model name or path."})
    sd_model_name_or_path: str = field(default=None, metadata={"help": "sd model name or path."})
    file: str = field(default="data/test.openpose.filelist", metadata={"help": "eval file."})
    seed: int = field(default=42, metadata={"help": "random seed."})
    scheduler_type: str = field(
        default="ddim",
        metadata={"help": "Type of scheduler to use. Should be one of ['pndm', 'lms', 'ddim', 'euler-ancest']"},
    )
    device: str = field(default="gpu", metadata={"help": "device"})
    batch_size: int = field(default=16, metadata={"help": "batch_size"})
    num_inference_steps: int = field(default=50, metadata={"help": "num_inference_steps"})
    save_path: str = field(default="output/adapter/", metadata={"help": "Path to the output file."})
    guidance_scales: str = field(default_factory=lambda: [5, 7, 9], metadata={"help": "guidance_scales list."})
    height: int = field(default=512, metadata={"help": "height."})
    width: int = field(default=512, metadata={"help": "width."})
    max_generation_limits: int = field(default=1000, metadata={"help": "max generation limits."})
    use_text_cond: bool = field(default=True, metadata={"help": "Whether or not use text condition"})
    use_default_neg_text_cond: bool = field(
        default=True, metadata={"help": "Whether or not use default negative text condition"}
    )
    generate_data_format: str = field(default="img2img", metadata={"help": "Generate data format."})
    generate_control_image_processor_type: str = field(default="openpose", metadata={"help": "Generate data format."})
