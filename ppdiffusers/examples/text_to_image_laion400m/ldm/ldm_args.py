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

import types
from dataclasses import asdict, dataclass, field
from typing import Optional

import paddle

from paddlenlp.utils.log import logger


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # use pretrained vae kl-8.ckpt (CompVis/stable-diffusion-v1-4/vae)
    vae_name_or_path: Optional[str] = field(
        default="CompVis/stable-diffusion-v1-4/vae", metadata={"help": "pretrained_vae_name_or_path"}
    )
    text_encoder_config_file: Optional[str] = field(
        default="./config/ldmbert.json", metadata={"help": "text_encoder_config_file"}
    )
    unet_config_file: Optional[str] = field(default="./config/unet.json", metadata={"help": "unet_config_file"})
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_max_length: Optional[int] = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: Optional[int] = field(default=200, metadata={"help": "num_inference_steps"})
    use_ema: Optional[bool] = field(default=False, metadata={"help": "Whether or not use ema"})
    pretrained_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model, when we want to resume training."}
    )
    image_logging_steps: Optional[int] = field(default=1000, metadata={"help": "Log image every X steps."})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "enable_xformers_memory_efficient_attention."}
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
        default=256,
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


@dataclass
class NoTrainerTrainingArguments:
    output_dir: str = field(
        default="outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU core/CPU for training."}
    )

    gradient_accumulation_steps: int = field(
        default=2,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.02, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=-1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: int = field(default=100, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=1000000000,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": 'The scheduler type to use. support ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'
        },
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default="logs", metadata={"help": "VisualDL log dir."})

    logging_steps: int = field(default=50, metadata={"help": "Log every X updates steps."})

    save_steps: int = field(default=5000, metadata={"help": "Save checkpoint every X updates steps."})

    seed: int = field(default=23, metadata={"help": "Random seed that will be set at the beginning of training."})
    dataloader_num_workers: int = field(
        default=6,
        metadata={
            "help": "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        },
    )
    report_to: str = field(
        default="visualdl", metadata={"help": "The list of integrations to report the results and logs to."}
    )
    recompute: bool = field(
        default=False,
        metadata={
            "help": "Recompute the forward pass to calculate gradients. Used for saving memory. "
            "Only support for networks with transformer blocks."
        },
    )

    def __str__(self):
        self_as_dict = asdict(self)
        self_as_dict = {k: f"<{k.upper()}>" if k.endswith("_token") else v for k, v in self_as_dict.items()}

        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    __repr__ = __str__

    def print_config(self, args=None, key=""):
        """
        print all config values.
        """
        logger.info("=" * 60)
        if args is None:
            args = self
            key = "Training"

        logger.info("{:^40}".format("{} Configuration Arguments".format(key)))
        logger.info("{:30}:{}".format("paddle commit id", paddle.version.commit))

        for a in dir(args):
            if a[:2] != "__":  # don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.info("{:30}:{}".format(a, v))

        logger.info("")
