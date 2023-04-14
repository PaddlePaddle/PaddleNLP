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
import itertools
import math
import os

import paddle
from control import (
    ControlNet,
    ControlNetTrainer,
    DataArguments,
    Fill50kDataset,
    ModelArguments,
)

from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.utils.log import logger


def unfreeze_params(params):
    for param in params:
        param.stop_gradient = False


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # report to custom_visualdl
    training_args.report_to = ["custom_visualdl"]
    training_args.resolution = data_args.resolution
    training_args.image_logging_steps = model_args.image_logging_steps = (
        math.ceil(model_args.image_logging_steps / training_args.logging_steps) * training_args.logging_steps
    )
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    model = ControlNet(model_args)
    train_dataset = Fill50kDataset(model.tokenizer, data_args.file_path)

    trainer = ControlNetTrainer(
        model=model, args=training_args, train_dataset=train_dataset, tokenizer=model.tokenizer
    )
    # must set recompute after trainer init
    trainer.model.set_recompute(training_args.recompute)

    if not model_args.sd_locked:
        params_to_train = itertools.chain(
            trainer.model.controlnet.parameters(),
            trainer.model.unet.up_blocks.parameters(),
            trainer.model.unet.conv_norm_out.parameters(),
            trainer.model.unet.conv_out.parameters(),
        )
        unfreeze_params(params_to_train)
    else:
        params_to_train = trainer.model.controlnet.parameters()
    trainer.set_optimizer_grouped_parameters(params_to_train)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
