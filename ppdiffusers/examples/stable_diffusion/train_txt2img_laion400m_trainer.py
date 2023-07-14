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
import os

import paddle
from sd import (
    SDDataArguments,
    SDModelArguments,
    SDTrainingArguments,
    StableDiffusionModel,
    StableDiffusionTrainer,
    TextImagePair,
)

from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint, set_seed
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((SDModelArguments, SDDataArguments, SDTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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
    if training_args.seed is not None:
        set_seed(training_args.seed)

    model = StableDiffusionModel(model_args)
    model.set_recompute(training_args.recompute)
    model.set_xformers(training_args.enable_xformers_memory_efficient_attention)
    model.set_ema(training_args.use_ema)

    if training_args.to_static:
        input_ids = paddle.static.InputSpec(name="input_ids", shape=[-1, model_args.model_max_length], dtype="int64")
        pixel_values = paddle.static.InputSpec(
            name="pixel_values", shape=[-1, 3, training_args.resolution, training_args.resolution], dtype="float32"
        )
        specs = [input_ids, pixel_values]
        paddle.jit.ignore_module([os])
        model = paddle.jit.to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(specs))

    train_dataset = TextImagePair(
        file_list=data_args.file_list,
        size=training_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation=data_args.interpolation,
        tokenizer=model.tokenizer,
    )

    trainer = StableDiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model.tokenizer,
    )

    if model_args.train_text_encoder:
        if training_args.text_encoder_learning_rate == training_args.unet_learning_rate:
            params_to_train = itertools.chain(model.text_encoder.parameters(), model.unet.parameters())
        else:
            # overwrite default learning rate with 1.0
            training_args.learning_rate = 1.0
            params_to_train = [
                {
                    "params": model.text_encoder.parameters(),
                    "learning_rate": training_args.text_encoder_learning_rate,
                },
                {
                    "params": model.unet.parameters(),
                    "learning_rate": training_args.unet_learning_rate,
                },
            ]
    else:
        params_to_train = model.unet.parameters()
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
