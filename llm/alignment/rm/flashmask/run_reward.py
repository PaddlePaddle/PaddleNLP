# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

""" Training Reward """

import os
import sys
from functools import partial

import paddle
from data import (
    preference_collate_fn,
    preprocess_preference_data,
    preprocess_process_data,
    process_collate_fn,
    zero_padding_process_collate_fn,
)
from reward_argument import DataArgument, ModelArgument, TrainingArguments
from reward_model import LlamaModelForPRM, LlamaModelForScore, MistralModelForPRM
from reward_trainer import RewardTrainer

from paddlenlp.datasets import (
    ZeroPaddingIterableDataset,
    ZeroPaddingMapDataset,
    load_dataset,
)
from paddlenlp.trainer import (
    IntervalStrategy,
    PdArgumentParser,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.transformers import AutoConfig, AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    """main"""
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.reward_tokens is not None:
        model_args.reward_tokens = model_args.reward_tokens.split(",")

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    if training_args.max_steps > 0:
        training_args.num_train_epochs = 1
    if data_args.autotuner_benchmark:
        training_args.num_train_epochs = 1
        training_args.max_steps = 5
        training_args.do_train = True
        training_args.do_export = False
        training_args.do_predict = False
        training_args.do_eval = False
        training_args.overwrite_output_dir = True
        training_args.load_best_model_at_end = False
        training_args.report_to = []
        training_args.save_strategy = IntervalStrategy.NO
        training_args.evaluation_strategy = IntervalStrategy.NO
    if data_args.benchmark:
        training_args.do_train = True
        training_args.do_export = False
        training_args.do_predict = False
        training_args.do_eval = False
        training_args.overwrite_output_dir = True
        training_args.load_best_model_at_end = False
        training_args.save_strategy = IntervalStrategy.NO
        training_args.evaluation_strategy = IntervalStrategy.NO

    paddle.set_device(training_args.device)
    set_seed(training_args.seed)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: "
        f"{training_args.world_size}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    logger.info("Start to load tokenizer & model.")
    if model_args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        seq_length=data_args.max_seq_len,
    )
    if training_args.process_reward:
        placeholder_token_id = tokenizer(model_args.placeholder_token, add_special_tokens=False)["input_ids"]
        if len(placeholder_token_id) != 1:
            print(
                f"Warning: The length of placeholder_token_id should be 1, but got {len(placeholder_token_id)}. Using {placeholder_token_id[-1]}: {tokenizer.convert_ids_to_tokens([placeholder_token_id[-1]])} instead."
            )
        model_kwargs["placeholder_token_id"] = placeholder_token_id[-1]
        for local_tk in model_args.reward_tokens:
            if len(tokenizer(local_tk, add_special_tokens=False)["input_ids"]) != 1:
                print(
                    f"Warning: The length of reward_token_id should be 1, but got {len(tokenizer(local_tk)['input_ids'])}. Using {tokenizer(local_tk)['input_ids'][-1]}: {tokenizer.convert_ids_to_tokens([tokenizer(local_tk)['input_ids'][-1]])} instead."
                )
        model_kwargs["reward_token_ids"] = [
            tokenizer(local_tk)["input_ids"][-1] for local_tk in model_args.reward_tokens
        ]
    if training_args.pipeline_parallel_degree > 1:
        raise ValueError("RM does not support pipeline parallelism yet.")

    if not data_args.autotuner_benchmark:
        if training_args.process_reward:
            if "llama" in model_args.model_name_or_path.lower():
                model = LlamaModelForPRM.from_pretrained(**model_kwargs)
            elif "mistral" in model_args.model_name_or_path.lower():
                model = MistralModelForPRM.from_pretrained(**model_kwargs)
            else:
                raise ValueError("PRM currently only supports Llama & Mistral models.")
        else:
            model = LlamaModelForScore.from_pretrained(**model_kwargs)
    else:
        config = AutoConfig.from_pretrained(**model_kwargs)
        if training_args.process_reward:
            if "llama" in model_args.model_name_or_path.lower():
                model = LlamaModelForPRM.from_config(config)
            elif "mistral" in model_args.model_name_or_path.lower():
                model = MistralModelForPRM.from_config(config)
            else:
                raise ValueError("PRM currently only supports Llama & Mistral models.")
        else:
            model = LlamaModelForScore.from_config(config)

    if model_args.flash_mask and not model.config.use_flash_attention:
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        model.config.use_flash_attention = True

    # TODO: support chat template in next pr
    # tokenizer.chat_template = None
    logger.info("Loading tokenizer & model successfully !")

    logger.info("Start to create dataset")
    if training_args.process_reward:
        trans_func = partial(preprocess_process_data, tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    else:
        trans_func = partial(
            preprocess_preference_data, tokenizer=tokenizer, data_args=data_args, model_args=model_args
        )

    if data_args.zero_padding:
        if data_args.lazy:
            zero_padding_dataset = ZeroPaddingIterableDataset
        else:
            zero_padding_dataset = ZeroPaddingMapDataset

    if training_args.do_train and training_args.should_load_dataset:
        train_ds = load_dataset(
            "json",
            data_files=data_args.train_dataset_path,
            lazy=data_args.lazy,
        )[0]
        train_ds = train_ds.map(trans_func)
        if data_args.zero_padding:
            logger.info("Creating train Zero Padding Data Stream. This may take a few minutes.")
            train_ds = (
                zero_padding_dataset(
                    train_ds,
                    tokenizer=tokenizer,
                    max_length=data_args.max_seq_len,
                    greedy_zero_padding=data_args.greedy_zero_padding,
                )
                if train_ds is not None
                else None
            )
    else:
        train_ds = None

    if training_args.do_eval and training_args.should_load_dataset:
        eval_ds = load_dataset(
            "json",
            data_files=data_args.dev_dataset_path,
            lazy=data_args.lazy,
        )[0]
        eval_ds = eval_ds.map(trans_func)
        if data_args.zero_padding:
            logger.info("Creating dev Zero Padding Data Stream. This may take a few minutes.")
            eval_ds = (
                zero_padding_dataset(
                    eval_ds,
                    tokenizer=tokenizer,
                    max_length=data_args.max_seq_len,
                )
                if eval_ds is not None
                else None
            )
    else:
        eval_ds = None
    logger.info("Creating dataset successfully ...")

    if data_args.zero_padding:
        data_collator = partial(
            preference_collate_fn if not training_args.process_reward else zero_padding_process_collate_fn,
            max_seq_len=data_args.max_seq_len,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        data_collator = partial(process_collate_fn, pad_token_id=tokenizer.pad_token_id)

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        process_reward=training_args.process_reward,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        if not data_args.autotuner_benchmark and not data_args.benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
