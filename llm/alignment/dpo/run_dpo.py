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

""" Training DPO """

import os
import sys
import time
from functools import partial

import paddle
from dpo_argument import DPODataArgument, DPOModelArgument, DPOTrainingArguments

from paddlenlp.datasets import ZeroPaddingMapDataset, load_dataset
from paddlenlp.trainer import (
    IntervalStrategy,
    PdArgumentParser,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForCausalLMPipe,
    register_sequence_parallel_allreduce_hooks,
)
from paddlenlp.trl import (
    DPOTrainer,
    calculate_effective_tokens,
    preference_collate_fn,
    preprocess_preference_data,
)
from paddlenlp.utils.log import logger

flash_mask_support_list = [LlamaForCausalLM, LlamaForCausalLMPipe]


def main():
    """main"""
    parser = PdArgumentParser((DPOModelArgument, DPODataArgument, DPOTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    logger.info("Start to load model & tokenizer.")
    model_kwargs = dict(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype=dtype,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        recompute_granularity=model_args.recompute_granularity,
        use_flash_attention=model_args.use_flash_attention,
        tensor_parallel_output=True,
    )
    if training_args.pipeline_parallel_degree > 1:
        raise ValueError("DPO does not support pipeline parallelism yet.")

    if not data_args.autotuner_benchmark:
        ref_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        config = AutoConfig.from_pretrained(**model_kwargs)
        model = AutoModelForCausalLM.from_config(config)
        model.set_state_dict(ref_model.state_dict())
    else:
        config = AutoConfig.from_pretrained(**model_kwargs)
        model = AutoModelForCausalLM.from_config(config)
        ref_config = AutoConfig.from_pretrained(**model_kwargs)
        ref_model = AutoModelForCausalLM.from_config(ref_config)
        model.set_state_dict(ref_model.state_dict())

    if model_args.flash_mask and not model.config.use_flash_attention:
        logger.warning("`flash_mask` must use with zero padding and flash attention.")
        model.config.use_flash_attention = True

    if model_args.flash_mask and not any(isinstance(model, cls) for cls in flash_mask_support_list):
        raise NotImplementedError(f"{model.__class__} not support flash mask.")
    if training_args.sequence_parallel:
        register_sequence_parallel_allreduce_hooks(
            model, training_args.gradient_accumulation_steps, training_args.fuse_sequence_parallel_allreduce
        )
    if model_args.tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # TODO: support chat template in next pr
    # tokenizer.chat_template = None
    logger.info("Loading model & tokenizer successfully !")

    logger.info("Start to create dataset")
    trans_func = partial(preprocess_preference_data, tokenizer=tokenizer, data_args=data_args, model_args=model_args)
    if training_args.do_train and training_args.should_load_dataset:
        train_ds = load_dataset(
            "json",
            data_files=data_args.train_dataset_path,
        )[0]
        logger.info("Creating train Zero Padding Data Stream. This may take a few minutes.")
        train_ds = (
            ZeroPaddingMapDataset(
                train_ds.map(trans_func),
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
        )[0]
        logger.info("Creating dev Zero Padding Data Stream. This may take a few minutes.")
        eval_ds = (
            ZeroPaddingMapDataset(
                eval_ds.map(trans_func),
                tokenizer=tokenizer,
                max_length=data_args.max_seq_len,
            )
            if eval_ds is not None
            else None
        )
    else:
        eval_ds = None
    logger.info("Creating dataset successfully ...")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=partial(
            preference_collate_fn,
            max_seq_len=data_args.max_seq_len,
        ),
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        if not data_args.autotuner_benchmark and not data_args.benchmark:
            trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            trainer.save_state()
        if data_args.benchmark:
            total_effective_tokens, total_tokens = calculate_effective_tokens(
                training_args, train_ds, data_args.max_seq_len
            )
            effective_tokens_per_second = total_effective_tokens / train_result.metrics["train_runtime"]
            total_tokens_per_second = total_tokens / train_result.metrics["train_runtime"]
            effective_ratio = 100 * total_effective_tokens / total_tokens
            logger.info(
                "[timelog] {}: {:.2f} % ({}) ".format(
                    "Effective ratio", effective_ratio, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Effective tokens per second", effective_tokens_per_second, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )
            logger.info(
                "[timelog] {}: {:.2f} token/s ({}) ".format(
                    "Tokens per second", total_tokens_per_second, time.strftime("%Y-%m-%d %H:%M:%S")
                )
            )

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)


if __name__ == "__main__":
    main()
