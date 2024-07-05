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

import os

from arguments import DataArguments, ModelArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from data import EmbedCollator, TrainDatasetForEmbedding
from models.modeling import BiEncoderModel

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, Trainer, get_last_checkpoint, set_seed
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set the dtype for loading model
    dtype = None
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"
    else:
        dtype = "float32"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    if training_args.pipeline_parallel_degree > 1 and training_args.negatives_cross_device:
        raise ValueError("Pipeline parallelism does not support cross batch negatives.")
    # Setup logging
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}",
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False,
    )
    tokenizer.padding_side = "right"

    model = BiEncoderModel.from_pretrained(
        model_name_or_path=model_args.model_name_or_path,
        normalized=model_args.normalized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        margin=training_args.margin,
        use_inbatch_neg=training_args.use_inbatch_neg,
        matryoshka_dims=training_args.matryoshka_dims if training_args.use_matryoshka else None,
        matryoshka_loss_weights=training_args.matryoshka_loss_weights if training_args.use_matryoshka else None,
    )

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logger.info(f"Freeze the parameters for {k}")
                v.stop_gradient = True

    if training_args.fine_tune_type == "bitfit":
        for k, v in model.named_parameters():
            # Only bias are allowed for training
            if "bias" in k:
                v.stop_gradient = False
            else:
                logger.info(f"Freeze the parameters for {k} shape: {v.shape}")
                v.stop_gradient = True

    if training_args.fine_tune_type == "lora":
        if "llama" in model_args.model_name_or_path or "baichuan" in model_args.model_name_or_path:
            target_modules = [".*q_proj.*", ".*k_proj.*", ".*v_proj.*"]
        else:
            target_modules = [".*query_key_value.*"]

        lora_config = LoRAConfig(
            target_modules=target_modules,
            r=8,
            lora_alpha=32,
            dtype=dtype,
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    train_dataset = TrainDatasetForEmbedding(
        args=data_args,
        tokenizer=tokenizer,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len,
        is_batch_negative=training_args.use_inbatch_neg,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
        ),
        tokenizer=tokenizer,
    )
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
