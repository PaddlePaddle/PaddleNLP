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

import logging
import os

from arguments import DataArguments, ModelArguments
from arguments import RetrieverTrainingArguments as TrainingArguments
from data import EmbedCollator, TrainDatasetForEmbedding
from modeling import BiEncoderModel
from utils import BiTrainer

from paddlenlp.trainer import PdArgumentParser, set_seed
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}",
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    # Set seed
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = BiEncoderModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        dtype="bfloat16",
        normalized=model_args.normalized,
        sentence_pooling_method=training_args.sentence_pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        temperature=training_args.temperature,
        use_flash_attention=model_args.use_flash_attention,
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
                logger.info(f"Freeze the parameters for {k}")
                v.stop_gradient = True

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)
    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EmbedCollator(
            tokenizer, query_max_len=data_args.query_max_len, passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
