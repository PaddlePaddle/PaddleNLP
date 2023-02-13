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

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import paddle
from dataset import DataCollatorForErnieHealth, MedicalCorpus

from paddlenlp.trainer import (
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import (
    ElectraConfig,
    ElectraTokenizer,
    ErnieHealthForTotalPretraining,
    ErnieHealthPretrainingCriterion,
)
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "ernie-health": (ElectraConfig, ErnieHealthForTotalPretraining, ErnieHealthPretrainingCriterion, ElectraTokenizer),
}


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluating.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    input_dir: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    masked_lm_prob: float = field(
        default=0.15,
        metadata={"help": "Mask token prob."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to pre-train from.
    """

    model_type: Optional[str] = field(
        default="ernie-health", metadata={"help": "Only support for ernie-health pre-training for now."}
    )
    model_name_or_path: str = field(
        default="ernie-health-chinese",
        metadata={
            "help": "Path to pretrained model or model identifier from https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html"
        },
    )


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.eval_iters = 10
    training_args.test_iters = training_args.eval_iters * 10
    # training_args.recompute = True

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

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

    config_class, model_class, criterion_class, tokenizer_class = MODEL_CLASSES["ernie-health"]

    # Loads or initialize a model.
    pretrained_models = list(tokenizer_class.pretrained_init_configuration.keys())
    pretrained_models.append("__internal_testing__/ernie-health-chinese")

    if model_args.model_name_or_path in pretrained_models:
        if model_args.model_name_or_path == "__internal_testing__/ernie-health-chinese":
            tokenizer = ElectraTokenizer.from_pretrained("__internal_testing__/ernie-health-chinese")
        else:
            tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)

        model_config = config_class()
        model = model_class(model_config)
    else:
        raise ValueError("Only support %s" % (", ".join(pretrained_models)))

    # Loads dataset.
    tic_load_data = time.time()
    logger.info("start load data : %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    train_dataset = MedicalCorpus(data_path=data_args.input_dir, tokenizer=tokenizer)
    logger.info("load data done, total : %s s" % (time.time() - tic_load_data))

    # Reads data and generates mini-batches.
    data_collator = DataCollatorForErnieHealth(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
        mlm_prob=data_args.masked_lm_prob,
        return_dict=True,
    )

    class CriterionWrapper(paddle.nn.Layer):
        """ """

        def __init__(self):
            """CriterionWrapper"""
            super(CriterionWrapper, self).__init__()
            model_config.gen_weight = model.gen_weight
            self.criterion = criterion_class(model_config)

        def forward(self, output, labels):
            """forward function

            Args:
                output (tuple): generator_logits, logits_rtd, logits_mts, logits_csp, disc_labels, mask
                labels (tuple): generator_labels

            Returns:
                Tensor: final loss.
            """
            generator_logits, logits_rtd, logits_mts, logits_csp, disc_labels, masks = output
            generator_labels = labels

            loss, gen_loss, rtd_loss, mts_loss, csp_loss = self.criterion(
                generator_logits, generator_labels, logits_rtd, logits_mts, logits_csp, disc_labels, masks
            )

            return loss

    trainer = Trainer(
        model=model,
        criterion=CriterionWrapper(),
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
