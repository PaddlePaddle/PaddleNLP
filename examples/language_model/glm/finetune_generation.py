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
from dataclasses import dataclass, field
from functools import partial

import paddle
from data import custom_convert_example
from utils import GLMTrainer

from paddlenlp.data import DefaultDataCollator
from paddlenlp.datasets import load_dataset
from paddlenlp.layers import LoRAConfig, LoRAModel
from paddlenlp.metrics import Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    task_name: str = field(default="dureader_qg", metadata={"help": "The name of task."})
    src_length: int = field(default=608, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=160, metadata={"help": "The max length of target text."})
    min_tgt_length: int = field(default=55, metadata={"help": "The min length of target text."})
    length_penalty: float = field(default=0.7, metadata={"help": "The length penalty."})
    no_repeat_ngram_size: int = field(default=3, metadata={"help": "The no repeat ngram size."})
    num_beams: int = field(default=5, metadata={"help": "The number of beams."})
    select_topk: bool = field(default=True, metadata={"help": "Whether to select top k tokens for generation."})
    top_p: float = field(
        default=0.0, metadata={"help": "The cumulative probability for top-p-filtering in the 'sampling' strategy."}
    )
    top_k: int = field(
        default=0,
        metadata={
            "help": "The number of highest probability tokens to keep for top-k-filtering in the 'sampling' strategy."
        },
    )
    no_block_position: bool = field(default=False)


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="THUDM/glm-2b", metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing parameter."})
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    setattr(training_args, "label_smoothing", model_args.label_smoothing)
    setattr(training_args, "lr_decay_ratio", model_args.lr_decay_ratio)

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
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

    # dtype = None
    # if training_args.fp16_opt_level == "O2":
    #     if training_args.fp16:
    #         dtype = "float16"
    #     if training_args.bf16:
    #         dtype = "bfloat16"

    # Load the pretrained language model.
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        output_predict=True,
        parallel_output=True,
        load_state_as_np=True,
        # low_cpu_mem_usage=True, # todo enable low_cpu_mem_usage=True
        # dtype=dtype,  # todo enable set dtype to avoid additional mem usage
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
    )
    if model_args.lora:
        # TODO: hardcode parameters for now. Change after MergedLoRA is introduced
        lora_config = LoRAConfig(
            target_modules=[".*query_key_value.*"],
            r=4,
            lora_alpha=8,
            merge_weights=True,
            enable_lora_list=[[True, False, True]],
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load the dataset.
    train_ds, dev_ds = load_dataset(data_args.task_name, splits=["train", "dev"])
    trans_func = partial(custom_convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func, is_test=False))
    test_ds = dev_ds.map(trans_func)

    collate_fn = DefaultDataCollator()

    def compute_metrics(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        predictions = [x[x != -100] for x in eval_preds.predictions]
        references = [x[x != -100] for x in eval_preds.label_ids]

        # for pred in predictions:

        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
        }

    trainer = GLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        do_generation=True,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(test_ds)
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":
    main()
