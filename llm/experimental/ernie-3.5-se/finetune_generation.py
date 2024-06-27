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
import sys
from dataclasses import dataclass, field
from functools import partial

import paddle
from data import convert_example, custom_instruction_convert_example, reader
from modeling import Ernie35ForCausalLM
from tokenizer import Ernie35Tokenizer
from utils import Ernie35Trainer, compute_metrics, compute_metrics_not_do_generation

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:

    data_name: str = field(default=None, metadata={"help": "The name of data."})
    task_name: str = field(default=None, metadata={"help": "The name of task."})
    dataset_path: str = field(default=None, metadata={"help": "The file name of train dataset."})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=256, metadata={"help": "The max length of target text."})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="baidu/ernie-3.5-se-3b",
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing parameter."})
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    eval_with_do_generation: bool = field(
        default=False, metadata={"help": "Evaluate with generation, instead for calc loss."}
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: str = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    # lora
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=4, metadata={"help": "Lora attention dimension"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.always_pad_to_max_length = False

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.benchmark = model_args.benchmark
    training_args.tgt_length = data_args.tgt_length

    training_args.profiler_options = model_args.profiler_options
    setattr(training_args, "label_smoothing", model_args.label_smoothing)
    setattr(training_args, "lr_decay_ratio", model_args.lr_decay_ratio)

    paddle.set_device(training_args.device)

    set_seed(args=training_args)

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

    # Set the dtype for loading model
    dtype = "float32"
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    model_class = Ernie35ForCausalLM

    # Load the pretrained language model.
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        use_flash_attention=model_args.use_flash_attention,
        dtype=dtype,  # todo enable set dtype to avoid additional mem usage
        use_progressive_seq_len=False,
        parallel_attn_hatf=True,
    )

    if model_args.lora:
        if model_args.lora_path is None:
            # Not yet support RowParallelLinear
            target_modules = [
                ".*q_proj.*",
                ".*v_proj.*",
                ".*k_proj.*",
                ".*gate_proj.*",
                ".*up_proj.*",
                ".*o_proj.*",
                ".*down_proj.*",
                ".*qkv_gate_up_proj.*",
            ]

            lora_config = LoRAConfig(
                target_modules=target_modules,
                r=model_args.lora_rank,
                lora_alpha=2 * model_args.lora_rank,
                merge_weights=model_args.merge_weights,
                tensor_parallel_degree=training_args.tensor_parallel_degree,
                dtype=dtype,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)

        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    tokenizer = Ernie35Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",  # Allow batch inference
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load the dataset.
    if training_args.benchmark:
        train_ds = load_dataset(reader, data_path="./data/train.txt", lazy=False)
        training_args.do_eval = False
        data_args.always_pad_to_max_length = True
        trans_func = partial(
            custom_instruction_convert_example, tokenizer=tokenizer, data_args=data_args, benchmark=True
        )
    elif training_args.do_train or training_args.do_eval:
        if data_args.data_name is not None:
            if data_args.task_name is not None:
                train_ds, dev_ds = load_dataset(data_args.data_name, data_args.task_name, splits=["train", "dev"])
            else:
                raise ValueError("`data_args.task_name` and `data_args.data_name` should be specified together")
        elif data_args.task_name is not None:
            if data_args.task_name == "squad":
                train_ds, dev_ds = load_dataset("squad", splits=["train_v1", "dev_v1"])
            else:
                train_ds, dev_ds = load_dataset(data_args.task_name, splits=["train", "dev"])
        elif data_args.dataset_path is not None:
            train_ds = load_dataset(reader, data_path=os.path.join(data_args.dataset_path, "train.json"), lazy=False)
            dev_ds = load_dataset(reader, data_path=os.path.join(data_args.dataset_path, "dev.json"), lazy=False)
        else:
            raise ValueError("Please set the correct data arguments(data_name, task_name, dataset_pat)")
        trans_func = partial(convert_example, tokenizer=tokenizer, data_args=data_args)

    if training_args.do_train:
        train_ds = train_ds.map(partial(trans_func))
    if training_args.do_eval:
        # pipeline_parallel eval is the same as training.
        dev_ds = dev_ds.map(partial(trans_func, is_test=model_args.eval_with_do_generation))

    model_max_length = data_args.src_length + data_args.tgt_length if not training_args.benchmark else 512
    collate_fn = DataCollatorForSeq2Seq(
        return_tensors="pd",
        tokenizer=tokenizer,
        max_length=model_max_length if data_args.always_pad_to_max_length else -1,
        padding="max_length" if data_args.always_pad_to_max_length else True,
        return_attention_mask=True,
    )

    def compute_metrics_trainer(eval_preds, tokenizer):
        all_preds = []
        all_labels = []
        preds = eval_preds.predictions
        preds = [x[x != -100] for x in preds]
        all_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        labels = [x[x != -100] for x in eval_preds.label_ids]
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        all_preds = [pred.strip() for pred in all_preds]
        all_labels = [label.strip() for label in all_labels]
        all_preds = [pred.strip("question:") for pred in all_preds]
        all_labels = [label.strip("question:") for label in all_labels]

        eval_result = compute_metrics(all_preds, all_labels)
        return eval_result

    compute_metrics_func = partial(
        compute_metrics_trainer,
        tokenizer=tokenizer,
    )

    trainer = Ernie35Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func
        if model_args.eval_with_do_generation
        else compute_metrics_not_do_generation,
        do_generation=model_args.eval_with_do_generation,
        data_collator=collate_fn,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate()
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":
    main()
