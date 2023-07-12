# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import paddle
from sklearn.metrics import accuracy_score
from utils import BloomTrainer, compute_metrics

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import bloom_postprocess_past_key_value
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoTokenizer, BloomForCausalLM
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    data_name: str = field(default=None, metadata={"help": "The name of data."})
    task_name_or_path: str = field(default=None, metadata={"help": "Path or name for dataset"})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=512, metadata={"help": "The max length of target text."})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="bigscience/bloom-560m",
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})

    # lora
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    # prefix tuning
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=10, metadata={"help": "Number of prefix tokens"})
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    # qat
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})
    qat_type: str = field(default="A8W8", metadata={"help": "Quantization type. Supported values: A8W8, W4,A8W4"})


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            yield json.loads(line.strip())


def convert_example(
    example,
    tokenizer,
    max_source_length,
    max_target_length,
    is_test=False,
):
    """Convert all examples into necessary features."""
    if "source" in example:
        source = None
        title = None
        target = None
        if "source" in example and "title" in example:
            source = example["source"]
            if "title" in example.keys():
                title = example["title"]
        elif "context" in example and "answer" in example:
            source = example["context"]
            if "answer" in example.keys():
                title = example["answer"]
        else:
            assert False, "Source and title are not in the input dictionary, nor are context and answer."

        if "target" in example.keys():
            target = example["target"]
        elif "question" in example.keys():
            target = example["question"]

        # Add the eos token for the source and target
        source = "答案：" + title + "，上下文：" + source + "在已知答案的前提下，问题："
    elif "content" in example:
        source = example["content"]
        target = example["summary"]
    elif "instruction" in example:
        source = example["instruction"]
        target = example["output"]
    elif "src" in example:
        source = example["src"][0] if isinstance(example["src"], list) else example["src"]
        target = example["tgt"][0] if isinstance(example["tgt"], list) else example["tgt"]
    else:
        raise ValueError("Please check dataset format.")

    target = target[: max_target_length - 1]
    target = target + tokenizer.eos_token

    target_tokenized = tokenizer(
        target,
        max_length=max_target_length,
        truncation=True,
    )

    source_tokenized = tokenizer(
        source,
        max_length=max_source_length,
        truncation=True,
    )
    if is_test:
        return dict(
            input_ids=source_tokenized["input_ids"],
            labels=target_tokenized["input_ids"],
        )
    else:
        input_ids = source_tokenized["input_ids"] + target_tokenized["input_ids"]
        labels = len(source_tokenized["input_ids"]) * [tokenizer.pad_token_id] + target_tokenized["input_ids"]

        # shift labels
        input_ids, labels = input_ids[:-1], labels[1:]
        return dict(
            input_ids=input_ids,
            labels=labels,
        )


def main():
    # Parse the model and data  arguements
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    setattr(training_args, "lr_decay_ratio", model_args.lr_decay_ratio)

    # Set the training device
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

    # Set the dtype for loading model
    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    # Load the pretrained language model.
    model = BloomForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,  # todo enable low_cpu_mem_usage=True
        dtype=dtype,  # todo enable set dtype to avoid additional mem usage
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        lm_shift_labels=False,
        use_recompute=training_args.recompute,
    )

    if model_args.lora:
        if model_args.lora_path is None:
            # Not yet support RowParallelLinear
            if training_args.tensor_parallel_degree > 1:
                target_modules = [".*query_key_value.*", ".*dense_h_to_4h.*"]
            else:
                target_modules = [".*query_key_value.*", ".*dense.*", ".*dense_h_to_4h.*", ".*dense_4h_to_h.*"]
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

    if model_args.qat:
        from paddle import nn
        from paddle.quantization import QAT, QuantConfig

        # from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
        # FakeQuanterChannelWiseAbsMaxObserver not yet merge in Paddle develop
        from paddle.quantization.quanters import FakeQuanterChannelWiseAbsMaxObserver
        from paddle.quantization.quanters.abs_max import (
            FakeQuanterWithAbsMaxObserverLayer,
        )
        from paddleslim.quant.quanters import PACTQuanter

        from paddlenlp.peft.lora import LoRALinear
        from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear

        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)

        if model_args.qat_type == "A8W8":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            # activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype="float32")
            # weight = FakeQuanterWithAbsMaxObserver(bit_length=8, dtype=dtype)
        elif model_args.qat_type == "W4":
            activation = None
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
            # weight = FakeQuanterWithAbsMaxObserver(bit_length=4, dtype=dtype)
        elif model_args.qat_type == "A8W4":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            # activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
            # weight = FakeQuanterWithAbsMaxObserver(bit_length=8, dtype=dtype)
        else:
            raise ValueError("qat_type should be one of ['A8W8', 'W4', 'A8W4']")

        q_config.add_type_config(LoRALinear, weight=weight, activation=activation)
        q_config.add_type_config(nn.Linear, weight=weight, activation=activation)

        qat = QAT(q_config)
        model = qat.quantize(model, inplace=True)

    if model_args.prefix_tuning:
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=model.config.n_head,
            num_hidden_layers=model.config.n_layer,
            hidden_size=model.config.hidden_size,
            prefix_projection=model_args.prefix_projection,
            prefix_projection_hidden_size=model.config.hidden_size,
            dtype=dtype,
        )
        model = PrefixModelForCausalLM(
            model=model,
            prefix_config=prefix_config,
            postprocess_past_key_value=bloom_postprocess_past_key_value,
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    # Load the Tokenzier
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load the dataset.
    if os.path.exists(os.path.join(data_args.task_name_or_path, "train.json")) and os.path.exists(
        os.path.join(data_args.task_name_or_path, "dev.json")
    ):
        train_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.task_name_or_path, "train.json"), lazy=False
        )
        dev_ds = load_dataset(
            read_local_dataset, path=os.path.join(data_args.task_name_or_path, "dev.json"), lazy=False
        )
    elif data_args.data_name is not None:
        train_ds, dev_ds = load_dataset(data_args.data_name, data_args.task_name_or_path, splits=["train", "dev"])
    else:
        train_ds, dev_ds = load_dataset(data_args.task_name_or_path, splits=["train", "dev"])

    # Load the dataset.
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_source_length=data_args.src_length,
        max_target_length=data_args.tgt_length,
    )

    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(partial(trans_func, is_test=model_args.eval_with_do_generation), lazy=False)

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.src_length + data_args.tgt_length,
        label_pad_token_id=tokenizer.pad_token_id,
        return_tensors="np",
    )

    @paddle.no_grad()
    def compute_metrics_trainer(eval_preds, tokenizer):
        all_preds = []
        all_labels = []
        preds = [x[x != -100] for x in eval_preds.predictions]
        all_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        labels = [x[x != -100] for x in eval_preds.label_ids]
        all_labels.extend(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        eval_result = compute_metrics(all_preds, all_labels)
        return eval_result

    compute_metrics_func = partial(
        compute_metrics_trainer,
        tokenizer=tokenizer,
    )

    def compute_metrics_not_do_generation(eval_preds):
        flattened_preds = np.array(eval_preds.predictions).flatten()
        flattened_labels = np.array(eval_preds.label_ids).flatten()
        cleaned_labels = [True if x != -100 and x != tokenizer.pad_token_id else False for x in flattened_labels]
        filtered_preds = flattened_preds[cleaned_labels]
        filtered_labels = flattened_labels[cleaned_labels]
        accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
        return {
            "accuracy": accuracy,
        }

    trainer = BloomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func
        if model_args.eval_with_do_generation
        else compute_metrics_not_do_generation,
        do_generation=model_args.eval_with_do_generation,
        data_collator=collate_fn,
        data_args=data_args,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("test", eval_result)


if __name__ == "__main__":
    main()
