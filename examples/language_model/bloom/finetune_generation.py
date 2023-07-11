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

import os
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import paddle
from sklearn.metrics import accuracy_score
from utils import BloomTrainer, compute_metrics, save_infer_result

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import bloom_postprocess_past_key_value
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoTokenizer, BloomForCausalLM
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    task_name: str = field(default="dureader_qg", metadata={"help": "The name of task."})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=512, metadata={"help": "The max length of target text."})
    min_tgt_length: int = field(default=0, metadata={"help": "The min length of target text."})
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
    generate_num: int = field(default=0, metadata={"help": "Save first k examples generation result in dev dataset"})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="bigscience/bloom-560m",
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing parameter."})
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    r: int = field(default=4, metadata={"help": "Lora attention dimension"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=10, metadata={"help": "Number of prefix tokens"})
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})


def convert_example(
    example,
    tokenizer,
    decoder_start_token_id,
    max_source_length,
    max_target_length,
    is_train=True,
):
    """Convert all examples into necessary features."""
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
    target = target[: max_target_length - 1]
    target = target + tokenizer.eos_token

    target_tokenized = tokenizer(
        target,
        max_length=max_target_length,
        truncation=True,
    )
    target_input_ids_len = (np.array(target_tokenized["input_ids"]) != tokenizer.pad_token_id).sum()

    source_tokenized = tokenizer(
        source,
        max_length=(max_source_length + max_target_length - target_input_ids_len - 1),
        padding="max_length",
        truncation=True,
    )

    input_ids = source_tokenized["input_ids"] + [tokenizer.eos_token_id] + target_tokenized["input_ids"]
    labels = (len(input_ids) - target_input_ids_len) * [tokenizer.pad_token_id] + target_tokenized["input_ids"]

    # shift labels
    input_ids, labels = input_ids[:-1], labels[1:]

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def custom_instruction_convert_example(example, tokenizer, data_args, is_test=True):
    instruction = ""
    input = ""
    response = ""
    if "instruction" in example and "output" in example:
        instruction = example["instruction"]
        response = example["output"]
    else:
        assert False, "instruction and output are not in the input dictionary."
    if "input" in example["input"]:
        input = example["input"]

    prompt = instruction + input
    # dataset for evaluation
    if is_test:
        inputs = {
            **tokenizer(prompt, max_length=data_args.src_length, truncation=True, padding="max_length"),
            "labels": tokenizer(response, max_length=data_args.tgt_length, truncation=True, padding="max_length")[
                "input_ids"
            ],
        }
    # dataset for training
    else:
        response = response[: data_args.tgt_length - 1]
        target_tokenized = tokenizer(
            response,
            add_special_tokens=False,
            max_length=data_args.tgt_length - 2,
            truncation=True,
            truncation_side="right",
        )
        target_input_ids_len = (np.array(target_tokenized["input_ids"]) != tokenizer.pad_token_id).sum()

        source_tokenized = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=data_args.src_length - 1,
            truncation=True,
            truncation_side="left",
        )

        input_ids = (
            [tokenizer.bos_token_id]
            + source_tokenized["input_ids"]
            + [tokenizer.bos_token_id]
            + target_tokenized["input_ids"]
        )
        labels = (len(input_ids) - target_input_ids_len) * [tokenizer.pad_token_id] + target_tokenized["input_ids"]

        # shift labels
        input_ids, labels = input_ids[:-1], labels[1:]

        inputs = {
            "input_ids": input_ids,
            "labels": labels,
        }
    return inputs


def main():
    # Parse the model and data  arguements
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    setattr(training_args, "label_smoothing", model_args.label_smoothing)
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
    dtype = None
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"

    # Load the pretrained language model.
    model = BloomForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,  # todo enable low_cpu_mem_usage=True
        dtype=dtype,  # todo enable set dtype to avoid additional mem usage
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        lm_shift_labels=False,
        use_recompute=training_args.recompute,
    )

    if model_args.lora:
        # hardcode parameters for now
        lora_config = LoRAConfig(
            target_modules=[".*query_key_value.*"],
            r=model_args.r,
            lora_alpha=2 * model_args.r,
            merge_weights=model_args.merge_weights,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            dtype=dtype,
        )
        model = LoRAModel(model, lora_config)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

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
    if model_args.do_generation:
        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            decoder_start_token_id=tokenizer.bos_token_id,
            max_source_length=data_args.src_length,
            max_target_length=data_args.tgt_length,
        )
        train_ds, dev_ds = load_dataset("dureader_qg", splits=("train", "dev"))

        train_ds = train_ds.map(trans_func, lazy=False)
        dev_ds = dev_ds.map(trans_func, lazy=False)
    else:
        train_ds, dev_ds = load_dataset("bellegroup", "school_math_0.25M", splits=["train", "dev"])
        trans_func = partial(custom_instruction_convert_example, tokenizer=tokenizer, data_args=data_args)

        train_ds = train_ds.map(partial(trans_func, is_test=False))
        dev_ds = dev_ds.map(partial(trans_func, is_test=False))

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
        compute_metrics=compute_metrics_func if model_args.do_generation else compute_metrics_not_do_generation,
        do_generation=model_args.do_generation,
        data_collator=collate_fn,
        data_args=data_args,
    )

    if training_args.fp16_opt_level == "O2" and not training_args.bf16:
        trainer.disable_autocast_context_manager()

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(merge_tensor_parallel=training_args.tensor_parallel_degree > 1)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(dev_ds)
        trainer.log_metrics("test", eval_result)

    if data_args.generate_num > 0:
        save_infer_result(
            trainer, dev_ds, k=data_args.generate_num, src_length=data_args.src_length, tgt_length=data_args.tgt_length
        )


if __name__ == "__main__":
    main()
