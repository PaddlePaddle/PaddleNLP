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

import copy
import json
import os
import sys
from dataclasses import dataclass, field
from functools import partial

import paddle
from benchmark_utils import (
    LlamaTrainer,
    compute_metrics,
    compute_metrics_not_do_generation,
)

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import llama_postprocess_past_key_value
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.transformers import (
    AutoModelForCausalLM,
    AutoModelForCausalLMPipe,
    AutoTokenizer,
)
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    data_name: str = field(default=None, metadata={"help": "The name of data."})
    task_name_or_path: str = field(default=None, metadata={"help": "The name of task."})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=256, metadata={"help": "The max length of target text."})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="facebook/llama-7b", metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "The label smoothing parameter."})
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    eval_with_do_generation: bool = field(
        default=False, metadata={"help": "Evaluate with generation, instead for calc loss."}
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
    # prefix
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=10, metadata={"help": "Number of prefix tokens"})
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    # qat
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})
    qat_type: str = field(default="A8W8", metadata={"help": "Quantization type. Supported values: A8W8, W4,A8W4"})


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def read_local_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            json_line = json.loads(line)
            yield json_line


def custom_instruction_convert_example(example, tokenizer, data_args, is_test=False, model_max_length=512):
    """
    Convert an example into necessary features.
    """

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

    if example.get("input", "") != "":
        input_seq = prompt_input.format_map(example)
    else:
        input_seq = prompt_no_input.format_map(example)

    output_seq = example["output"] + tokenizer.eos_token

    # To compatible with compile training mode in benchmark, input will be pad to fix length
    source_tokenized = tokenizer(
        input_seq,
        return_tensors="pd",
        max_length=model_max_length,
        truncation=True,
    )

    source_input_ids_len = (
        source_tokenized["input_ids"].not_equal(paddle.to_tensor(tokenizer.pad_token_id)).sum().item()
    )

    example_tokenized = tokenizer(
        input_seq + output_seq,
        return_tensors="pd",
        max_length=model_max_length,
        truncation=True,
    )

    input_ids = example_tokenized["input_ids"][0]
    labels = copy.deepcopy(input_ids)
    labels[:source_input_ids_len] = -100

    if is_test:
        return dict(
            input_ids=source_tokenized["input_ids"][0],
            labels=labels,
        )

    # shift labels
    input_ids, labels = input_ids[:-1], labels[1:]

    return dict(
        input_ids=input_ids,
        labels=labels,
    )


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.always_pad_to_max_length = training_args.pipeline_parallel_degree > 1

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.tgt_length = data_args.tgt_length

    training_args.profiler_options = model_args.profiler_options
    setattr(training_args, "label_smoothing", model_args.label_smoothing)
    setattr(training_args, "lr_decay_ratio", model_args.lr_decay_ratio)

    paddle.set_device(training_args.device)

    set_seed(seed=training_args.seed)

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

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        if model_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")
        model_class = AutoModelForCausalLMPipe

    # Load the pretrained language model.
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        tensor_parallel_output=False,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        use_flash_attention=model_args.use_flash_attention,
        dtype=dtype,  # todo enable set dtype to avoid additional mem usage
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

    if model_args.qat:
        from paddle import nn
        from paddle.quantization import QAT, QuantConfig
        from paddle.quantization.quanters.abs_max import (
            FakeQuanterWithAbsMaxObserverLayer,
        )
        from paddleslim.quant.quanters import (
            FakeQuanterChannelWiseAbsMaxObserver,
            PACTQuanter,
        )

        from paddlenlp.peft.lora import LoRALinear
        from paddlenlp.peft.lora.lora_quant_layers import QuantedLoRALinear

        q_config = QuantConfig(activation=None, weight=None)
        q_config.add_qat_layer_mapping(LoRALinear, QuantedLoRALinear)

        if model_args.qat_type == "A8W8":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            # activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=8, dtype="float32")
        elif model_args.qat_type == "W4":
            activation = None
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
        elif model_args.qat_type == "A8W4":
            activation = PACTQuanter(quanter=FakeQuanterWithAbsMaxObserverLayer, init_value=20, dtype=dtype)
            # activation = FakeQuanterWithAbsMaxObserver(moving_rate=0.9, bit_length=8, dtype=dtype)
            weight = FakeQuanterChannelWiseAbsMaxObserver(bit_length=4, dtype="float32")
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
            postprocess_past_key_value=llama_postprocess_past_key_value,
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",  # Allow batch inference
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load the dataset.
    train_ds = load_dataset(read_local_dataset, path="./data/train.txt", lazy=False)
    training_args.do_eval = False
    data_args.always_pad_to_max_length = True
    trans_func = partial(custom_instruction_convert_example, tokenizer=tokenizer, data_args=data_args)

    train_ds = train_ds.map(partial(trans_func))

    model_max_length = 512
    collate_fn = DataCollatorForSeq2Seq(
        return_tensors="pd",
        tokenizer=tokenizer,
        max_length=model_max_length if data_args.always_pad_to_max_length else -1,
        padding="max_length" if data_args.always_pad_to_max_length else True,
        max_label_length=model_max_length if data_args.always_pad_to_max_length else None,
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

    trainer = LlamaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
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
