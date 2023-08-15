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
from modeling_pp import GPTForCausalLMPipe
from utils import (
    DataCollatorForSupervisedDataset,
    GPTTrainer,
    compute_metrics,
    convert_example,
)

from paddlenlp.datasets import load_dataset
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import (
    PdArgumentParser,
    TrainingArguments,
    get_last_checkpoint,
    set_seed,
)
from paddlenlp.transformers import AutoTokenizer, GPTConfig, GPTForCausalLM
from paddlenlp.utils.log import logger

MODEL_CLASSES = {
    "gpt": (GPTConfig, GPTForCausalLM),
}


@dataclass
class DataArgument:
    task_name: str = field(default="squad", metadata={"help": "The name of task."})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=142, metadata={"help": "The max length of target text."})
    generate_num: int = field(default=0, metadata={"help": "Save first k examples generation result in dev dataset"})


@dataclass
class ModelArgument:
    model_type: str = field(
        default="gpt-cn", metadata={"help": "Build-in pretrained model from the different model type."}
    )
    model_name_or_path: str = field(
        default="gpt-cpm-large-cn", metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    use_flash_attn: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    enable_fuse_transformer: bool = field(
        default=False,
        metadata={"help": "gpt, enable_fuse_transformer"},
    )

    fuse_attention_qkv: bool = field(
        default=False,
        metadata={"help": "gpt, fuse_attention_qkv"},
    )
    eval_with_do_generation: bool = field(
        default=True, metadata={"help": "Evaluate with generation, instead for calc loss."}
    )
    lr_decay_ratio: float = field(default=0.1, metadata={"help": "The ratio for learning rate decrease"})
    # lora
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # data_args.always_pad_to_max_length = False
    data_args.always_pad_to_max_length = training_args.pipeline_parallel_degree > 1
    setattr(training_args, "lr_decay_ratio", model_args.lr_decay_ratio)

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.tgt_length = data_args.tgt_length
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

    config_class, model_class = MODEL_CLASSES[model_args.model_type]
    if training_args.pipeline_parallel_degree > 1:
        model_class = GPTForCausalLMPipe
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "left"

    # Load and set the pretrained configuration
    config = config_class.from_pretrained(model_args.model_name_or_path)
    config.enable_fuse_transformer = model_args.enable_fuse_transformer
    config.fuse_attention_qkv = model_args.fuse_attention_qkv
    config.use_flash_attn = model_args.use_flash_attn
    config.use_recompute = training_args.recompute

    config.tensor_parallel_degree = training_args.tensor_parallel_degree
    config.tensor_parallel_rank = training_args.tensor_parallel_rank
    config.ignore_index = tokenizer.pad_token_id

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        dtype=dtype,
        load_state_as_np=True,
    )
    if model_args.lora:
        if model_args.lora_path is None:
            target_modules = [
                ".*qkv_proj.*",
                ".*q_proj.*",
                ".*k_proj.*",
                ".*v_proj.*",
                ".*linear1.*",
                ".*linear2.*",
                ".*out_proj.*",
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

    # Load the dataset.
    if training_args.do_train or training_args.do_eval:
        train_ds, dev_ds = load_dataset(data_args.task_name, splits=["train_v1", "dev_v1"])
        trans_func = partial(
            convert_example,
            tokenizer=tokenizer,
            max_source_length=data_args.src_length,
            max_target_length=data_args.tgt_length,
        )

    if training_args.do_train:
        train_ds = train_ds.map(partial(trans_func))
    if training_args.do_eval:
        is_test = model_args.eval_with_do_generation
        dev_ds = dev_ds.map(partial(trans_func, is_test=is_test))

    collate_fn = DataCollatorForSupervisedDataset(
        tokenizer, max_length=1024 if data_args.always_pad_to_max_length else 0
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

    trainer = GPTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=dev_ds if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_func
        if (model_args.eval_with_do_generation and training_args.do_eval)
        else None,
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
