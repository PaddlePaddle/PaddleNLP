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
from data import convert_example, custom_instruction_convert_example, reader
from modeling_pp import LlamaForCausalLMPipe
from utils import (
    LlamaTrainer,
    compute_metrics,
    compute_metrics_not_do_generation,
    save_infer_result,
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
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    # task_name: str = field(default="school_math_0.25M", metadata={"help": "The name of task."})
    # data_name: str = field(default="bellegroup", metadata={"help": "The name of data."})
    task_name: str = field(default="squad", metadata={"help": "The name of task."})
    src_length: int = field(default=608, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=160, metadata={"help": "The max length of target text."})
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
        default="facebook/llama-7b", metadata={"help": "Build-in pretrained model name or the path to local model."}
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
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})
    eval_with_do_generation: bool = field(
        default=True, metadata={"help": "Evaluate with generation, instead for calc loss."}
    )
    instruction_generation: bool = field(default=False, metadata={"help": "Instruction generation finetuning"})
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: str = field(
        default=None,
        metadata={"help": "profiler_options."},
    )


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.always_pad_to_max_length = training_args.pipeline_parallel_degree > 1

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    training_args.benchmark = model_args.benchmark
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

    model_class = AutoModelForCausalLM
    if training_args.pipeline_parallel_degree > 1:
        if model_args.eval_with_do_generation and training_args.do_eval:
            raise ValueError("Plese set eval_with_do_generation to false in pipeline parallel mode.")
        model_class = LlamaForCausalLMPipe

    # Load the pretrained language model.
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        dtype=dtype,  # todo enable set dtype to avoid additional mem usage
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
        fp16_opt_level=training_args.fp16_opt_level,
        use_flash_attention=model_args.use_flash_attention,
        use_recompute=training_args.recompute,
    )

    if model_args.lora:
        # TODO: hardcode parameters for now. Change after MergedLoRA is introduced
        lora_config = LoRAConfig(
            target_modules=[".*q_proj.*", ".*v_proj.*"],
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
            postprocess_past_key_value=llama_postprocess_past_key_value,
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",  # Allow batch inference
    )
    tokenizer.pad_token = tokenizer.unk_token

    trans_func = partial(custom_instruction_convert_example, tokenizer=tokenizer, data_args=data_args)
    # Load the dataset.
    if training_args.benchmark:
        train_ds = load_dataset(reader, data_path="./data/train.txt", lazy=False)
        training_args.do_eval = False
        data_args.always_pad_to_max_length = True
    elif training_args.do_train or training_args.do_eval:
        if model_args.instruction_generation:
            train_ds, dev_ds = load_dataset("bellegroup", "school_math_0.25M", splits=["train", "dev"])
        else:
            train_ds, dev_ds = load_dataset("squad", splits=["train_v1", "dev_v1"])
            trans_func = partial(convert_example, tokenizer=tokenizer, data_args=data_args)

    if training_args.do_train:
        train_ds = train_ds.map(partial(trans_func))
    if training_args.do_eval:
        # pipeline_parallel eval is the same as training.
        is_test = model_args.eval_with_do_generation
        dev_ds = dev_ds.map(partial(trans_func, is_test=is_test))

    model_max_length = 1024 if not training_args.benchmark else 512
    collate_fn = DataCollatorForSeq2Seq(
        return_tensors="pd",
        tokenizer=tokenizer,
        max_length=model_max_length if data_args.always_pad_to_max_length else -1,
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

    if data_args.generate_num > 0:
        save_infer_result(
            trainer, dev_ds, k=data_args.generate_num, src_length=data_args.src_length, tgt_length=data_args.tgt_length
        )


if __name__ == "__main__":
    main()
