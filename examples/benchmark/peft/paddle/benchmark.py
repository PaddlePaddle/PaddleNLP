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

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import paddle.profiler as profiler
from datasets import load_dataset
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from utils import CustomTrainer, ProfilerCallback

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import InTokensMapDataset
from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, set_seed
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer, GPTForCausalLM

"""
单卡
python benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 --lora \
    --logging_steps 50 --output_dir outputs

多卡mp
python -m paddle.distributed.launch --gpus "0,1,2,3" benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 8 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 --tensor_parallel_degree 4 \
    --logging_steps 50 --output_dir outputs

多卡sharding 3
python -m paddle.distributed.launch --gpus "0,1,2,3" benchmark.py --model_name_or_path bigscience/bloomz-7b1-mt  \
    --num_train_epochs 1 --per_device_train_batch_size 4 \
    --evaluation_strategy no --save_strategy no \
    --fp16 --fp16_opt_level O2 \
    --sharding "stage3" --sharding_parallel_degree 4 \
    --logging_steps 50 --output_dir outputs
"""


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str = field(default=None, metadata={"help": "model name or local path"})
    lora: Optional[bool] = field(default=False, metadata={"help": "whether to use LoRA"})
    english: Optional[bool] = field(default=False, metadata={"help": "whether to english benchmark dataset"})
    profiler: Optional[bool] = field(default=False, metadata={"help": "whether to use profiler"})
    train_data_size: int = field(default=1000, metadata={"help": "Number of dataset for training"})
    intokens_length: int = field(default=2048, metadata={"help": "Intokens length"})
    intokens: Optional[bool] = field(default=False, metadata={"help": "whether to use intokens"})
    use_flash_attention: bool = field(default=False, metadata={"help": "Whether to use flash attention"})


def main():
    parser = PdArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(args=training_args)
    # Set the dtype for loading model
    dtype = None
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"
        if training_args.bf16:
            dtype = "bfloat16"
    else:
        dtype = "float32"
    if model_args.model_name_or_path in ["gpt3-6.7B-en", "gpt3-13B-en"]:
        tokenizer = AutoTokenizer.from_pretrained("gpt3-13B-en")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if "llama" in model_args.model_name_or_path or "Baichuan" in model_args.model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.model_name_or_path in ["gpt3-6.7B-en", "gpt3-13B-en"]:
        model = GPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention=model_args.use_flash_attention,
            dtype=dtype,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
        )
        tracker = get_rng_state_tracker()
        tracker.add("global_seed", 111)
        tracker.add("local_seed", 222)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            use_flash_attention=model_args.use_flash_attention,
            dtype=dtype,
            tensor_parallel_degree=training_args.tensor_parallel_degree,
            tensor_parallel_rank=training_args.tensor_parallel_rank,
        )

    if model_args.lora:
        if "llama" in model_args.model_name_or_path or "Baichuan" in model_args.model_name_or_path:
            target_modules = [".*q_proj.*", ".*k_proj.*", ".*v_proj.*"]
        elif model_args.model_name_or_path in ["gpt3-6.7B-en", "gpt3-13B-en"]:
            target_modules = [
                ".*qkv_proj.*",
                ".*q_proj.*",
                ".*k_proj.*",
                ".*v_proj.*",
                ".*linear1.*",
                ".*linear2.*",
                ".*out_proj.*",
            ]
        elif "chatglm2" in model_args.model_name_or_path:
            target_modules = [
                ".*query.*",
                ".*key.*",
                ".*value.*",
                ".*dense.*",
                ".*dense_h_to_4h.*",
                ".*dense_4h_to_h.*",
            ]
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

    def preprocess_function(example, max_src_length=256, max_tgt_length=384, intokens=False):
        inputs = example["instruction"]
        targets = example["output"]
        if "input" in example:
            inputs += example["input"]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        # shift input and labels
        model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
        model_inputs["labels"] = model_inputs["labels"][1:]
        seq_length = len(model_inputs["input_ids"])
        model_inputs["position_ids"] = list(range(seq_length))
        if intokens:
            model_inputs["attention_mask"] = np.tril(np.ones([seq_length, seq_length], dtype=bool))
        return model_inputs

    def preprocess_function_chatglm(example, max_src_length=256, max_tgt_length=384, intokens=False):
        inputs = example["instruction"]
        targets = example["output"]
        if "input" in example:
            inputs += example["input"]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        # shift input and labels
        model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
        model_inputs["labels"] = model_inputs["labels"][1:]

        if intokens:
            context_length = model_inputs["input_ids"].index(tokenizer.bos_token_id)
            seq_length = len(model_inputs["input_ids"])
            position_ids = np.arange(seq_length, dtype=np.int64)
            block_position_ids = np.concatenate(
                [
                    np.zeros(context_length, dtype=np.int64),
                    np.arange(1, seq_length - context_length + 1, dtype=np.int64),
                ]
            )
            model_inputs["position_ids"] = np.stack([position_ids, block_position_ids], axis=0)
            attention_mask = np.tri(seq_length, seq_length, dtype=bool)
            attention_mask[:, :context_length] = 1
            model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def preprocess_function_bloom(example, max_src_length=256, max_tgt_length=384, intokens=False):
        inputs = example["instruction"]
        targets = example["output"]
        if "input" in example:
            inputs += example["input"]
        model_inputs = tokenizer(inputs, max_length=max_src_length, truncation=True, return_attention_mask=False)
        labels = tokenizer(targets, max_length=max_tgt_length, truncation=True, return_attention_mask=False)
        labels_input_ids = labels["input_ids"] + [tokenizer.eos_token_id]
        model_inputs["labels"] = [-100] * len(model_inputs["input_ids"]) + labels_input_ids
        model_inputs["input_ids"] = model_inputs["input_ids"] + labels_input_ids
        # shift input and labels
        model_inputs["input_ids"] = model_inputs["input_ids"][:-1]
        model_inputs["labels"] = model_inputs["labels"][1:]

        if intokens:
            model_inputs["attention_mask"] = np.tril(
                np.ones([len(model_inputs["input_ids"]), len(model_inputs["input_ids"])], dtype=bool)
            )
        return model_inputs

    def preprocess_function_gpt(example, max_source_length=256, max_target_length=384, intokens=False):
        """
        Convert an example into necessary features.
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        # NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
        # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
        inputs = example["instruction"]
        targets = example["output"]
        if "input" in example:
            inputs += example["input"]

        input_seq = inputs
        output_seq = targets

        outputs = tokenizer(
            output_seq,
            max_length=max_target_length,
            # pad_to_max_seq_len=True,
            truncation_strategy="longest_first",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        inputs = tokenizer(
            input_seq,
            max_length=max_source_length,
            # pad_to_max_seq_len=True,
            truncation_strategy="longest_first",
            return_attention_mask=False,
            return_length=False,
        )

        final = {}
        for k in outputs.keys():
            final[k] = inputs[k] + outputs[k]
            if k == "input_ids":
                final["labels"] = [tokenizer.pad_token_id] * len(inputs["input_ids"]) + outputs[k]

        # shift inputs and labels
        final["input_ids"] = final["input_ids"][:-1]
        final["labels"] = final["labels"][1:]
        return final

    if model_args.english:
        dataset = load_dataset("tatsu-lab/alpaca")
    else:
        dataset = load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")

    # select first 10k examples for benchmarking
    dataset = dataset["train"].select(range(model_args.train_data_size))
    if "chatglm2" in model_args.model_name_or_path:
        dataset = dataset.map(
            lambda example: preprocess_function(example, intokens=model_args.intokens),
        )
    elif "chatglm" in model_args.model_name_or_path:
        dataset = dataset.map(
            lambda example: preprocess_function_chatglm(example, intokens=model_args.intokens),
        )
    elif "bloom" in model_args.model_name_or_path:

        dataset = dataset.map(
            lambda example: preprocess_function_bloom(example, intokens=model_args.intokens),
        )
    elif model_args.model_name_or_path in ["gpt3-6.7B-en", "gpt3-13B-en"]:
        dataset = dataset.map(
            lambda example: preprocess_function_gpt(example, intokens=model_args.intokens),
        )
    else:
        dataset = dataset.map(lambda example: preprocess_function(example, intokens=model_args.intokens))
    total_effective_tokens = sum([len(i["input_ids"]) for i in dataset]) * training_args.num_train_epochs
    if model_args.intokens:
        dataset = InTokensMapDataset(
            dataset,
            tokenizer=tokenizer,
            max_length=model_args.intokens_length,
        )
    if model_args.profiler:
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            profile_memory=True,
            scheduler=profiler.make_scheduler(closed=1, ready=2, record=1, repeat=1),
            on_trace_ready=profiler.export_chrome_tracing("./log"),
        )
    if model_args.model_name_or_path in ["gpt3-6.7B-en", "gpt3-13B-en"]:
        data_collator = DataCollatorForSeq2Seq(
            return_tensors="pd", tokenizer=tokenizer, label_pad_token_id=tokenizer.pad_token_id
        )
    else:
        data_collator = DataCollatorForSeq2Seq(return_tensors="pd", tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=[ProfilerCallback(prof)] if model_args.profiler else [],
        args=training_args,
        data_collator=data_collator,
    )

    train_metrics = trainer.train()
    tokens_per_second = trainer.total_observed_tokens / train_metrics.metrics["train_runtime"]
    effective_tokens_per_second = total_effective_tokens / train_metrics.metrics["train_runtime"]
    print(f"Tokens per second: {tokens_per_second:.2f}")
    print(f"Effective Tokens per second: {effective_tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
