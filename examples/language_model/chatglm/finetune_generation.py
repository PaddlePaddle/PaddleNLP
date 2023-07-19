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

import numpy as np
import paddle
from data import convert_chatglm_example, convert_chatglm_v2_example, read_local_dataset
from sklearn.metrics import accuracy_score
from utils import ChatGLMTrainer

from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import BLEU, Rouge1, Rouge2, RougeL
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.peft.prefix import (
    chatglm_pad_attention_mask,
    chatglm_postprocess_past_key_value,
    chatglm_v2_pad_attention_mask,
)
from paddlenlp.trainer import PdArgumentParser, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
from paddlenlp.utils.log import logger


@dataclass
class DataArgument:
    data_name: str = field(default=None, metadata={"help": "The name of data."})
    task_name_or_path: str = field(default=None, metadata={"help": "Path or name for dataset"})
    src_length: int = field(default=512, metadata={"help": "The max length of source text."})
    tgt_length: int = field(default=512, metadata={"help": "The max length of target text."})
    num_beams: int = field(default=5, metadata={"help": "The number of beams."})


@dataclass
class ModelArgument:
    model_name_or_path: str = field(
        default="THUDM/chatglm-6b", metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    eval_with_do_generation: bool = field(default=False, metadata={"help": "Whether to do generation for evaluation"})
    # lora
    lora: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_path: str = field(default=None, metadata={"help": "Initialize lora state dict."})
    lora_rank: int = field(default=8, metadata={"help": "Lora attention dimension"})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    # prefix
    prefix_tuning: bool = field(default=False, metadata={"help": "Whether to use Prefix technique"})
    num_prefix_tokens: int = field(default=64, metadata={"help": "Number of prefix tokens"})
    prefix_projection: bool = field(default=False, metadata={"help": "Whether to project the prefix tokens"})
    # qat
    qat: bool = field(default=False, metadata={"help": "Whether to use QAT technique"})
    qat_type: str = field(default="A8W8", metadata={"help": "Quantization type. Supported values: A8W8, W4,A8W4"})


def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

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

    dtype = paddle.get_default_dtype()
    if training_args.fp16_opt_level == "O2":
        if training_args.fp16:
            dtype = "float16"

    # Load the pretrained language model.
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        tensor_parallel_degree=training_args.tensor_parallel_degree,
        tensor_parallel_rank=training_args.tensor_parallel_rank,
    )
    if "chatglm2" in model_args.model_name_or_path:
        multi_query_group_num = model.config.multi_query_group_num
        attention_mask_pad_fn = chatglm_v2_pad_attention_mask
    else:
        multi_query_group_num = None
        attention_mask_pad_fn = chatglm_pad_attention_mask
        # If ChatGLM, set lm_shift_labels to False
        model.config.lm_shift_labels = False

    if model_args.prefix_tuning:
        prefix_config = PrefixConfig(
            num_prefix_tokens=model_args.num_prefix_tokens,
            num_attention_heads=model.config.num_attention_heads,
            num_hidden_layers=model.config.num_hidden_layers,
            multi_query_group_num=multi_query_group_num,
            hidden_size=model.config.hidden_size,
            prefix_projection=model_args.prefix_projection,
            prefix_projection_hidden_size=model.config.hidden_size,
            dtype=dtype,
        )
        model = PrefixModelForCausalLM(
            model=model,
            prefix_config=prefix_config,
            postprocess_past_key_value=chatglm_postprocess_past_key_value,
            pad_attention_mask=attention_mask_pad_fn,
        )
        model.mark_only_prefix_as_trainable()
        model.print_trainable_parameters()
    if model_args.lora:
        if model_args.lora_path is None:
            # RowParallelLinear doesn't support LoRA yet
            if "chatglm2" in model_args.model_name_or_path:
                if training_args.tensor_parallel_degree > 1:
                    target_modules = [".*query.*", ".*key.*", ".*value.*", ".*dense_h_to_4h.*"]
                else:
                    target_modules = [
                        ".*query.*",
                        ".*key.*",
                        ".*value.*",
                        ".*dense.*",
                        ".*dense_h_to_4h.*",
                        ".*dense_4h_to_h.*",
                    ]
            else:
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
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )
            model = LoRAModel(model, lora_config)
        else:
            model = LoRAModel.from_pretrained(model=model, lora_path=model_args.lora_path)
        model.mark_only_lora_as_trainable()
        model.print_trainable_parameters()

    if model_args.qat:
        from paddle import nn
        from paddle.quantization import QAT, QuantConfig

        # FakeQuanterChannelWiseAbsMaxObserver not yet merge in Paddle develop
        from paddle.quantization.quanters import FakeQuanterChannelWiseAbsMaxObserver
        from paddle.quantization.quanters.abs_max import (
            FakeQuanterWithAbsMaxObserverLayer,
        )
        from paddleslim.quant.quanters import PACTQuanter

        # from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
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

    convert_example = (
        convert_chatglm_v2_example if "chatglm2" in model_args.model_name_or_path else convert_chatglm_example
    )
    trans_func = partial(convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func, is_test=False))
    dev_ds = dev_ds.map(partial(trans_func, is_test=model_args.eval_with_do_generation))

    collate_fn = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, max_length=data_args.src_length + data_args.tgt_length, padding=True
    )

    def compute_metrics_do_generation(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        bleu4 = BLEU(n_size=4)

        predictions = [x[x != -100].tolist() for x in eval_preds.predictions]
        references = [x[x != -100].tolist() for x in eval_preds.label_ids]
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(references, skip_special_tokens=True)
        # for pred in predictions:
        rouge1_score = rouge1.score(predictions, references)
        rouge2_score = rouge2.score(predictions, references)
        for pred, ref in zip(predictions, references):
            rougel.add_inst(pred, [ref])
            bleu4.add_inst(pred, [ref])
        return {
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougel": rougel.score(),
            "bleu4": bleu4.score(),
        }

    def compute_metrics(eval_preds):
        flattened_preds = np.array(eval_preds.predictions).flatten()
        flattened_labels = np.array(eval_preds.label_ids).flatten()
        filtered_preds = flattened_preds[flattened_labels != -100]
        filtered_labels = flattened_labels[flattened_labels != -100]
        accuracy = accuracy_score(y_true=filtered_labels, y_pred=filtered_preds)
        return {
            "accuracy": accuracy,
        }

    trainer = ChatGLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_do_generation if model_args.eval_with_do_generation else compute_metrics,
        data_collator=collate_fn,
        data_args=data_args,
        do_generation=model_args.eval_with_do_generation,
    )
    # if training_args.fp16_opt_level == "O2":
    #     trainer.disable_autocast_context_manager()

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
    with paddle.amp.auto_cast(enable=False):
        main()
