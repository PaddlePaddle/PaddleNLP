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
from data import cnn_dm_convert_example
from utils import GLMTrainer, generate

from paddlenlp.data import DefaultDataCollator
from paddlenlp.datasets import load_dataset
from paddlenlp.layers import LoRAConfig, get_lora_model, mark_only_lora_as_trainable
from paddlenlp.metrics import Rouge1, Rouge2, RougeL
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from paddlenlp.transformers import AutoModelForConditionalGeneration, AutoTokenizer


@dataclass
class DataArgument:
    task_name: str = field(default="cnn_dailymail", metadata={"help": "The name of task."})
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

    # Load the pretrained language model.
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, output_predict=True, parallel_output=True, load_state_as_np=True
    )
    if model_args.lora:
        # TODO: hardcode parameters for now. Change after MergedLoRA is introduced
        lora_config = LoRAConfig(
            target_modules=[".*query_key_value.*"],
            r=4,
            lora_alpha=8,
            merge_weights=True,
        )
        model = get_lora_model(model, lora_config)
        mark_only_lora_as_trainable(model)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model.generate = partial(
        generate,
        self=model,
        tgt_length=data_args.tgt_length,
        min_tgt_length=data_args.min_tgt_length,
        num_beams=data_args.num_beams,
        length_penalty=data_args.length_penalty,
        no_repeat_ngram_size=data_args.no_repeat_ngram_size,
        end_token_id=tokenizer.eop_token_id,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.smask_token_id,
        no_block_position=data_args.no_block_position,
        select_topk=data_args.select_topk,
        top_k=data_args.top_k,
        top_p=data_args.top_p,
    )

    # Load the dataset.
    train_ds, dev_ds, test_ds = load_dataset(data_args.task_name, splits=["train", "dev", "test"])
    trans_func = partial(cnn_dm_convert_example, tokenizer=tokenizer, data_args=data_args)
    train_ds = train_ds.map(partial(trans_func, is_test=False))
    dev_ds = dev_ds.map(trans_func)
    test_ds = test_ds.map(trans_func)

    collate_fn = DefaultDataCollator()

    def compute_metrics(eval_preds):
        rouge1 = Rouge1()
        rouge2 = Rouge2()
        rougel = RougeL()
        predictions = [x[x != -100] for x in eval_preds.predictions]
        references = [x[x != -100] for x in eval_preds.label_ids]

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
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_result = trainer.evaluate(test_ds)
        trainer.log_metrics("test", eval_result)

    if training_args.do_export:
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path=export_path)


if __name__ == "__main__":
    main()
