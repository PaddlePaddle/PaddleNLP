# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Optional

import paddle
from paddle.static import InputSpec
from utils import PromptTrainerForGeneration, compute_metrics

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PrefixTemplate,
    PromptModelForGeneration,
    PromptTuningArguments,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import AutoTokenizer, GPTLMHeadModel
from paddlenlp.utils.log import logger


@dataclass
class DataArguments:
    prompt: str = field(
        default="{'prefix':'文本摘要'}{'text':'text'}{'sep'}{'text':'labels', 'token_type': 1}",
        metadata={"help": "Add prompt.'文本摘要'、'text' variable and 'labels' immutable."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="gpt-cpm-small-cn-distill",
        metadata={"help": "Build-in pretrained model name or the path to local model."},
    )
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})
    dropout: float = field(default=0.1, metadata={"help": "The dropout used for pretrained model."})
    predict_with_generate: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to generate in predcit.")},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": ("The number of beams to use in beam search.")},
    )
    max_target_length: Optional[int] = field(
        default=64,
        metadata={
            "help": (
                "The maximum total sequence length for target text after "
                "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
                "during ``evaluate`` and ``predict``."
            )
        },
    )


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.generation_max_length = model_args.max_target_length
    training_args.predict_with_generate = model_args.predict_with_generate
    training_args.generation_num_beams = model_args.num_beams

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = "<pad>"
    tokenizer.sep_token = "<sep>"
    tokenizer.add_tokens("[Space]", special_tokens=True)
    model = GPTLMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        hidden_dropout_prob=model_args.dropout,
        attention_probs_dropout_prob=model_args.dropout,
    )

    # Define template for preprocess.
    template = PrefixTemplate(data_args.prompt, tokenizer, training_args.max_seq_length, model)
    logger.info("Using template: {}".format(template.prompt))

    # Load datasets.
    train_ds, dev_ds = load_dataset("lcsts_new")

    def convert_label_keyword(input_dict):
        if "text" not in input_dict:
            input_dict["text"] = input_dict.pop("source")
        if "labels" not in input_dict:
            input_dict["labels"] = input_dict.pop("target")
        return input_dict

    train_ds.map(convert_label_keyword)
    dev_ds.map(convert_label_keyword)

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForGeneration(
        model,
        template,
        freeze_plm=training_args.freeze_plm,
        freeze_dropout=training_args.freeze_dropout,
        max_predict_len=training_args.generation_max_length,
    )

    dev_compute_metrics = partial(compute_metrics, tokenizer=tokenizer)
    trainer = PromptTrainerForGeneration(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=dev_compute_metrics,
    )

    # Traininig.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # Export static model.
    if training_args.do_export:
        template = prompt_model.template
        template_keywords = template.extract_template_keywords(template.prompt)
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64"),  # input_ids,
            InputSpec(shape=[None, None], dtype="int64"),  # token_type_ids
            InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            InputSpec(shape=[None, None, None, None], dtype="float32"),  # attention_mask
            InputSpec(shape=[None], dtype="int64"),  # masked_positions
            InputSpec(shape=[None, None], dtype="int64"),  # soft_token_ids
        ]
        if "encoder" in template_keywords:
            input_spec.append(InputSpec(shape=[None, None], dtype="int64"))  # encoder_ids
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, input_spec=input_spec, export_type=model_args.export_type)


if __name__ == "__main__":
    main()
