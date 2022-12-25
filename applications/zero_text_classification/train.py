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

from dataclasses import dataclass, field

import paddle
from utils import MetricReport, read_local_dataset

from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import (
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    UTCTemplate,
)
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import UTC, AutoTokenizer


@dataclass
class DataArguments:
    dataset_dir: str = field(
        metadata={"help": "Local dataset directory including train.txt, dev.txt and label.txt (optional)."}
    )
    train_file: str = field(default="train.txt", metadata={"help": "Train dataset file name."})
    dev_file: str = field(default="dev.txt", metadata={"help": "Dev dataset file name."})
    shuffle_choices: bool = field(default=False, metadata={"help": "Whether to shuffle choices."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata={"help": "Build-in pretrained model."})


def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)
    omask_dict = {"additional_special_tokens": ["[O-MASK]"]}
    tokenizer.add_special_tokens(omask_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Define template for preprocess and verbalizer for postprocess.
    prompt = (
        "{'text': 'question'}{'sep': None, 'token_type': 1}{'options': 'choices', 'add_omask': True}"
        "{'cls': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
    )
    template = UTCTemplate(prompt, tokenizer, training_args.max_seq_length)  # , max_position_id=511)

    # Load and preprocess dataset.
    train_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_dir,
        data_file=data_args.train_file,
        shuffle_choices=data_args.shuffle_choices,
        lazy=False,
    )
    dev_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_dir,
        data_file=data_args.dev_file,
        shuffle_choices=data_args.shuffle_choices,
        lazy=False,
    )

    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = paddle.nn.functional.sigmoid(paddle.to_tensor(eval_preds.predictions))
        preds = paddle.reshape(preds[preds > 0], [-1])
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        acc = float(((preds > data_args.threshold).astype("int64") == labels).mean().numpy())
        metric.update(preds, labels)
        micro_f1, macro_f1 = metric.accumulate()
        return {"acc": acc, "micro_f1": micro_f1, "macro_f1": macro_f1}

    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=compute_metrics,
    )

    # Training.
    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
