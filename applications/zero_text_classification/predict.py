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
from paddle.metric import Accuracy
from utils import read_local_dataset

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
    test_path: str = field(default="test.txt", metadata={"help": "Test dataset file name."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-1.0-large-zh-cw", metadata={"help": "Build-in pretrained model."})
    model_state: str = field(default=None, metadata={"help": "Build-in pretrained model."})


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
    template = UTCTemplate(prompt, tokenizer, training_args.max_seq_length, max_position_id=511)

    # Load and preprocess dataset.
    test_ds = load_dataset(read_local_dataset, data_path=data_args.test_path, lazy=False)

    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )
    if model_args.model_state is not None:
        model_state = paddle.load(model_args.model_state)
        prompt_model.set_state_dict(model_state)

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = Accuracy()
        print(eval_preds.predictions.shape)
        print(eval_preds.label_ids.shape)

        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)
        print("preds", preds)
        print("labels", labels)
        preds = paddle.nn.functional.softmax(preds, axis=1)
        labels = paddle.argmax(labels, axis=1)
        print("preds", preds)
        print("labels", labels)
        correct = metric.compute(preds, labels)
        metric.update(correct)
        acc = metric.accumulate()
        return {"acc": acc}

    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        criterion=criterion,
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        compute_metrics=compute_metrics,
    )

    test_ret = trainer.predict(test_ds)
    trainer.log_metrics("test", test_ret.metrics)


if __name__ == "__main__":
    main()
