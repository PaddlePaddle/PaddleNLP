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

from functools import partial

import paddle
import paddle.nn as nn
from utils import DataArguments, ModelArguments, load_config, token_convert_example

from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import CompressionArguments, PdArgumentParser, Trainer
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    model_args, data_args, compression_args = load_config(
        model_args.config, "TokenClassification", data_args.dataset, model_args, data_args, compression_args
    )
    paddle.set_device(compression_args.device)

    data_args.dataset = data_args.dataset.strip()

    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    raw_datasets = load_dataset(data_args.dataset)
    label_list = raw_datasets["train"].label_list
    data_args.label_list = label_list
    data_args.ignore_label = -100

    data_args.no_entity_id = 0
    num_classes = len(label_list)

    # Define tokenizer, model, loss function.
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ErnieForTokenClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)

    class criterion(nn.Layer):
        def __init__(self):
            super(criterion, self).__init__()
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=data_args.ignore_label)

        def forward(self, *args, **kwargs):
            return paddle.mean(self.loss_fn(*args, **kwargs))

    loss_fct = criterion()

    # Define dataset pre-process function
    trans_fn = partial(
        token_convert_example,
        tokenizer=tokenizer,
        no_entity_id=data_args.no_entity_id,
        max_seq_length=data_args.max_seq_length,
        return_length=True,
    )

    # Define data collector
    data_collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=data_args.ignore_label)

    # Dataset pre-process
    train_dataset = raw_datasets["train"].map(trans_fn)
    eval_dataset = raw_datasets["test"].map(trans_fn)
    train_dataset.label_list = label_list
    train_dataset.ignore_label = data_args.ignore_label
    trainer = Trainer(
        model=model,
        criterion=loss_fct,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
