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
from utils import DataArguments, ModelArguments, load_config, seq_convert_example

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import CompressionArguments, PdArgumentParser, Trainer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    model_args, data_args, compression_args = load_config(
        model_args.config, "SequenceClassification", data_args.dataset, model_args, data_args, compression_args
    )

    paddle.set_device(compression_args.device)

    data_args.dataset = data_args.dataset.strip()

    # Log model and data config
    compression_args.print_config(model_args, "Model")
    compression_args.print_config(data_args, "Data")

    raw_datasets = load_dataset("clue", data_args.dataset)

    data_args.label_list = getattr(raw_datasets["train"], "label_list", None)
    num_classes = len(raw_datasets["train"].label_list)

    criterion = paddle.nn.CrossEntropyLoss()
    # Defines tokenizer, model, loss function.
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    model = ErnieForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)

    # Defines dataset pre-process function
    trans_fn = partial(
        seq_convert_example, tokenizer=tokenizer, label_list=data_args.label_list, max_seq_len=data_args.max_seq_length
    )

    # Defines data collector
    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataset = raw_datasets["train"].map(trans_fn)
    eval_dataset = raw_datasets["dev"].map(trans_fn)

    trainer = Trainer(
        model=model,
        args=compression_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        criterion=criterion,
    )  # Strategy`dynabert` needs arguments `criterion`

    compression_args.print_config()

    trainer.compress()


if __name__ == "__main__":
    main()
