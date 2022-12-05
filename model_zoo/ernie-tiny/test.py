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
from typing import Optional

import numpy as np
import paddle
from model import JointErnie
from utils import (
    get_label_name,
    input_preprocess,
    intent_cls_postprocess,
    read_test_file,
    slot_cls_postprocess,
)

from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import CompressionArguments, PdArgumentParser
from paddlenlp.transformers import AutoTokenizer


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    test_path: str = field(default=None, metadata={"help": "Test data path. Defaults to None."})
    intent_label_path: str = field(default=None, metadata={"help": "Intent label dict path. Defaults to None."})
    slot_label_path: str = field(default=None, metadata={"help": "Slot label dict path. Defaults to None."})
    max_seq_length: Optional[int] = field(
        default=16,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    ignore_index: Optional[int] = field(default=9999, metadata={"help": ""})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="ernie-3.0-tiny-medium-v2",
        metadata={"help": "Path to pretrained model. Defaults to 'ernie-3.0-tiny-medium-v2'"},
    )
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for JointErnie. Defaults to 0.1."})


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    paddle.set_device(compression_args.device)

    intent_label_names, slot_label_names, _, _ = get_label_name(data_args.intent_label_path, data_args.slot_label_path)

    test_dataset = load_dataset(
        read_test_file,
        filename=data_args.test_path,
        lazy=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-tiny-medium-v2")

    model = JointErnie.from_pretrained(
        model_args.model_name_or_path,
        intent_dim=len(intent_label_names),
        slot_dim=len(slot_label_names),
        dropout=model_args.dropout,
    )

    model.eval()

    for data in test_dataset:
        query_list = [data["query"]]
        query_input_dict = input_preprocess(query_list, tokenizer, max_seq_length=16)
        input_ids = paddle.to_tensor(query_input_dict["input_ids"])
        intent_logits_tensor, slot_logits_tensor = model(input_ids)

        intent_logits = np.array(intent_logits_tensor)
        slot_logits = np.array(slot_logits_tensor)

        intent_out = intent_cls_postprocess(intent_logits, intent_label_names)
        slots_out = slot_cls_postprocess(slot_logits, query_list, slot_label_names)

        print(intent_out, "\n", slots_out)


if __name__ == "__main__":
    main()
