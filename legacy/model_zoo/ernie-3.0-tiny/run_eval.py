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
import paddle
from utils import (
    get_label_name,
    input_preprocess,
    intent_cls_postprocess,
    read_example,
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
    ignore_index: Optional[int] = field(default=0, metadata={"help": ""})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="ernie-3.0-tiny-nano-v2-zh",
        metadata={"help": "Path to pretrained model. Defaults to 'ernie-3.0-tiny-nano-v2-zh'"},
    )
    infer_prefix: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )

    dropout: float = field(default=0.1, metadata={"help": "Dropout rate for JointErnie. Defaults to 0.1."})
    dynamic: bool = field(default=False)


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, compression_args = parser.parse_args_into_dataclasses()

    paddle.set_device(compression_args.device)

    intent_label_names, slot_label_names, intent2id, slot2id = get_label_name(
        data_args.intent_label_path, data_args.slot_label_path
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    paddle.enable_static()
    place = paddle.set_device(compression_args.device)
    exe = paddle.static.Executor(place)

    program, feed_target_names, fetch_targets = paddle.static.load_inference_model(model_args.infer_prefix, exe)

    if compression_args.do_eval:
        test_dataset = load_dataset(
            read_example,
            filename=data_args.test_path,
            intent2id=intent2id,
            slot2id=slot2id,
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            no_entity_id=data_args.ignore_index,
            lazy=False,
        )
        intent_right, slot_right = 0, 0
        for data in test_dataset:
            input_ids = np.array(data["input_ids"])
            intent_logits, slot_logits = exe.run(
                program, feed={"input_ids": input_ids.reshape(1, -1).astype("int32")}, fetch_list=fetch_targets
            )
            slot_pred = slot_logits.argmax(axis=-1)
            intent_pred = intent_logits.argmax(axis=-1)

            intent_label = np.array(data["intent_label"])
            slot_label = np.array(data["slot_label"])

            padding_mask = input_ids == 0
            padding_mask |= (input_ids == 2) | (input_ids == 1)

            if intent_label == intent_pred:
                intent_right += 1
                if intent_label in (0, 2, 3, 4, 6, 7, 8, 10):
                    slot_right += 1
                elif ((slot_pred == slot_label) | padding_mask).all():
                    slot_right += 1
        accuracy = slot_right / len(test_dataset) * 100
        intent_accuracy = intent_right / len(test_dataset) * 100

        print("accuray: %.2f, intent_accuracy: %.2f" % (accuracy, intent_accuracy))
    else:
        test_dataset = load_dataset(
            read_test_file,
            filename=data_args.test_path,
            lazy=False,
        )
        for data in test_dataset:
            query_list = [data["query"]]
            query_input_dict = input_preprocess(query_list, tokenizer, max_seq_length=16)
            input_ids = query_input_dict["input_ids"]
            intent_logits, slot_logits = exe.run(program, feed={"input_ids": input_ids}, fetch_list=fetch_targets)

            # Shows result
            intent_out = intent_cls_postprocess(intent_logits, intent_label_names)
            slots_out = slot_cls_postprocess(slot_logits, query_list, slot_label_names)
            print(query_list, "\n", intent_out, "\n", slots_out)


if __name__ == "__main__":
    main()
