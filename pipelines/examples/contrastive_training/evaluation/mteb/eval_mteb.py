# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import logging

from mteb import MTEB
from mteb_models import EncodeModel

from paddlenlp.transformers import AutoModel, AutoTokenizer


def get_model(peft_model_name, base_model_name):
    if peft_model_name is not None:
        raise NotImplementedError("PEFT model is not supported yet")
    else:
        base_model = AutoModel.from_pretrained(base_model_name)
        return base_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", default="bge-large-en-v1.5", type=str)
    parser.add_argument("--peft_model_name_or_path", default=None, type=str)
    parser.add_argument("--output_folder", default="tmp", type=str)

    parser.add_argument("--task_name", default="SciFact", type=str)
    parser.add_argument(
        "--task_split",
        default="test",
        help='Note that some datasets do not have "test", they only have "dev"',
        type=str,
    )

    parser.add_argument("--query_instruction", default=None, help="add prefix instruction before query", type=str)
    parser.add_argument(
        "--document_instruction", default=None, help="add prefix instruction before document", type=str
    )

    parser.add_argument("--pooling_method", default="last", help="choose in [mean, last, cls]", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)

    parser.add_argument("--pad_token", default="unk_token", help="unk_token, eos_token or pad_token", type=str)
    parser.add_argument("--padding_side", default="left", help="right or left", type=str)
    parser.add_argument("--add_bos_token", default=0, help="1 means add token", type=int)
    parser.add_argument("--add_eos_token", default=1, help="1 means add token", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Args: {}".format(args))

    model = get_model(args.peft_model_name_or_path, args.base_model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    assert hasattr(tokenizer, args.pad_token), f"Tokenizer does not have {args.pad_token} token"
    token_dict = {"unk_token": tokenizer.unk_token, "eos_token": tokenizer.eos_token, "pad_token": tokenizer.pad_token}
    tokenizer.pad_token = token_dict[args.pad_token]

    assert args.padding_side in [
        "right",
        "left",
    ], f"padding_side should be either 'right' or 'left', but got {args.padding_side}"
    assert not (
        args.padding_side == "left" and args.pooling_method == "cls"
    ), "Padding 'left' is not supported for pooling method 'cls'"
    tokenizer.padding_side = args.padding_side

    assert args.add_bos_token in [0, 1], f"add_bos_token should be either 0 or 1, but got {args.add_bos_token}"
    assert args.add_eos_token in [0, 1], f"add_eos_token should be either 0 or 1, but got {args.add_eos_token}"
    tokenizer.add_bos_token = bool(args.add_bos_token)
    tokenizer.add_eos_token = bool(args.add_eos_token)

    encode_model = EncodeModel(
        model=model,
        tokenizer=tokenizer,
        pooling_method=args.pooling_method,
        query_instruction=args.query_instruction,
        document_instruction=args.document_instruction,
        eval_batch_size=args.eval_batch_size,
        max_seq_length=args.max_seq_length,
    )

    logger.info("Ready to eval")
    evaluation = MTEB(tasks=[args.task_name])
    evaluation.run(
        encode_model,
        output_folder=f"{args.output_folder}/{args.task_name}/{args.pooling_method}",
        eval_splits=[args.task_split],
    )
