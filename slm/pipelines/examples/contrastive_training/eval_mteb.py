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
import sys

import mteb
import paddle
from models.modeling import BiEncoderModel
from models.modeling_nv import NVEncodeModel
from mteb import MTEB

from paddlenlp.peft import LoRAConfig, LoRAModel
from paddlenlp.transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class MTEB_EvalModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode_queries(self, queries, **kwargs):
        return self.model.encode_queries(queries)

    def encode_corpus(self, corpus, **kwargs):
        return self.model.encode_corpus(corpus)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--peft_model_name_or_path", default=None, type=str)
    parser.add_argument("--output_folder", default="tmp", type=str)

    parser.add_argument("--task_name", default="SciFact", type=str)
    parser.add_argument(
        "--task_split", default="test", type=str
    )  # some datasets do not have "test", they only have "dev"

    parser.add_argument("--query_instruction", default="query: ", type=str)
    parser.add_argument("--document_instruction", default="document: ", type=str)

    parser.add_argument("--pooling_method", default="last", type=str)  # mean, last, cls
    parser.add_argument("--max_seq_length", default=4096, type=int)
    parser.add_argument("--eval_batch_size", default=1, type=int)

    parser.add_argument("--pad_token", default="unk_token", type=str)  # unk_token, eos_token
    parser.add_argument("--padding_side", default="left", type=str)  # right, left
    parser.add_argument("--add_bos_token", default=0, type=int)
    parser.add_argument("--add_eos_token", default=1, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    if "NV-Embed" in args.base_model_name_or_path:
        logger.info("Using NV-Embed")

        query_prefix = "Instruct: " + args.query_instruction + "\nQuery: "
        passage_prefix = ""

        if args.task_name == "QuoraRetrieval":
            assert args.document_instruction != "document: ", "QuoraRetrieval requires a document instruction"
            passage_prefix = "Instruct: " + args.document_instruction + "\nQuery: "  # because this is STS task

        encode_model = NVEncodeModel.from_pretrained(
            args.base_model_name_or_path,
            tokenizer_path=args.base_model_name_or_path,
            eval_batch_size=args.eval_batch_size,
            max_seq_length=args.max_seq_length,
            query_instruction=query_prefix,
            document_instruction=passage_prefix,
            dtype="bfloat16" if args.peft_model_name_or_path else "float16",
        )

        if args.peft_model_name_or_path is not None:
            lora_config = LoRAConfig.from_pretrained(args.peft_model_name_or_path)
            lora_config.merge_weights = True
            encode_model = LoRAModel.from_pretrained(
                encode_model, args.peft_model_name_or_path, lora_config=lora_config, dtype="bfloat16"
            )
        tokenizer = encode_model.tokenizer

    else:
        logger.info("Using Normal AutoModel")

        assert args.add_bos_token in [0, 1], f"add_bos_token should be either 0 or 1, but got {args.add_bos_token}"
        assert args.add_eos_token in [0, 1], f"add_eos_token should be either 0 or 1, but got {args.add_eos_token}"
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
        assert hasattr(tokenizer, args.pad_token), f"Tokenizer does not have {args.pad_token} token"
        token_dict = {"unk_token": tokenizer.unk_token, "eos_token": tokenizer.eos_token}
        tokenizer.pad_token = token_dict[args.pad_token]
        assert args.padding_side in [
            "right",
            "left",
        ], f"padding_side should be either 'right' or 'left', but got {args.padding_side}"
        assert not (
            args.padding_side == "left" and args.pooling_method == "cls"
        ), "Padding 'left' is not supported for pooling method 'cls'"
        tokenizer.padding_side = args.padding_side
        tokenizer.add_bos_token = bool(args.add_bos_token)
        tokenizer.add_eos_token = bool(args.add_eos_token)

        encode_model = BiEncoderModel(
            model_name_or_path=args.base_model_name_or_path,
            normalized=True,
            sentence_pooling_method=args.pooling_method,
            tokenizer=tokenizer,
            eval_batch_size=args.eval_batch_size,
            max_seq_length=args.max_seq_length,
        )

        if args.peft_model_name_or_path:
            lora_config = LoRAConfig.from_pretrained(args.peft_model_name_or_path)
            lora_config.merge_weights = True
            encode_model.config = (
                encode_model.model_config
            )  # for NV-Embed, this is no needed, but for repllama, this is needed
            encode_model.config.tensor_parallel_degree = 1
            encode_model = LoRAModel.from_pretrained(
                encode_model, args.peft_model_name_or_path, lora_config=lora_config, dtype=lora_config.dtype
            )

    encode_model.eval()
    mtb_eval_model = MTEB_EvalModel(encode_model, tokenizer)

    logger.info("Ready to eval")
    evaluation = MTEB(tasks=mteb.get_tasks(tasks=[args.task_name]))
    evaluation.run(
        encode_model,
        output_folder=f"{args.output_folder}/{args.task_name}/{args.pooling_method}",
        eval_splits=[args.task_split],
    )
