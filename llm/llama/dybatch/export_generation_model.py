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
from __future__ import annotations

import argparse
import os
import shutil

import paddle
from modeling import LlamaForCausalLMDyBatch
from paddle.distributed import fleet
from utils import generate_rank_mapping, get_infer_model_path

from paddlenlp.transformers import AutoTokenizer, LlamaConfig


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name_or_dir",
        default="facebook/llama-65b",
        type=str,
        required=True,
        help="name or path of the trained model to be exported.",
    )
    parser.add_argument(
        "--model_dtype",
        default="float16",
        type=str,
        help="Model dtype selected in the list: float32, float16, bfloat16",
    )
    parser.add_argument(
        "--output_dir",
        default="./inference_model",
        type=str,
        # required=True,
        help="The output file prefix used to save the exported inference model.",
    )
    parser.add_argument(
        "--model_prefix",
        default="llama",
        type=str,
        help="prefix name of pdmodel/pdiparams file",
    )
    parser.add_argument(
        "--max_src_length",
        default=1024,
        type=int,
        help="max length of input sentence",
    )
    parser.add_argument(
        "--max_length",
        default=1024,
        type=int,
        help="max length of output sentence",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_dir)
    paddle.set_default_dtype(args.model_dtype)

    shutil.copyfile(os.path.join(args.model_name_or_dir, "config.json"), os.path.join(args.output_dir, "config.json"))

    tensor_parallel_degree = paddle.distributed.get_world_size()
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    config = LlamaConfig.from_pretrained(args.model_name_or_dir)
    config.tensor_parallel_degree = tensor_parallel_degree
    config.dtype = args.model_dtype
    model = LlamaForCausalLMDyBatch.from_pretrained(args.model_name_or_dir, config=config)

    model.eval()

    input_spec = [
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),  # input_ids
        paddle.static.InputSpec(
            shape=[None, 1, None, None], dtype=args.model_dtype, name="attention_mask"
        ),  # attention_mask
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="position_ids"),  # position_ids
        paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="penalty_score"),  # penalty_score
        paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="frequency_score"),  # frequency_score
        paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="presence_score"),  # presence_score
        paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="min_length"),  # min_decode_length
        paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="max_length"),  # max_decode_length
        paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="temperature"),  # temperature
        paddle.static.InputSpec(shape=[None, 1], dtype="float32", name="top_p"),  # top_p
        paddle.static.InputSpec(shape=[None], dtype="int64", name="eos_token_id"),  # eos_token_id
        paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_encoder"),  # seq_len_encoder
        paddle.static.InputSpec(shape=[None, 1], dtype="int32", name="seq_len_decoder"),  # seq_len_decoder
        paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="step_idx"),  # step_idx
        paddle.static.InputSpec(shape=[None, 1], dtype="bool", name="stop_flags"),  # stop_flags
        paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_ids"),  # tgt_ids
        paddle.static.InputSpec(shape=[None, 1], dtype="int64", name="tgt_pos"),  # tgt_pos
        paddle.static.InputSpec(
            shape=[None, 1, 1, None], dtype=args.model_dtype, name="tgt_generation_mask"
        ),  # tgt_generation_mask
        paddle.static.InputSpec(shape=[None, None], dtype="int64", name="pre_ids"),  # pre_ids
        paddle.static.InputSpec(shape=[1], dtype="int64", name="stop_nums"),  # stop_nums
        [
            paddle.static.InputSpec(
                shape=[
                    2,
                    None,
                    model.config.num_attention_heads // tensor_parallel_degree,
                    args.max_src_length + args.max_length,
                    model.config.hidden_size // model.config.num_attention_heads,
                ],
                dtype=args.model_dtype,
                name="cache_kvs_{}".format(i),
            )
            for i in range(model.config.num_hidden_layers)
        ],  # cache_kvs
    ]
    model = paddle.jit.to_static(model.generate_dybatch, input_spec=input_spec)

    # Save converted static graph model
    paddle.jit.save(model, get_infer_model_path(args.output_dir, args.model_prefix))
    # Also save tokenizer for inference usage
    tokenizer.save_pretrained(args.output_dir)
    # Generation rank_mapping.csv for distributed model
    generate_rank_mapping(os.path.join(args.output_dir, "rank_mapping.csv"))


if __name__ == "__main__":
    main()
