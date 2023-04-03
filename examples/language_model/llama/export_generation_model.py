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

import argparse
import os

import paddle
from configuration import LlamaConfig
from modeling import LlamaForCausalLM
from tokenizer import LlamaTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="checkpoints",
        type=str,
        required=False,
        help="Path of the trained model to be exported.",
    )
    parser.add_argument(
        "--output_path",
        default="inference/llama",
        type=str,
        help="The output file prefix used to save the exported inference model.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    paddle.seed(100)
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()
    strategy = paddle.distributed.fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": tensor_parallel_degree,
        "pp_degree": 1,
        "sharding_degree": 1,
    }
    paddle.distributed.fleet.init(is_collective=True, strategy=strategy)

    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    config = LlamaConfig.from_pretrained(args.model_path)

    # Set the generaiton the hyperparameter
    config.max_length = 100
    config.min_length = 0
    config.decode_strategy = "sampling"
    config.temperature = 1.0
    config.top_k = 1
    config.top_p = 1.0
    config.repetition_penalty = 1.0
    config.use_cache = True
    config.use_recompute = False

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
        load_state_as_np=True,
        low_cpu_mem_usage=True,
        config=config,
    )

    model.eval()
    model = paddle.jit.to_static(
        model.generate,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # attention_mask
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # position_ids
            config.max_length,
            config.min_length,
            config.decode_strategy,
            config.temperature,
            config.top_k,
            config.top_p,
            config.repetition_penalty,
        ],
    )

    # Save converted static graph model
    paddle.jit.save(model, args.output_path)
    # Also save tokenizer for inference usage
    tokenizer.save_pretrained(os.path.dirname(args.output_path))


if __name__ == "__main__":
    main()
