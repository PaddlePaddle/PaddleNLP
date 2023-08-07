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
import paddle
from paddle.distributed import fleet

from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of model.")
    parser.add_argument("--merge_model_path", default=None, required=True, help="The directory of merged model.")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument("--dtype", type=str, default=None, required=True, help="Model dtype")
    parser.add_argument("--with_tokenizer", type=bool, default=True, help="Save tokenizer at the same time")
    return parser.parse_args()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = 0
    if tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        hcg = fleet.get_hybrid_communicate_group()
        tensor_parallel_rank = hcg.get_model_parallel_rank()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=args.dtype,
        tensor_parallel_degree=tensor_parallel_degree,
        tensor_parallel_rank=tensor_parallel_rank,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tensor_parallel_rank == 0:
        model.save_pretrained(args.merge_model_path, merge_tensor_parallel=tensor_parallel_degree > 1)
        tokenizer.save_pretrained(args.merge_model_path)


if __name__ == "__main__":
    merge()
