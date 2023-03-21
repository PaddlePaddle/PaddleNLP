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

import os

import paddle
from modeling import BloomConfig, BloomForGeneration
from paddle.distributed import fleet
from tap import Tap
from transformers import AutoTokenizer
from utils import set_hyrbid_parallel_seed

from paddlenlp.utils.log import logger


class Args(Tap):
    model_name_or_path: str  # bloom model name, eg: bloom-560m
    device: str = "gpu"
    dtype: str = "float16"  # dtype of model, when running `bigscience/bloom` which is under bfloat16, it should be set to bfloat16
    max_length: int = 20  # max length of generated sequence length
    top_k: int = 1


def get_custom_black_list(args: Args):
    if args.dtype == "float16":
        return ["multinomial"]

    return [
        "reduce_sum",
        "c_softmax_with_cross_entropy",
        "elementwise_div",
        "lookup_table",
        "lookup_table_v2",
        "layer_norm",
        "set_value",
        "fill_constant",
        "softmax",
        "multinomial",
    ]


def left_padding(inputs, pad_id):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = [extend_max_lenth(value, max_length, to_pad_id) for value in values]
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)

    return inputs


def convert_examples(sentences: list[str], tokenizer, max_length: int = 20):
    features = tokenizer.batch_encode_plus(
        sentences,
        padding="max_length",
        max_length=max_length,
    )
    features = left_padding(features, tokenizer.pad_token_id)
    return paddle.to_tensor(features["input_ids"], dtype="int64")


def load_model(args: Args, config: BloomConfig):
    # disable init_weights method
    # BloomModel.init_weights = lambda *_: None
    if args.dtype == "float16":
        return BloomForGeneration.from_pretrained(args.model_name_or_path, config=config)

    if not os.path.isdir(args.model_name_or_path):
        raise ValueError("you must download the sharded model weight file into local dir")

    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": args.dp_degree,
        "mp_degree": args.mp_degree,
        "pp_degree": 1,
        "sharding_degree": args.sharding_degree,
    }

    # Set control in tensor parallel
    strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

    fleet.init(is_collective=True, strategy=strategy)

    # Obtain rank message of hybrid parallel
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()

    sharding_size = hcg.get_sharding_parallel_world_size()
    data_world_rank = dp_rank * sharding_size + sharding_rank

    # Seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, data_world_rank, mp_rank)

    weight_file = os.path.join(args.model_name_or_path, f"auto_dist{mp_rank}.pdparams")
    logger.info(f"start to load rank weight<{weight_file}> for model")
    with paddle.LazyGuard():
        # TODO(wj-Mcat): hack in init_weights because there are some unsupported ops
        # BloomForGeneration.init_weights = lambda x: x
        model = BloomForGeneration(config)

    # set state dict
    state_dict = paddle.load(weight_file, return_numpy=True)
    model.set_state_dict(state_dict)
    return model


def generate(args: Args):
    paddle.set_device(args.device)

    assert args.dtype in ["float16", "bfloat16"], "dtype must be one of `float16`, `bfloat16`"
    paddle.set_default_dtype(args.dtype)

    config = BloomConfig.from_pretrained(args.model_name_or_path)
    config.top_k = 1
    model = load_model(args, config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"

    sentences = [
        "Where is the captial of China?",
        "I love you,",
        "What is your problem ?",
    ]
    input_ids = convert_examples(sentences, tokenizer, args.max_length)
    print("start to forward model ...")

    with paddle.amp.auto_cast(False, custom_black_list=get_custom_black_list(args), level="O2", dtype=args.dtype):
        decoded_ids = model(input_ids)[0]

    decoded_ids = decoded_ids.detach().tolist()
    for index, ids in enumerate(decoded_ids):
        print("========================================")
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded_sentence = tokenizer.convert_tokens_to_string(tokens)
        print("query:\t" + sentences[index])
        print("result:\t" + decoded_sentence)


if __name__ == "__main__":
    args = Args().parse_args(known_only=True)
    generate(args)
