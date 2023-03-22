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
from pprint import pprint as print

import paddle
from args import parse_args
from paddle import LazyGuard
from paddle.distributed import fleet
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils import set_hyrbid_parallel_seed

from paddlenlp.trainer import get_last_checkpoint
from paddlenlp.transformers import BloomConfig, BloomForGeneration
from paddlenlp.utils.log import logger

paddle.set_default_dtype("bfloat16")


def left_padding(inputs, pad_id, padding="longest"):
    assert "input_ids" in inputs, "input_ids should be in inputs!"
    max_length = 0
    for ids in inputs["input_ids"]:
        max_length = max(max_length, len(ids))

    def extend_max_lenth(value, max_length, to_pad_id):
        return [to_pad_id] * (max_length - len(value)) + value

    def extend_filed(name, max_length, to_pad_id):
        values = inputs[name]
        res = []
        for index, value in enumerate(values):
            res.append(extend_max_lenth(value, max_length, to_pad_id))
        inputs[name] = res

    extend_filed("input_ids", max_length, pad_id)
    if "attention_mask" in inputs:
        extend_filed("attention_mask", max_length, 0)
    if "position_ids" in inputs:
        extend_filed("position_ids", max_length, 0)

    return inputs


def convert_examples(sentences: list[str], tokenizer: PreTrainedTokenizer, max_length: int = 20):
    features = tokenizer.batch_encode_plus(
        sentences,
        padding="max_length",
        max_length=max_length,
    )
    features = left_padding(features, tokenizer.pad_token_id)
    return paddle.to_tensor(features["input_ids"], dtype="int64")


def get_bloom_mesh_info(layer_size: int):
    all_splits = [
        ["bloom.word_embeddings.weight", "embedding_0.w_0"],
        ["bloom.word_embeddings_layernorm.weight", "layer_norm_0.w_0"],
        ["bloom.word_embeddings_layernorm.bias", "layer_norm_0.b_0"],
    ]
    for layer_index in range(layer_size):
        layer_norm_size = 2
        layer_linear_size = 4
        all_splits.extend(
            [
                [
                    f"bloom.h.{layer_index}.input_layernorm.weight",
                    f"layer_norm_{layer_index * layer_norm_size + 1}.w_0",
                ],
                [f"bloom.h.{layer_index}.input_layernorm.bias", f"layer_norm_{layer_index * layer_norm_size + 1}.b_0"],
                [
                    f"bloom.h.{layer_index}.self_attention.query_key_value.weight",
                    f"linear_{layer_index * layer_linear_size + 0}.w_0",
                ],
                [
                    f"bloom.h.{layer_index}.self_attention.query_key_value.bias",
                    f"linear_{layer_index * layer_linear_size + 0}.b_0",
                ],
                [
                    f"bloom.h.{layer_index}.self_attention.dense.weight",
                    f"linear_{layer_index * layer_linear_size + 1}.w_0",
                ],
                [
                    f"bloom.h.{layer_index}.self_attention.dense.bias",
                    f"linear_{layer_index * layer_linear_size + 1}.b_0",
                ],
                [
                    f"bloom.h.{layer_index}.mlp.dense_h_to_4h.weight",
                    f"linear_{layer_index * layer_linear_size + 2}.w_0",
                ],
                [f"bloom.h.{layer_index}.mlp.dense_h_to_4h.bias", f"linear_{layer_index * layer_linear_size + 2}.b_0"],
                [
                    f"bloom.h.{layer_index}.mlp.dense_4h_to_h.weight",
                    f"linear_{layer_index * layer_linear_size + 3}.w_0",
                ],
                [f"bloom.h.{layer_index}.mlp.dense_4h_to_h.bias", f"linear_{layer_index * layer_linear_size + 3}.b_0"],
                [f"bloom.h.{layer_index}.post_attention_layernorm.weight", f"layer_norm_{layer_index * 2 + 2}.w_0"],
                [f"bloom.h.{layer_index}.post_attention_layernorm.bias", f"layer_norm_{layer_index * 2 + 2}.b_0"],
            ]
        )

    all_splits.extend(
        [
            ["bloom.ln_f.weight", f"layer_norm_{layer_size * layer_norm_size + 1}.w_0"],
            ["bloom.ln_f.bias", f"layer_norm_{layer_size * layer_norm_size + 1}.b_0"],
        ]
    )
    return {splits[1]: splits[0] for splits in all_splits}


@paddle.no_grad()
def do_generation(args):
    paddle.set_device(args.device)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = BloomConfig.from_pretrained(args.model_name_or_path)

    # Detecting last checkpoint.
    last_checkpoint = None
    training_args = args
    training_args.overwrite_output_dir = False
    training_args.resume_from_checkpoint = True
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    WEIGHTS_NAME = "model_state.pdparams"
    if args.mp_degree > 1:
        WEIGHTS_NAME = f"auto_dist{mp_rank}.pdparams"
        BloomForGeneration.resource_files_names = {"model_state": WEIGHTS_NAME}

    config.mp_rank = mp_rank
    config.mp_degree = args.mp_degree

    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["use_recompute"] = args.use_recompute
    config["enable_fuse_transformer"] = False
    config["use_cache"] = True

    with LazyGuard():
        model = BloomForGeneration(config=config)

    state_dict = paddle.load(os.path.join(args.model_name_or_path, f"auto_dist{mp_rank}.pdparams"), return_numpy=True)

    if "bloom.word_embeddings.weight" not in state_dict:
        mapping = get_bloom_mesh_info(70)
        state_dict = {mapping[key]: value for key, value in state_dict.items()}

    model.set_state_dict(state_dict)

    # disable the distributed model wrapper to ignore some checking error, but it should be here
    # model = fleet.distributed_model(model)

    # sentence examples to do generation
    sentences = [
        "Where is the captial of China?",
        "I love you, but",
        "What is your problem ?",
    ]
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
    tokenizer.padding_side = "left"
    input_ids = convert_examples(sentences, tokenizer, None)
    custom_black_list = [
        "reduce_sum",
        "c_softmax_with_cross_entropy",
        "elementwise_div",
        "lookup_table",
        "lookup_table_v2",
        "layer_norm",
        "set_value",
        "fill_constant",
        "softmax",
    ]
    custom_white_list = []
    with paddle.amp.auto_cast(
        args.use_pure_fp16,
        custom_black_list=custom_black_list,
        custom_white_list=custom_white_list,
        level="O2",
        dtype="bfloat16",
    ):
        decoded_ids = model(input_ids)[0]

    decoded_ids = decoded_ids.detach().tolist()
    for ids in decoded_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(tokenizer.convert_tokens_to_string(tokens))


if __name__ == "__main__":
    args = parse_args()
    do_generation(args)
