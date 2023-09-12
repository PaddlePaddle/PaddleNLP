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
from dataclasses import dataclass, field

import paddle
from paddle.distributed import fleet
from predictor import ModelArgument, PredictorArgument, create_predictor
from tqdm import tqdm
from utils import generate_rank_mapping, get_infer_model_path

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.utils.log import logger


@dataclass
class ExportArgument:
    output_path: str = field(default=None, metadata={"help": "The output path of model."})


def load_inference_model(model_path, model_name, param_name, exe):
    model_abs_path = os.path.join(model_path, model_name)
    param_abs_path = os.path.join(model_path, param_name)
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return paddle.static.io.load_inference_model(model_path, exe, model_name, param_name)
    else:
        return paddle.static.io.load_inference_model(model_path, exe)


def validate_pdmodel(model_path, model_prefix):
    paddle.enable_static()
    place = paddle.CUDAPlace(0)
    exe = paddle.static.Executor(place)
    scope = paddle.static.Scope()

    with paddle.static.scope_guard(scope):
        net_program, feed_target_names, fetch_targets = paddle.static.io.load_inference_model(
            os.path.join(model_path, model_prefix), exe
        )

        for block in net_program.blocks:
            ops: list[paddle.framework.Operator] = block.ops
            for op in tqdm(ops, desc="checking the validation of ops"):
                if op.type.lower() == "print":
                    logger.warning(f"UNEXPECTED OP<{op.type}> which will reduce the performace of the static model")


def main():
    parser = PdArgumentParser((PredictorArgument, ModelArgument, ExportArgument))
    predictor_args, model_args, export_args = parser.parse_args_into_dataclasses()

    paddle.set_default_dtype(predictor_args.dtype)
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()
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

    # set predictor type
    predictor = create_predictor(predictor_args, model_args, tensor_parallel_degree, tensor_parallel_rank)
    predictor.model.eval()

    predictor.model.to_static(
        get_infer_model_path(export_args.output_path, predictor_args.model_prefix),
        {"dtype": predictor_args.dtype, "export_precache": predictor_args.export_precache},
    )
    predictor.model.config.save_pretrained(export_args.output_path)
    predictor.tokenizer.save_pretrained(export_args.output_path)
    generate_rank_mapping(os.path.join(export_args.output_path, "rank_mapping.csv"))

    validate_pdmodel(export_args.output_path, "model")


if __name__ == "__main__":
    main()
