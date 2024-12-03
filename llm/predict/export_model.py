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
from predict.predictor import ModelArgument, PredictorArgument, create_predictor

from paddlenlp.trainer import PdArgumentParser
from paddlenlp.trl import llm_utils


@dataclass
class ExportArgument:
    output_path: str = field(default=None, metadata={"help": "The output path of model."})


def add_inference_args_to_config(model_config, args):
    """Add export arguments to config."""
    model_config.infer_model_block_size = args.block_size
    model_config.infer_model_max_seq_len = args.total_max_length
    model_config.infer_model_cachekv_int8_type = args.cachekv_int8_type
    model_config.infer_model_dtype = args.dtype
    model_config.infer_model_paddle_commit = paddle.version.commit


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
    predictor = create_predictor(predictor_args, model_args)
    predictor.model.eval()

    predictor.model.to_static(
        llm_utils.get_infer_model_path(export_args.output_path, predictor_args.model_prefix),
        {
            "dtype": predictor_args.dtype,
            "export_precache": predictor_args.export_precache,
            "cachekv_int8_type": predictor_args.cachekv_int8_type,
            "speculate_method": predictor_args.speculate_method,
        },
    )
    add_inference_args_to_config(predictor.model.config, predictor_args)
    predictor.model.config.save_pretrained(export_args.output_path)
    if predictor.generation_config is not None:
        predictor.generation_config.save_pretrained(export_args.output_path)
    else:
        predictor.model.generation_config.save_pretrained(export_args.output_path)

    predictor.tokenizer.save_pretrained(export_args.output_path)

    if tensor_parallel_degree > 1:
        export_args.output_path = os.path.join(export_args.output_path, f"rank_{tensor_parallel_rank}")

    if predictor_args.device == "npu":
        from npu.llama.export_utils import process_params

        process_params(os.path.join(export_args.output_path, predictor_args.model_prefix))


if __name__ == "__main__":
    main()
