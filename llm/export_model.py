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
from predictor import ModelArgument, PredictorArgument, create_predictor
from utils import generate_rank_mapping, get_infer_model_path

from paddlenlp.trainer import PdArgumentParser


@dataclass
class ExportArgument:
    output_path: str = field(default=None, metadata={"help": "The output path of model."})


def main():
    parser = PdArgumentParser((PredictorArgument, ModelArgument, ExportArgument))
    predictor_args, model_args, export_args = parser.parse_args_into_dataclasses()

    paddle.set_default_dtype(predictor_args.dtype)

    # set predictor type
    predictor = create_predictor(predictor_args, model_args)
    predictor.model.eval()

    predictor.model.to_static(
        get_infer_model_path(export_args.output_path, predictor_args.model_prefix), {"dtype": predictor_args.dtype}
    )
    predictor.model.config.save_pretrained(export_args.output_path)
    predictor.tokenizer.save_pretrained(export_args.output_path)
    generate_rank_mapping(os.path.join(export_args.output_path, "rank_mapping.csv"))


if __name__ == "__main__":
    main()
