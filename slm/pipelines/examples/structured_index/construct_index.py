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

import os

from arguments import StructuredIndexerPipelineArguments
from src.structured_index import StructuredIndexer

from paddlenlp.trainer import PdArgumentParser

if __name__ == "__main__":
    parser = PdArgumentParser(StructuredIndexerPipelineArguments)
    (args,) = parser.parse_args_into_dataclasses()

    structured_indexer = StructuredIndexer(log_dir=args.log_dir)
    assert os.path.exists(args.source)
    if os.path.isfile(args.source):
        structured_indexer.pipeline(
            filepath=args.source,
            parse_model_name_or_path=args.parse_model_name_or_path,
            parse_model_url=args.parse_model_url,
            summarize_model_name_or_path=args.summarize_model_name_or_path,
            summarize_model_url=args.summarize_model_url,
            encode_model_name_or_path=args.encode_model_name_or_path,
        )
    else:
        for root, _, files in os.walk(args.source):
            for file in files:
                filepath = os.path.join(root, file)
                structured_indexer.pipeline(
                    filepath=filepath,
                    parse_model_name_or_path=args.parse_model_name_or_path,
                    parse_model_url=args.parse_model_url,
                    summarize_model_name_or_path=args.summarize_model_name_or_path,
                    summarize_model_url=args.summarize_model_url,
                    encode_model_name_or_path=args.encode_model_name_or_path,
                )
