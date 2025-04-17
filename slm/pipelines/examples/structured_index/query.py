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

from arguments import RetrievalArguments
from src.structured_index import StructuredIndexer

from paddlenlp.trainer import PdArgumentParser

if __name__ == "__main__":
    parser = PdArgumentParser(RetrievalArguments)
    (args,) = parser.parse_args_into_dataclasses()

    structured_indexer = StructuredIndexer(log_dir=args.log_dir)

    from src.utils import load_data

    if args.query_text is None:
        queries_dict = load_data(args.query_filepath, mode="Searching")
    else:

        assert isinstance(args.query_text, str)
        queries_dict = {"0": args.query_text}

    structured_indexer.search(
        queries_dict=queries_dict,
        output_dir=args.search_result_dir,
        model_name_or_path=args.encode_model_name_or_path,
        top_k=args.top_k,
        embedding_batch_size=args.embedding_batch_size,
    )
