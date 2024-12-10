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
import sys

from paddlenlp.mergekit import MergeConfig, MergeModel
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.utils.log import logger


def merge():
    parser = PdArgumentParser((MergeConfig))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        merge_config = parser.parse_json_file_and_cmd_lines()[0]
    else:
        merge_config = parser.parse_args_into_dataclasses()[0]

    mergekit = MergeModel(merge_config)
    logger.info("Start to merge model.")
    mergekit.merge_model()
    logger.info("Finish merging model.")


if __name__ == "__main__":
    merge()
