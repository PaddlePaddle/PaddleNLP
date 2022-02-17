#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""create squad train squad_data"""

from utils.model_utils.config_args import args_config as args
from utils.reading_comprehension.dataProcessing import build_data_change
from paddlenlp.transformers import LukeTokenizer
import os

args.wiki_link_db_file = os.path.join(args.wiki_data, "enwiki_20160305.pkl")
args.model_redirects_file = os.path.join(args.wiki_data,
                                         "enwiki_20181220_redirects.pkl")
args.link_redirects_file = os.path.join(args.wiki_data,
                                        "enwiki_20160305_redirects.pkl")

args.tokenizer = LukeTokenizer.from_pretrained('luke-base')
args.entity_vocab = args.tokenizer.get_entity_vocab()
build_data_change(args)
