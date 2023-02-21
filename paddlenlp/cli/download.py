# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import json
import os
from typing import List, Tuple

from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger

COMMUNITY_MODEL_CONFIG_FILE_NAME = "community_models.json"


def load_community_models() -> List[Tuple[str, str]]:
    """load community models based on remote models.json

    Returns:
        List[Tuple[str, str]]: the name tuples of community models
    """
    # 1. check & download community models.json
    local_community_model_config_path = os.path.join(MODEL_HOME, "community_models.json")

    if not os.path.exists(local_community_model_config_path):
        logger.info("download community model configuration from server ...")
        remote_community_model_path = "/".join([COMMUNITY_MODEL_PREFIX, COMMUNITY_MODEL_CONFIG_FILE_NAME])
        cache_dir = os.path.join(MODEL_HOME)
        local_community_model_config_path = get_path_from_url(remote_community_model_path, root_dir=cache_dir)

    # 2. load configuration
    #
    # config = {
    #   "model_name": {
    #       "type": "",
    #       "files": ["", ""]
    #   }
    # }
    #

    with open(local_community_model_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_names = set()
    for model_name, obj in config.items():
        model_names.add((model_name, obj.get("model_type", "")))
    logger.info(f"find {len(model_names)} community models ...")
    return model_names
