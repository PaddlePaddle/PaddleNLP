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

import os
from paddlenlp.utils.downloader import get_path_from_url, download_check
from .utils import PPDIFFUSERS_CACHE, DOWNLOAD_SERVER


def ppdiffusers_bos_download(pretrained_model_name_or_path,
                             filename=None,
                             subfolder=None):
    cache_dir = pretrained_model_name_or_path if os.path.isdir(
        pretrained_model_name_or_path) else os.path.join(
            PPDIFFUSERS_CACHE, pretrained_model_name_or_path)
    url = DOWNLOAD_SERVER + "/" + pretrained_model_name_or_path
    if subfolder is not None:
        url = url + "/" + subfolder
        cache_dir = os.path.join(cache_dir, subfolder)
    if filename is not None:
        url = url + "/" + filename
    file_path = get_path_from_url(url, cache_dir)
    return file_path
