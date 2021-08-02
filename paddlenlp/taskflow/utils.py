# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from paddle.dataset.common import md5file
from ..utils.downloader import get_path_from_url, DownloaderCheck
from ..utils.env import MODEL_HOME

DOC_FORMAT = r"""
    Examples:
        .. code-block:: python
               """
DOWNLOAD_CHECK = False


def download_file(save_dir, filename, url, md5=None, task=None):
    """
    Download the file from the url to specified directory. 
    Check md5 value when the file is exists, if the md5 value is the same as the existed file, just use 
    the older file, if not, will download the file from the url.

    Args:
        save_dir(string): The specified directory saving the file.
        fiename(string): The specified filename saveing the file.
        url(string): The url downling the file.
        md5(string, optional): The md5 value that checking the version downloaded. 
    """
    global DOWNLOAD_CHECK
    if not DOWNLOAD_CHECK:
        DOWNLOAD_CHECK = True
        checker = DownloaderCheck(task)
        checker.start()
        checker.join()
    default_root = os.path.join(MODEL_HOME, save_dir)
    fullname = os.path.join(default_root, filename)
    if os.path.exists(fullname):
        if md5 and (not md5file(fullname) == md5):
            get_path_from_url(url, default_root, md5)
    else:
        get_path_from_url(url, default_root, md5)
    return fullname


def add_docstrings(*docstr):
    """
    The function that add the doc string to doc of class.
    """

    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + "".join(DOC_FORMAT) + "".join(docstr)
        return fn

    return docstring_decorator
