# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union, Tuple, List, Dict

from abc import ABC

from .. import C
from .. import Encoding


class PostProcessor(ABC):
    def num_special_tokens_to_add(self, is_pair: bool = True):
        return self._postprocessor.num_special_tokens_to_add(is_pair)

    def __call__(self, encoding: Encoding, pair_encoding: Encoding, add_special_tokens: bool):
        return self._postprocessor(encoding, pair_encoding, add_special_tokens)


class BertPostProcessor(PostProcessor):
    def __init__(self, sep: Tuple[str, int] = ("[SEP]", 102), cls: Tuple[str, int] = ("[CLS]", 101)):
        self._postprocessor = C.postprocessors.BertPostProcessor(sep, cls)


class RobertaPostProcessor(PostProcessor):
    def __init__(
        self,
        sep: Tuple[str, int] = ("</s>", 2),
        cls: Tuple[str, int] = ("<s>", 0),
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ):
        self._postprocessor = C.postprocessors.RobertaPostProcessor(sep, cls, trim_offsets, add_prefix_space)


class ByteLevelPostProcessor(PostProcessor):
    def __init__(self, add_prefix_space: bool = True, trim_offsets: bool = True, use_regex: bool = True):
        self._postprocessor = C.postprocessors.ByteLevelPostProcessor(add_prefix_space, trim_offsets, use_regex)


class TemplatePostProcessor(PostProcessor):
    def __init__(
        self, single: Union[str, List[str]], pair: Union[str, List[str]], special_tokens: List[Tuple[str, int]]
    ):
        self._postprocessor = C.postprocessors.TemplatePostProcessor(single, pair, special_tokens)
