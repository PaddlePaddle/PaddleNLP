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

__version__ = "0.1.1"

from typing import Tuple, Union, Tuple, List

TextInputSequence = str
PreTokenizedInputSequence = Union[List[str], Tuple[str]]

TextEncodeInput = Union[TextInputSequence, Tuple[
    TextInputSequence, TextInputSequence], List[TextInputSequence], ]

PreTokenizedEncodeInput = Union[PreTokenizedInputSequence, Tuple[
    PreTokenizedInputSequence, PreTokenizedInputSequence], List[
        PreTokenizedInputSequence], ]

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]

EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]

from .core_tokenizers import (Tokenizer, Encoding, AddedToken, Token, PadMethod,
                              TruncMethod, OffsetType, Direction, TruncStrategy,
                              PadStrategy)
from .core_tokenizers import models, normalizers, pretokenizers, postprocessors, decoders

from .tokenizers_impl import ErnieFasterTokenizer
