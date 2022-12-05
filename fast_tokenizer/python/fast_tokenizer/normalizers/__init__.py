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


class NormalizedString:
    def __init__(self, raw_str: str):
        self._normalized = C.normalizers.NormalizedString(raw_str)

    def __str__(self):
        return str(self._normalized)


class Normalizer(ABC):
    def normalize_str(self, sequence: str):
        return self._normalizer.normalize_str(sequence)

    def __call__(self, normalized: NormalizedString):
        return self._normalizer(normalized._normalized)

    def __getstate__(self):
        return self._normalizer.__getstate__()


class BertNormalizer(Normalizer):
    def __init__(
        self,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool = True,
        lowercase: bool = True,
    ):
        self._normalizer = C.normalizers.BertNormalizer(clean_text, handle_chinese_chars, strip_accents, lowercase)


class ReplaceNormalizer(Normalizer):
    def __init__(self, pattern: str, content: str):
        self._normalizer = C.normalizers.ReplaceNormalizer(pattern, content)


class StripNormalizer(Normalizer):
    def __init__(self, left: bool = True, right: bool = True):
        self._normalizer = C.normalizers.StripNormalizer(left, right)


class StripAccentsNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.StripAccentsNormalizer()


class NFCNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.NFCNormalizer()


class NFDNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.NFDNormalizer()


class NFKCNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.NFKCNormalizer()


class NFKDNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.NFKDNormalizer()


class NmtNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.NmtNormalizer()


class LowercaseNormalizer(Normalizer):
    def __init__(self):
        self._normalizer = C.normalizers.LowercaseNormalizer()


class SequenceNormalizer(Normalizer):
    def __init__(self, normalizer_list=[]):
        normalizer_list = [normalizer._normalizer for normalizer in normalizer_list]
        self._normalizer = C.normalizers.SequenceNormalizer(normalizer_list)


class PrecompiledNormalizer(Normalizer):
    def __init__(self, precompiled_charsmap: str):
        self._normalizer = C.normalizers.PrecompiledNormalizer(precompiled_charsmap)
