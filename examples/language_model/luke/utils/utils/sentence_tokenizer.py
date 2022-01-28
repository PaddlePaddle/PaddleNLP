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
"""sentence tokenizer file"""
from typing import List, Tuple
import pkg_resources


class SentenceTokenizer:
    """ Base class for all sentence tokenizers in this project."""

    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """span token"""
        raise NotImplementedError

    @classmethod
    def from_name(cls, name: str):
        """name fun"""

        if name == "opennlp":
            return OpenNLPSentenceTokenizer()
        return ICUSentenceTokenizer(name)


class ICUSentenceTokenizer:
    """ Segment text to sentences. """

    def __init__(self, locale="en"):
        """init fun"""
        from icu import Locale, BreakIterator

        # ICU includes lists of common abbreviations that can be used to filter, to ignore,
        # these false sentence boundaries for some languages.
        # (http://userguide.icu-project.org/boundaryanalysis)
        if locale in {"en", "de", "es", "it", "pt"}:
            locale += "@ss=standard"
        self.locale = Locale(locale)
        self.breaker = BreakIterator.createSentenceInstance(self.locale)

    def span_tokenize(self, text: str):
        """
        ICU's BreakIterator gives boundary indices by counting *codeunits*, not *codepoints*.
        (https://stackoverflow.com/questions/30775689/python-length-of-unicode-string-confusion)
        As a result, something like this can happen.

        ```
        text = "󰡕test."  󰡕 is a non-BMP (Basic Multilingual Plane) character, which consists of two codeunits.
        len(text)
        >>> 6
        icu_tokenizer.span_tokenize(text)
        >>> [(0, 7)]
        ```

        This results in undesirable bugs in following stages.
        So, we have decided to replace non-BMP characters with an arbitrary BMP character, and then run BreakIterator.
        """

        # replace non-BMP characters with a whitespace
        # (https://stackoverflow.com/questions/36283818/remove-characters-outside-of-the-bmp-emojis-in-python-3)
        text = "".join(c if c <= "\uFFFF" else " " for c in text)

        self.breaker.setText(text)
        start_idx = 0
        spans = []
        for end_idx in self.breaker:
            spans.append((start_idx, end_idx))
            start_idx = end_idx
        return spans


class OpenNLPSentenceTokenizer(SentenceTokenizer):
    """open sentence token"""

    _java_initialized = False

    def __init__(self):
        """init fun"""
        self._initialized = False

    def __reduce__(self):
        """reduce fun"""
        return self.__class__, tuple()

    def initialize(self):
        """init fun"""
        # we need to delay the initialization of Java in order for this class to
        # properly work with multiprocessing
        if not OpenNLPSentenceTokenizer._java_initialized:
            import jnius_config

            jnius_config.add_options("-Xrs")
            jnius_config.set_classpath(pkg_resources.resource_filename(__name__, "/resources/opennlp-tools-1.5.3.jar"))
            OpenNLPSentenceTokenizer._java_initialized = True

        from jnius import autoclass

        File = autoclass("java.io.File")
        SentenceModel = autoclass("opennlp.tools.sentdetect.SentenceModel")
        SentenceDetectorME = autoclass("opennlp.tools.sentdetect.SentenceDetectorME")

        sentence_model_file = pkg_resources.resource_filename(__name__, "resources/en-sent.bin")
        sentence_model = SentenceModel(File(sentence_model_file))
        self._tokenizer = SentenceDetectorME(sentence_model)

        self._initialized = True

    def span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        """span token"""
        if not self._initialized:
            self.initialize()

        # replace non-BMP characters with a whitespace
        # (https://stackoverflow.com/questions/36283818/remove-characters-outside-of-the-bmp-emojis-in-python-3)
        text = "".join(c if c <= "\uFFFF" else " " for c in text)

        return [(span.getStart(), span.getEnd()) for span in self._tokenizer.sentPosDetect(text)]
