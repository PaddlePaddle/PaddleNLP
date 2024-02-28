# Copyright (c) 2023 Technology Innovation Institute (TII) and PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer

__all__ = ["RWTokenizer"]


class RWTokenizer(GPTTokenizer):
    """
    Constructs a RWModel tokenizer based on byte-level Byte-Pair-Encoding.

    This tokenizer inherits from :class:`~paddlenlp.transformers.GPTTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            Path to the vocab file.
            The vocab file contains a mapping from vocabulary strings to indices.
        merges_file (str):
            Path to the merge file.
            The merge file is used to split the input sentence into "subword" units.
            The vocab file is then used to encode those units as intices.
        errors (str):
            Paradigm to follow when decoding bytes to UTF-8.
            Defaults to `'replace'`.
        max_len (int, optional):
            The maximum value of the input sequence length.
            Defaults to `None`.

    Examples:
        .. code-block::

            from paddlenlp.transformers import RWTokenizer

            tokenizer = RWTokenizer.from_pretrained('tiiuae/falcon-7b')
            print(tokenizer('Welcome to use PaddlePaddle and PaddleNLP'))

            '''
            {'input_ids': [11302, 271, 745, 337, 18849, 59, 18849, 273, 337, 18849, 57, 15549]}
            '''

    """

    resource_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # for save_pretrained
    model_input_names = ["input_ids"]

    pretrained_resource_files_map = {
        "vocab_file": {
            "falcon-7b": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b/vocab.json",
            "falcon-7b-instruct": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b-instruct/vocab.json",
            "OpenBuddy/openbuddy-falcon-7b-v5-fp16": "https://bj.bcebos.com/paddlenlp/models/community/OpenBuddy/openbuddy-falcon-7b-v5-fp16/vocab.json",
        },
        "merges_file": {
            "falcon-7b": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b/merges.txt",
            "falcon-7b-instruct": "https://bj.bcebos.com/paddlenlp/models/community/tiiuae/falcon-7b-instruct/merges.txt",
            "OpenBuddy/openbuddy-falcon-7b-v5-fp16": "https://bj.bcebos.com/paddlenlp/models/community/OpenBuddy/openbuddy-falcon-7b-v5-fp16/merges.txt",
        },
    }
    padding_side = "right"

    def __init__(self, vocab_file, merges_file, **kwargs):  # The token of newline.
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = RWTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )

        self.spaces_between_special_tokens = kwargs.get("spaces_between_special_tokens", True)
        super().__init__(vocab_file, merges_file, **kwargs)

    def decode(
        self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = True, **kwargs
    ) -> str:
        return super(RWTokenizer, self).decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            **kwargs,
        )
