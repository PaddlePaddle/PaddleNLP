# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import _locale
import jieba
from subword_nmt import subword_nmt

# By default, the Windows system opens the file with GBK code,
# and the subword_nmt package does not support setting open encoding,
# so it is set to UTF-8 uniformly.
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])


class STACLTokenizer:

    def __init__(self, bpe_dict, is_chinese):
        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', bpe_dict])
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)
        self.is_chinese = is_chinese

    def tokenize(self, raw_string):
        """
        Tokenize string(BPE/jieba+BPE)
        """
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string
        if self.is_chinese:
            raw_string = ' '.join(jieba.cut(raw_string))
        bpe_str = self.bpe.process_line(raw_string)
        return ' '.join(bpe_str.split())


if __name__ == '__main__':
    tokenizer_zh = STACLTokenizer('data/nist2m/2M.zh2en.dict4bpe.zh',
                                  is_chinese=True)
    print(tokenizer_zh.tokenize('玻利维亚举行总统与国会选举'))
