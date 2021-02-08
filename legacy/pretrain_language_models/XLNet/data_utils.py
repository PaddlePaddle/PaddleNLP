"""this file is a copy of https://github.com/zihangdai/xlnet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}

VOCAB_SIZE = 32000
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]
