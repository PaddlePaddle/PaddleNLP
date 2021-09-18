import paddle
import paddle.fluid.core as core
import paddle.nn as nn
import paddlenlp
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from paddlenlp.layers import FastTokenizer

import numpy as np


class FastBertForSequenceClassification(nn.Layer):
    def __init__(self, vocab_path, num_classes):
        super(FastBertForSequenceClassification, self).__init__()
        self.bert_cls = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", num_classes=num_classes)
        self.tokenizer = FastTokenizer(vocab_path)

    def forward(self, text):
        input_ids, token_type_ids = self.tokenizer(text, max_seq_len=128)
        # paddle.static.Print(input_ids, message='input_ids')
        logits = self.bert_cls(input_ids, token_type_ids)
        return logits