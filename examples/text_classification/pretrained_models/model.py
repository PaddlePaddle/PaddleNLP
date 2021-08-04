import paddle
import paddle.fluid.core as core
import paddle.nn as nn
import paddlenlp
from paddlenlp.transformers import BertForSequenceClassification
from paddlenlp.layers import FastTokenizer


class FastBertForSequenceClassification(nn.Layer):
    def __init__(self, vocab_path, num_classes):
        super(FastBertForSequenceClassification, self).__init__()
        self.bert_cls = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", num_classes=num_classes)
        self.tokenizer = FastTokenizer(vocab_path)

    def forward(self, text):
        input_ids, token_type_ids = self.tokenizer(text)
        logits = self.bert_cls(input_ids, token_type_ids)

        # if paddle.isnan(paddle.sum(logits)):
        #     print("*"*10)
        #     print(text)
        #     print(input_ids)
        #     print(token_type_ids)
        return logits
