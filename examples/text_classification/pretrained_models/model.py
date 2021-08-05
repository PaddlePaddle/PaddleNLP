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
        inputs_ids = input_ids.pin_memory()
        token_type_ids = token_type_ids.pin_memory()
        logits = self.bert_cls(input_ids, token_type_ids)

        # check_input_ids = input_ids.numpy()[0]
        # check_token_type_ids = token_type_ids.numpy()[0]

        # encoded_inputs = self.test_tokenizer(text, max_seq_len=128, is_split_into_words=False)[0]
        # test_input_ids = np.array(encoded_inputs["input_ids"])
        # test_token_type_ids = np.array(encoded_inputs["token_type_ids"])

        # diff = check_input_ids - test_input_ids
        # if not np.array_equal(test_input_ids, check_input_ids):
        #     print("np.array_equal(test_input_ids, check_input_ids) "*5)
        #     print(test_input_ids)
        #     print(check_input_ids)

        # assert np.array_equal(test_input_ids, check_input_ids)
        # if (diff.sum() !=0):
        #     print("diff is not 0. "*10)
        #     print(text)
        #     print(input_ids)
        #     print(token_type_ids)

        # if paddle.isnan(paddle.sum(logits)):
        #     print("*"*10)
        #     print(text)
        #     print(input_ids)
        #     print(token_type_ids)
        return logits
