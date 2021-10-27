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

import paddle.nn as nn
from paddlenlp.experimental import FasterTokenizer
from paddlenlp.transformers import BertForSequenceClassification, BertForTokenClassification
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieForTokenClassification
from paddlenlp.transformers import RobertaForSequenceClassification, RobertaForTokenClassification
from paddlenlp.transformers import BertTokenizer, ErnieTokenizer, RobertaTokenizer

__all__ = ["FastSequenceClassificationModel", "FastTokenClassificationModel"]


class FasterPretrainedModel(nn.Layer):
    def __init__(self,
                 vocab,
                 model,
                 do_lower_case=False,
                 is_split_into_words=False,
                 max_seq_len=512,
                 pad_to_max_seq_len=True):
        super(FasterPretrainedModel, self).__init__()
        self.tokenizer = FasterTokenizer(
            vocab,
            do_lower_case=do_lower_case,
            is_split_into_words=is_split_into_words)
        self.model = model
        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = pad_to_max_seq_len

    def forward(self, text, text_pair=None):
        input_ids, token_type_ids = self.tokenizer(
            text=text,
            text_pair=text_pair,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len)

        logits = self.model(input_ids, token_type_ids)
        return logits


class FastSequenceClassificationModel(object):
    name_model = {
        'bert-base-uncased': (BertForSequenceClassification, BertTokenizer),
        'bert-large-uncased': (BertForSequenceClassification, BertTokenizer),
        'bert-base-multilingual-uncased':
        (BertForSequenceClassification, BertTokenizer),
        'bert-base-cased': (BertForSequenceClassification, BertTokenizer),
        'bert-base-chinese': (BertForSequenceClassification, BertTokenizer),
        'bert-base-multilingual-cased':
        (BertForSequenceClassification, BertTokenizer),
        'bert-large-cased': (BertForSequenceClassification, BertTokenizer),
        'bert-wwm-chinese': (BertForSequenceClassification, BertTokenizer),
        'bert-wwm-ext-chinese': (BertForSequenceClassification, BertTokenizer),
        'macbert-base-chinese': (BertForSequenceClassification, BertTokenizer),
        'macbert-large-chinese': (BertForSequenceClassification, BertTokenizer),
        'ernie-1.0': (ErnieForSequenceClassification, ErnieTokenizer),
        'ernie-2.0-en': (ErnieForSequenceClassification, ErnieTokenizer),
        'ernie-2.0-en-finetuned-squad':
        (ErnieForSequenceClassification, ErnieTokenizer),
        'ernie-2.0-large-en': (ErnieForSequenceClassification, ErnieTokenizer),
        'roberta-wwm-ext': (RobertaForSequenceClassification, RobertaTokenizer),
        'roberta-wwm-ext-large':
        (RobertaForSequenceClassification, RobertaTokenizer),
        'rbt3': (RobertaForSequenceClassification, RobertaTokenizer),
        'rbtl3': (RobertaForSequenceClassification, RobertaTokenizer),
    }

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        num_classes,
                        max_seq_len=512,
                        pad_to_max_seq_len=True,
                        **kwargs):
        if pretrained_model_name_or_path in cls.name_model:
            model_cls, tokenizer_cls = cls.name_model[
                pretrained_model_name_or_path]
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path,
                num_classes=num_classes,
                **kwargs)
            tokenizer = tokenizer_cls.from_pretrained(
                pretrained_model_name_or_path)

            return FasterPretrainedModel(
                tokenizer.vocab.token_to_idx,
                model,
                do_lower_case=tokenizer.do_lower_case,
                is_split_into_words=False,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=pad_to_max_seq_len)
        else:
            raise ValueError("Unknown name %s. Now %s surports  %s" %
                             (pretrained_model_name_or_path, cls.__name__,
                              list(name_model.keys())))


class FastTokenClassificationModel(object):
    name_model = {
        'bert-base-uncased': (BertForTokenClassification, BertTokenizer),
        'bert-large-uncased': (BertForTokenClassification, BertTokenizer),
        'bert-base-multilingual-uncased':
        (BertForTokenClassification, BertTokenizer),
        'bert-base-cased': (BertForTokenClassification, BertTokenizer),
        'bert-base-chinese': (BertForTokenClassification, BertTokenizer),
        'bert-base-multilingual-cased':
        (BertForTokenClassification, BertTokenizer),
        'bert-large-cased': (BertForTokenClassification, BertTokenizer),
        'bert-wwm-chinese': (BertForTokenClassification, BertTokenizer),
        'bert-wwm-ext-chinese': (BertForTokenClassification, BertTokenizer),
        'macbert-base-chinese': (BertForTokenClassification, BertTokenizer),
        'macbert-large-chinese': (BertForTokenClassification, BertTokenizer),
        'ernie-1.0': (ErnieForTokenClassification, ErnieTokenizer),
        'ernie-2.0-en': (ErnieForTokenClassification, ErnieTokenizer),
        'ernie-2.0-en-finetuned-squad':
        (ErnieForTokenClassification, ErnieTokenizer),
        'ernie-2.0-large-en': (ErnieForTokenClassification, ErnieTokenizer),
        'roberta-wwm-ext': (RobertaForTokenClassification, RobertaTokenizer),
        'roberta-wwm-ext-large':
        (RobertaForTokenClassification, RobertaTokenizer),
        'rbt3': (RobertaForTokenClassification, RobertaTokenizer),
        'rbtl3': (RobertaForTokenClassification, RobertaTokenizer),
    }

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        num_classes,
                        max_seq_len=512,
                        pad_to_max_seq_len=True,
                        **kwargs):
        if pretrained_model_name_or_path in cls.name_model:
            model_cls, tokenizer_cls = cls.name_model[
                pretrained_model_name_or_path]
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path,
                num_classes=num_classes,
                **kwargs)
            tokenizer = tokenizer_cls.from_pretrained(
                pretrained_model_name_or_path)

            return FasterPretrainedModel(
                tokenizer.vocab.token_to_idx,
                model,
                do_lower_case=tokenizer.do_lower_case,
                is_split_into_words=False,
                max_seq_len=max_seq_len,
                pad_to_max_seq_len=pad_to_max_seq_len)
        else:
            raise ValueError("Unknown name %s. Now %s surports  %s" %
                             (pretrained_model_name_or_path, cls.__name__,
                              list(name_model.keys())))
