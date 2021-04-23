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

dependencies = ['paddle', 'paddlenlp']

from paddlenlp.transformers import BertModel, BertForSequenceClassification
from paddlenlp.transformers import BertForTokenClassification, BertForQuestionAnswering
from paddlenlp.transformers import BertForPretraining, BertPretrainingCriterion
from paddlenlp.transformers import BertTokenizer

MODEL_CLASSES = {
    "bert": BertModel,
    "sequence_classification": BertForSequenceClassification,
    "token_classification": BertForTokenClassification,
    "question_answering": BertForQuestionAnswering,
    "pretrain": BertForPretraining
}


def bert(model_name='bert-base-uncased',
         model_select='sequence_classification'):
    model_class = MODEL_CLASSES[model_select]

    model = model_class.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    return model, tokenizer
