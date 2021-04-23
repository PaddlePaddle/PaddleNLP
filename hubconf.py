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

dependencies = ['paddle']

from paddlenlp.transformers import BertModel, BertForSequenceClassification
from paddlenlp.transformers import BertForTokenClassification, BertForQuestionAnswering
from paddlenlp.transformers import BertForPretraining
from paddlenlp.transformers import BertTokenizer

BERT_MODEL_CLASSES = {
    "bert": BertModel,
    "sequence_classification": BertForSequenceClassification,
    "token_classification": BertForTokenClassification,
    "question_answering": BertForQuestionAnswering,
    "pretrain": BertForPretraining
}


def bert(model_name_or_path='bert-base-uncased',
         model_select='sequence_classification'):
    """
    Returns BERT model from given pretrained model.

    Args:
        model_name_or_path (str, optional):  A name of or a file path to a
            pretrained model. It could be 'bert-base-uncased',
            'bert-large-uncased', 'bert-base-multilingual-uncased',
            'bert-base-cased', 'bert-base-chinese', 'bert-large-cased',
            'bert-base-multilingual-cased', 'bert-wwm-chinese' or
            'bert-wwm-ext-chinese'. Default: 'bert-base-uncased'.
        model_select (str, optional): model class to select. It could be
            'bert', 'sequence_classification', 'token_classification',
            'question_answering' or 'pretrain'. If 'sequence_classification'
            is chosen, model class would be `BertForSequenceClassification`.
            The document of BERT model could be seen at `bert.modeling
            <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.bert.modeling.html>`_
            Default: 'sequence_classification'.
    
    Returns:
        tuple: Returns the pretrained bert model and bert tokenizer.

    Example:

        .. code-block:: python

          import paddle.hub as hub

          model, tokenizer = hub.load('PaddlePaddle/PaddleNLP:develop', model='bert')

    """
    model_class = BERT_MODEL_CLASSES[model_select]

    model = model_class.from_pretrained(model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

    return model, tokenizer
