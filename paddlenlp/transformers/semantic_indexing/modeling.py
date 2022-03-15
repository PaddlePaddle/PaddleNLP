# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..ernie.modeling import ErniePretrainedModel

__all__ = [
    'ErnieSemanticIndexingPretrainedModel', 'ErnieForRespresent', 'DualEncoder'
]


class ErnieSemanticIndexingPretrainedModel(ErniePretrainedModel):
    """
    A class for pretrained ERNIE models for matching. It provides ERNIE related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models. 

    Refer to :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for
    more details.
    """

    pretrained_init_configuration = {
        "ernie-base-cn-query": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
        },
        "ernie-base-cn-title": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "relu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 513,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 18000,
            "pad_token_id": 0,
        },
        "ernie-base-en-query": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
        "ernie-base-en-title": {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 4,
            "vocab_size": 30522,
            "pad_token_id": 0,
        },
    }
    pretrained_resource_files_map = {
        "model_state": {
            "ernie-base-cn-query":
            "https://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie_base_cn_query.pdparams",
            "ernie-base-cn-title":
            "https://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie_base_cn_title.pdparams",
            "ernie-base-en-query":
            "https://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie_base_en_query.pdparams",
            "ernie-base-en-title":
            "https://bj.bcebos.com/paddlenlp/models/transformers/semantic_indexing/ernie_base_en_title.pdparams",
        }
    }

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5


class ErnieForRespresent(ErnieSemanticIndexingPretrainedModel):
    """
    Example:

        .. code-block::

            from paddlenlp.transformers import ErnieMatchingModel

            query_model = ErnieMatchingModel.from_pretrained("ernie-base-cn-query")
            title_model = ErnieMatchingModel.from_pretrained("ernie-base-cn-title")
        
            inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
    
            query_embedding = query_model(**inputs)
            title_embedding = title_model(**inputs)
    """

    def __init__(self, ernie):
        super(ErnieForRespresent, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        sequence_output, _ = self.ernie(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        # Outputs pooled_embedding
        pooled_output = sequence_output[:, 0]
        return pooled_output


class DualEncoder(nn.Layer):
    """
    This class encapsulates two ERNIE matching models into one model, so query
    embedding and title embedding could be obtained using one model. And this
    class allows two ERNIE matching models to be trained at the same time.

    Example:

        .. code-block::

            from paddlenlp.transformers import ErnieMatchingModel, DualErnieForMatching
        
            query_model = ErnieMatchingModel.from_pretrained("ernie-base-matching-query")
            title_model = ErnieMatchingModel.from_pretrained("ernie-base-matching-title")
        
            model = DualErnieForMatching(query_model, title_model, output_emb_size=512)

            inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

            # Get query embedding
            query_embedding = model.get_pooled_embedding(**inputs)

            # Get title embedding
            title_embedding = model.get_pooled_embedding(**inputs, is_query=False)

    """

    def __init__(self,
                 query_pretrained_model,
                 title_pretrained_model,
                 dropout=None,
                 output_emb_size=None,
                 use_cross_batch=False):
        super().__init__()
        self.query_ernie = query_pretrained_model
        self.title_ernie = title_pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                text_embeddings = self.get_pooled_embedding(
                    input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def get_pooled_embedding(self,
                             input_ids,
                             token_type_ids=None,
                             position_ids=None,
                             attention_mask=None,
                             is_query=True):
        if is_query:
            pooled_embedding = self.query_ernie(input_ids, token_type_ids,
                                                position_ids, attention_mask)
        else:
            pooled_embedding = self.title_ernie(input_ids, token_type_ids,
                                                position_ids, attention_mask)
        return pooled_embedding

    def cosine_sim(self,
                   query_input_ids,
                   title_input_ids,
                   query_token_type_ids=None,
                   query_position_ids=None,
                   query_attention_mask=None,
                   title_token_type_ids=None,
                   title_position_ids=None,
                   title_attention_mask=None):

        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids,
            title_token_type_ids,
            title_position_ids,
            title_attention_mask,
            is_query=False)

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding,
                                axis=-1)
        return cosine_sim

    def forward(self,
                query_input_ids,
                pos_title_input_ids,
                neg_title_input_ids,
                is_prediction=False,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                pos_title_token_type_ids=None,
                pos_title_position_ids=None,
                pos_title_attention_mask=None,
                neg_title_token_type_ids=None,
                neg_title_position_ids=None,
                neg_title_attention_mask=None):
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids,
            query_attention_mask)

        pos_title_cls_embedding = self.get_pooled_embedding(
            pos_title_input_ids, pos_title_token_type_ids,
            pos_title_position_ids, pos_title_attention_mask)

        neg_title_cls_embedding = self.get_pooled_embedding(
            neg_title_input_ids, neg_title_token_type_ids,
            neg_title_position_ids, neg_title_attention_mask)

        all_title_cls_embedding = paddle.concat(
            x=[pos_title_cls_embedding, neg_title_cls_embedding], axis=0)

        if is_prediction:
            logits = paddle.dot(query_cls_embedding, pos_title_cls_embedding)
            outputs = {
                "probs": logits,
                "q_rep": query_cls_embedding,
                "p_rep": pos_title_cls_embedding
            }
            return outputs

        if self.use_cross_batch:
            tensor_list = []
            paddle.distributed.all_gather(tensor_list, all_title_cls_embedding)
            all_title_cls_embedding = paddle.concat(x=tensor_list, axis=0)

        # multiply
        logits = paddle.matmul(
            query_cls_embedding, all_title_cls_embedding, transpose_y=True)

        batch_size = query_cls_embedding.shape[0]

        labels = paddle.arange(
            batch_size * self.rank * 2,
            batch_size * (self.rank * 2 + 1),
            dtype='int64')
        labels = paddle.reshape(labels, shape=[-1, 1])

        accuracy = paddle.metric.accuracy(input=logits, label=labels)
        loss = F.cross_entropy(input=logits, label=labels)
        outputs = {"loss": loss, "accuracy": accuracy}

        return outputs
