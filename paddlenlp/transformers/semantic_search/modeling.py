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

__all__ = ["ErnieDualEncoder", "ErnieCrossEncoder"]


class ErnieEncoder(ErniePretrainedModel):
    def __init__(self, ernie, dropout=None, output_emb_size=None, num_classes=2):
        super(ErnieEncoder, self).__init__()
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_classes)
        # Compatible to ERNIE-Search for adding extra linear layer
        self.output_emb_size = output_emb_size
        if output_emb_size is not None and output_emb_size > 0:
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(
                self.ernie.config["hidden_size"], output_emb_size, weight_attr=weight_attr
            )
        self.apply(self.init_weights)

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, pool_output = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        return sequence_output, pool_output


class ErnieDualEncoder(nn.Layer):
    """
    This class encapsulates two ErnieEncoder models into one model, so query
    embedding and title embedding could be obtained using one model. And this
    class allows two ErnieEncoder models to be trained at the same time.

    Example:

        .. code-block::

            import paddle
            from paddlenlp.transformers import ErnieDualEncoder, ErnieTokenizer

            model = ErnieDualEncoder("rocketqa-zh-dureader-query-encoder", "rocketqa-zh-dureader-para-encoder")
            tokenizer = ErnieTokenizer.from_pretrained("rocketqa-zh-dureader-query-encoder")

            inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

            # Get query embedding
            query_embedding = model.get_pooled_embedding(**inputs)

            # Get title embedding
            title_embedding = model.get_pooled_embedding(**inputs, is_query=False)

    """

    def __init__(
        self,
        query_model_name_or_path=None,
        title_model_name_or_path=None,
        share_parameters=False,
        output_emb_size=None,
        dropout=None,
        reinitialize=False,
        use_cross_batch=False,
    ):

        super().__init__()
        self.query_ernie, self.title_ernie = None, None
        self.use_cross_batch = use_cross_batch
        self.output_emb_size = output_emb_size
        if query_model_name_or_path is not None:
            self.query_ernie = ErnieEncoder.from_pretrained(query_model_name_or_path, output_emb_size=output_emb_size)
        if share_parameters:
            self.title_ernie = self.query_ernie
        elif title_model_name_or_path is not None:
            self.title_ernie = ErnieEncoder.from_pretrained(title_model_name_or_path, output_emb_size=output_emb_size)
        assert (self.query_ernie is not None) or (
            self.title_ernie is not None
        ), "At least one of query_ernie and title_ernie should not be None"

        # Compatible to rocketv2 initialization for setting layer._epsilon to 1e-5
        if reinitialize:
            self.apply(self.init_weights)

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5

    def get_semantic_embedding(self, data_loader):
        self.eval()
        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)

                text_embeddings = self.get_pooled_embedding(input_ids, token_type_ids=token_type_ids)

                yield text_embeddings

    def get_pooled_embedding(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, is_query=True
    ):
        """Get the first feature of each sequence for classification"""
        assert (is_query and self.query_ernie is not None) or (
            not is_query and self.title_ernie
        ), "Please check whether your parameter for `is_query` are consistent with DualEncoder initialization."
        if is_query:
            sequence_output, _ = self.query_ernie(input_ids, token_type_ids, position_ids, attention_mask)
            if self.output_emb_size is not None and self.output_emb_size > 0:
                cls_embedding = self.query_ernie.emb_reduce_linear(sequence_output[:, 0])
            else:
                cls_embedding = sequence_output[:, 0]

        else:
            sequence_output, _ = self.title_ernie(input_ids, token_type_ids, position_ids, attention_mask)
            if self.output_emb_size is not None and self.output_emb_size > 0:
                cls_embedding = self.title_ernie.emb_reduce_linear(sequence_output[:, 0])
            else:
                cls_embedding = sequence_output[:, 0]
        return cls_embedding

    def cosine_sim(
        self,
        query_input_ids,
        title_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        title_token_type_ids=None,
        title_position_ids=None,
        title_attention_mask=None,
    ):
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        title_cls_embedding = self.get_pooled_embedding(
            title_input_ids, title_token_type_ids, title_position_ids, title_attention_mask, is_query=False
        )

        cosine_sim = paddle.sum(query_cls_embedding * title_cls_embedding, axis=-1)
        return cosine_sim

    def forward(
        self,
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
        neg_title_attention_mask=None,
    ):
        query_cls_embedding = self.get_pooled_embedding(
            query_input_ids, query_token_type_ids, query_position_ids, query_attention_mask
        )

        pos_title_cls_embedding = self.get_pooled_embedding(
            pos_title_input_ids,
            pos_title_token_type_ids,
            pos_title_position_ids,
            pos_title_attention_mask,
            is_query=False,
        )

        neg_title_cls_embedding = self.get_pooled_embedding(
            neg_title_input_ids,
            neg_title_token_type_ids,
            neg_title_position_ids,
            neg_title_attention_mask,
            is_query=False,
        )

        all_title_cls_embedding = paddle.concat(x=[pos_title_cls_embedding, neg_title_cls_embedding], axis=0)

        if is_prediction:
            logits = paddle.dot(query_cls_embedding, pos_title_cls_embedding)
            outputs = {"probs": logits, "q_rep": query_cls_embedding, "p_rep": pos_title_cls_embedding}
            return outputs

        if self.use_cross_batch:
            tensor_list = []
            paddle.distributed.all_gather(tensor_list, all_title_cls_embedding)
            all_title_cls_embedding = paddle.concat(x=tensor_list, axis=0)

        logits = paddle.matmul(query_cls_embedding, all_title_cls_embedding, transpose_y=True)

        batch_size = query_cls_embedding.shape[0]

        labels = paddle.arange(batch_size * self.rank * 2, batch_size * (self.rank * 2 + 1), dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        accuracy = paddle.metric.accuracy(input=logits, label=labels)
        loss = F.cross_entropy(input=logits, label=labels)
        outputs = {"loss": loss, "accuracy": accuracy}

        return outputs


class ErnieCrossEncoder(nn.Layer):
    """
    Example:

        .. code-block::

            import paddle
            from paddlenlp.transformers import ErnieCrossEncoder, ErnieTokenizer

            model = ErnieCrossEncoder("rocketqa-zh-dureader-cross-encoder")
            tokenizer = ErnieTokenizer.from_pretrained("rocketqa-zh-dureader-cross-encoder")

            inputs = tokenizer("你们好", text_pair="你好")
            inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}

            # Get embedding of text pair.
            embedding = model.matching(**inputs)

    """

    def __init__(self, pretrain_model_name_or_path, num_classes=2, reinitialize=False, dropout=None):
        super().__init__()

        self.ernie = ErnieEncoder.from_pretrained(pretrain_model_name_or_path, num_classes=num_classes)
        # Compatible to rocketv2 initialization for setting layer._epsilon to 1e-5
        if reinitialize:
            self.apply(self.init_weights)

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-5

    def matching(
        self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, return_prob_distributation=False
    ):
        """Use the pooled_output as the feature for pointwise prediction, eg. RocketQAv1"""
        _, pooled_output = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        pooled_output = self.ernie.dropout(pooled_output)
        cls_embedding = self.ernie.classifier(pooled_output)
        probs = F.softmax(cls_embedding, axis=1)
        if return_prob_distributation:
            return probs
        return probs[:, 1]

    def matching_v2(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """Use the cls token embedding as the feature for listwise prediction, eg. RocketQAv2"""
        sequence_output, _ = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        pooled_output = self.ernie.dropout(sequence_output[:, 0])
        probs = self.ernie.classifier(pooled_output)
        return probs

    def matching_v3(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        """Use the pooled_output as the feature for listwise prediction, eg. ERNIE-Search"""
        sequence_output, pooled_output = self.ernie(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        pooled_output = self.ernie.dropout(pooled_output)
        probs = self.ernie.classifier(pooled_output)
        return probs

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, labels=None):
        probs = self.matching(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            return_prob_distributation=True,
        )
        if labels is not None:
            accuracy = paddle.metric.accuracy(input=probs, label=labels)
            loss = F.cross_entropy(input=probs, label=labels)
            outputs = {"loss": loss, "accuracy": accuracy}
            return outputs
        else:
            return probs[:, 1]
