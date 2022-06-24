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


import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import AutoTokenizer, AutoModel, ErnieForMaskedLM

from data import mask_tokens
from custom_ernie import ErnieModel as CustomErnie


class Extractor(nn.Layer):
    def __init__(self, pretrained_model_name, dropout=None, margin=0.0, scale=20, output_emb_size=None):
        super().__init__()
        self.ptm = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        # if output_emb_size is greater than 0, then add Linear layer to reduce embedding_size
        self.output_emb_size = output_emb_size
        if output_emb_size > 0:
            weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
            self.emb_reduce_linear = paddle.nn.Linear(768, output_emb_size, weight_attr=weight_attr)

        self.margin = margin
        # Used scaling cosine similarity to ease converge
        self.scale = scale

    def get_pooled_embedding(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, with_pooler=False): 
        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        if not with_pooler:
            ori_cls_embedding = sequence_output[:, 0, :]
        if self.output_emb_size > 0:
            cls_embedding = self.emb_reduce_linear(ori_cls_embedding)

        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = F.normalize(cls_embedding, p=2, axis=-1)
        return cls_embedding, ori_cls_embedding

    def cosine_sim(
        self,
        query_input_ids,
        key_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        key_token_type_ids=None,
        key_position_ids=None,
        key_attention_mask=None,
        with_pooler=False
    ):
        query_cls_embedding, _ = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler
        )
        key_cls_embedding, _ = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler
        )

        cosine_sim = paddle.sum(query_cls_embedding * key_cls_embedding, axis=-1)
        return cosine_sim

    def forward(
        self,
        query_input_ids,
        key_input_ids,
        query_token_type_ids=None,
        query_position_ids=None,
        query_attention_mask=None,
        key_token_type_ids=None,
        key_position_ids=None,
        key_attention_mask=None,
        with_pooler=False
    ):
        query_cls_embedding, ori_query_cls_embedding = self.get_pooled_embedding(
            query_input_ids,
            query_token_type_ids,
            query_position_ids,
            query_attention_mask,
            with_pooler=with_pooler
        )
        key_cls_embedding, ori_key_cls_embedding = self.get_pooled_embedding(
            key_input_ids,
            key_token_type_ids,
            key_position_ids,
            key_attention_mask,
            with_pooler=with_pooler
        )
        cosine_sim = paddle.matmul(query_cls_embedding, key_cls_embedding, transpose_y=True)

        # substract margin from all positive samples cosine_sim()
        margin_diag = paddle.full(
            shape=[query_cls_embedding.shape[0]],
            fill_value=self.margin,
            dtype=paddle.get_default_dtype()
        )

        cosine_sim = cosine_sim - paddle.diag(margin_diag)

        # scale cosine to ease training converge
        cosine_sim *= self.scale

        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])

        loss = F.cross_entropy(input=cosine_sim, label=labels)
        ori_cls_embedding = paddle.concat([ori_query_cls_embedding, ori_key_cls_embedding], axis=0)
        return loss, ori_cls_embedding

class Discriminator(nn.Layer):
    def __init__(self, ptm_model_name):
        super(Discriminator, self).__init__()
        self.ptm = CustomErnie.from_pretrained(ptm_model_name)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
    
    def forward(self, input_ids, labels, cls_input, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.ptm(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, cls_input=cls_input)
        pred_scores = self.classifier(sequence_output)
        loss = F.cross_entropy(input=pred_scores, label=labels)

        return loss


class DiffCSE(nn.Layer):
    def __init__(self, args, tokenizer):
        super(DiffCSE, self).__init__()
        self.args = args
        self.mode = args.mode
        self.tokenizer = tokenizer
        self.generator_name = args.generator_name
        self.discriminator_name = args.discriminator_name

        self.extractor = Extractor(args.extractor_name, margin=args.margin, scale=args.scale, output_emb_size=args.output_emb_size)
        self.generator = ErnieForMaskedLM.from_pretrained(args.generator_name)
        self.discriminator = Discriminator(args.discriminator_name)

    def train_forward(self, 
                query_input_ids, 
                key_input_ids,
                query_token_type_ids=None, 
                key_token_type_ids=None,
                query_attention_mask=None,
                key_attention_mask=None,
                cls_token = 1
                ):

        # extract senmantic vector with extractor and then comput CL loss
        loss, ori_cls_embedding = self.extractor(query_input_ids, key_input_ids, query_token_type_ids=query_token_type_ids, key_token_type_ids=key_token_type_ids, query_attention_mask=query_attention_mask, key_attention_mask=key_attention_mask)

        
        with paddle.no_grad():
            # mask tokens for query and key input_ids and then predict mask token with generator
            input_ids = paddle.concat([query_input_ids, key_input_ids], axis=0)
            attention_mask = paddle.concat([query_attention_mask, key_attention_mask], axis=0)
            mlm_input_ids, _ = mask_tokens(paddle.concat([query_input_ids, key_input_ids], axis=0), self.tokenizer, mlm_probability=self.args.mlm_probability)
            pred_tokens = self.generator(mlm_input_ids, attention_mask=attention_mask).argmax(-1)

        pred_tokens[:, 0] = cls_token
        e_inputs = pred_tokens * attention_mask
        replaced = pred_tokens != input_ids
        e_labels = paddle.cast(replaced, dtype="int64") * attention_mask

        rtd_loss = self.discriminator(e_inputs, e_labels, cls_input=ori_cls_embedding)
        loss = loss + self.args.lambda_weight * rtd_loss

        return loss
    

    def test_forward(self, 
                query_input_ids, 
                key_input_ids,
                query_token_type_ids=None, 
                key_token_type_ids=None,
                query_attention_mask=None,
                key_attention_mask=None
                ):

        # compute cosine similarity for query and key text
        cos_sim = self.extractor.cosine_sim(query_input_ids, key_input_ids, query_token_type_ids=query_token_type_ids, key_token_type_ids=key_token_type_ids, query_attention_mask=query_attention_mask, key_attention_mask=key_attention_mask)

        return cos_sim

    def forward(self, 
                query_input_ids, 
                key_input_ids,
                query_token_type_ids=None, 
                key_token_type_ids=None,
                query_attention_mask=None,
                key_attention_mask=None,
                cls_token = 1,
                mode="train"
                ):
        if mode == "train":
            return self.train_forward(query_input_ids, key_input_ids, query_token_type_ids=query_token_type_ids, key_token_type_ids=key_token_type_ids, query_attention_mask=query_attention_mask, key_attention_mask=key_attention_mask, cls_token=cls_token)
        else:
            return self.test_forward(query_input_ids, key_input_ids, query_token_type_ids=query_token_type_ids, key_token_type_ids=key_token_type_ids, query_attention_mask=query_attention_mask, key_attention_mask=key_attention_mask)




            
            

        




















