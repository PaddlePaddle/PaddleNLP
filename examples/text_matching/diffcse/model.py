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


class ProjectionMLP(nn.Layer):

    def __init__(self, in_dim):
        super(ProjectionMLP, self).__init__()
        hidden_dim = in_dim * 2
        out_dim = in_dim
        affine = False
        list_layers = [
            nn.Linear(in_dim, hidden_dim, bias_attr=False),
            nn.BatchNorm1D(hidden_dim),
            nn.ReLU()
        ]
        list_layers += [
            nn.Linear(hidden_dim, out_dim, bias_attr=False),
            nn.BatchNorm1D(out_dim)
        ]
        self.net = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.net(x)


class Similarity(nn.Layer):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(axis=-1)
        self.record = None
        self.pos_avg = 0.0
        self.neg_avg = 0.0

    def forward(self, x, y, one_vs_one=False):
        if one_vs_one:
            sim = self.cos(x, y)
            return sim

        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        sim = self.cos(x, y)
        self.record = sim.detach()
        min_size = min(self.record.shape[0], self.record.shape[1])
        num_item = self.record.shape[0] * self.record.shape[1]
        self.pos_avg = paddle.diag(self.record).sum().item() / min_size
        self.neg_avg = (self.record.sum().item() - paddle.diag(
            self.record).sum().item()) / (num_item - min_size)
        return sim / self.temp


class Encoder(nn.Layer):

    def __init__(self,
                 pretrained_model_name):  #, temp=0.05, output_emb_size=None):
        super(Encoder, self).__init__()
        self.ptm = AutoModel.from_pretrained(pretrained_model_name)
        self.mlp = ProjectionMLP(self.ptm.config['hidden_size'])

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                with_pooler=False):
        # Note: cls_embedding is poolerd embedding with act tanh
        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids,
                                                  position_ids, attention_mask)
        if not with_pooler:
            ori_cls_embedding = sequence_output[:, 0, :]
        else:
            ori_cls_embedding = cls_embedding

        mlp_cls_embedding = self.mlp(ori_cls_embedding)

        return mlp_cls_embedding

    # def forward(
    #     self,
    #     query_input_ids,
    #     key_input_ids,
    #     query_token_type_ids=None,
    #     query_position_ids=None,
    #     query_attention_mask=None,
    #     key_token_type_ids=None,
    #     key_position_ids=None,
    #     key_attention_mask=None,
    #     with_pooler=False
    # ):
    #     mlp_query_cls_embedding = self.get_pooled_embedding(
    #         query_input_ids,
    #         query_token_type_ids,
    #         query_position_ids,
    #         query_attention_mask,
    #         with_pooler=with_pooler
    #     )
    #     mlp_key_cls_embedding = self.get_pooled_embedding(
    #         key_input_ids,
    #         key_token_type_ids,
    #         key_position_ids,
    #         key_attention_mask,
    #         with_pooler=with_pooler
    #     )

    #     return mlp_query_cls_embedding, mlp_key_cls_embedding

    # def single_forward(
    #     self,
    #     query_input_ids,
    #     query_token_type_ids=None,
    #     query_position_ids=None,
    #     query_attention_mask=None,
    #     with_pooler=False
    # ):
    #     query_cls_embedding = self.get_pooled_embedding(
    #         query_input_ids,
    #         query_token_type_ids,
    #         query_position_ids,
    #         query_attention_mask,
    #         with_pooler=with_pooler
    #     )

    #     return query_cls_embedding


class Discriminator(nn.Layer):

    def __init__(self, ptm_model_name):
        super(Discriminator, self).__init__()
        self.ptm = CustomErnie.from_pretrained(ptm_model_name)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                labels=None,
                cls_input=None,
                token_type_ids=None,
                attention_mask=None,
                mode="train"):
        if mode == "train":
            assert labels is not None and cls_input is not None
            sequence_output, _ = self.ptm(input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          cls_input=cls_input)
            pred_scores = self.classifier(sequence_output)
            loss = F.cross_entropy(input=pred_scores, label=labels)
            pooled_output = sequence_output[:, 0, :]
            return loss, pooled_output, pred_scores.argmax(-1)
        else:
            assert cls_input is not None
            sequence_output, _ = self.ptm(input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          cls_input=cls_input)
            pooled_output = sequence_output[:, 0, :]
            return pooled_output


class DiffCSE(nn.Layer):

    def __init__(self,
                 encoder_name,
                 generator_name,
                 discriminator_name,
                 enc_tokenizer,
                 gen_tokenizer,
                 dis_tokenizer,
                 temp=0.05,
                 output_emb_size=32,
                 mlm_probability=0.15,
                 lambda_weight=0.15):
        super(DiffCSE, self).__init__()
        self.encoder_name = encoder_name
        self.generator_name = generator_name
        self.discriminator_name = discriminator_name
        self.enc_tokenizer = enc_tokenizer
        self.gen_tokenizer = gen_tokenizer
        self.dis_tokenizer = dis_tokenizer
        self.temp = temp
        self.output_emb_size = output_emb_size
        self.mlm_probability = mlm_probability
        self.lambda_weight = lambda_weight

        self.encoder = Encoder(encoder_name)
        self.generator = ErnieForMaskedLM.from_pretrained(generator_name)
        self.discriminator = Discriminator(discriminator_name)

        self.temp = temp
        self.sim = Similarity(temp)
        if output_emb_size is not None:
            self.emb_reduce_linear = nn.Linear(
                self.encoder.ptm.config['hidden_size'] +
                self.discriminator.ptm.config['hidden_size'], output_emb_size)

        self.rtd_acc = 0.0
        self.rtd_rep_acc = 0.0
        self.rtd_fix_acc = 0.0

    def train_forward(self,
                      query_input_ids,
                      key_input_ids,
                      query_token_type_ids=None,
                      key_token_type_ids=None,
                      query_attention_mask=None,
                      key_attention_mask=None):

        # extract senmantic vector with encoder and then comput CL loss
        mlp_query_cls_embedding = self.encoder(
            query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask)
        mlp_key_cls_embedding = self.encoder(key_input_ids,
                                             token_type_ids=key_token_type_ids,
                                             attention_mask=key_attention_mask)

        mlp_cls_embedding = paddle.concat(
            [mlp_query_cls_embedding, mlp_key_cls_embedding], axis=0)

        with paddle.no_grad():
            # mask tokens for query and key input_ids and then predict mask token with generator
            input_ids = paddle.concat([query_input_ids, key_input_ids], axis=0)
            if self.encoder_name != self.generator_name:
                input_ids = self.encode_by_generator(input_ids)
            attention_mask = paddle.concat(
                [query_attention_mask, key_attention_mask], axis=0)
            mlm_input_ids, _ = mask_tokens(input_ids,
                                           self.gen_tokenizer,
                                           mlm_probability=self.mlm_probability)
            # predict tokens using generator
            pred_tokens = self.generator(
                mlm_input_ids, attention_mask=attention_mask).argmax(axis=-1)

        pred_tokens = pred_tokens.detach()

        if self.generator_name != self.discriminator_name:
            pred_tokens = self.encode_by_discriminator(pred_tokens)
            input_ids = self.encode_by_discriminator(input_ids)

        pred_tokens[:, 0] = self.dis_tokenizer.cls_token_id
        e_inputs = pred_tokens * attention_mask
        replaced = pred_tokens != input_ids
        e_labels = paddle.cast(replaced, dtype="int64") * attention_mask
        rtd_loss, dis_pooled_output, prediction = self.discriminator(
            e_inputs, labels=e_labels, cls_input=mlp_cls_embedding)
        dis_mlp_query_cls_embedding, dis_mlp_key_cls_embedding = paddle.split(
            dis_pooled_output, num_or_sections=2, axis=0)
        query_cls_embedding = paddle.concat(
            [mlp_query_cls_embedding, dis_mlp_query_cls_embedding], axis=-1)
        key_cls_embedding = paddle.concat(
            [mlp_key_cls_embedding, dis_mlp_key_cls_embedding], axis=-1)

        if self.output_emb_size is not None:
            query_cls_embedding = self.emb_reduce_linear(query_cls_embedding)
            key_cls_embedding = self.emb_reduce_linear(key_cls_embedding)

        cosine_sim = self.sim(query_cls_embedding, key_cls_embedding)
        labels = paddle.arange(0, query_cls_embedding.shape[0], dtype="int64")
        labels = paddle.reshape(labels, shape=[-1, 1])
        cl_loss = F.cross_entropy(input=cosine_sim, label=labels)

        loss = cl_loss + self.lambda_weight * rtd_loss

        rep = (e_labels == 1) * attention_mask
        fix = (e_labels == 0) * attention_mask
        self.rtd_rep_acc = float((prediction * rep).sum() / rep.sum())
        self.rtd_fix_acc = float(1.0 - (prediction * fix).sum() / fix.sum())
        self.rtd_acc = float(((prediction == e_labels) * attention_mask).sum() /
                             attention_mask.sum())

        return loss, rtd_loss

    def encode_by_generator(self, batch_tokens):
        new_tokens = []
        for one_tokens in batch_tokens:
            one_gen_tokens = self.enc_tokenizer.convert_ids_to_tokens(
                one_tokens.tolist())
            new_tokens.append(
                self.gen_tokenizer.convert_tokens_to_ids(one_gen_tokens))

        return paddle.to_tensor(new_tokens)

    def encode_by_discriminator(self, batch_tokens):
        new_tokens = []
        for one_tokens in batch_tokens:
            one_gen_tokens = self.gen_tokenizer.convert_ids_to_tokens(
                one_tokens.tolist())
            new_tokens.append(
                self.dis_tokenizer.convert_tokens_to_ids(one_gen_tokens))

        return paddle.to_tensor(new_tokens)

    def encode_by_discriminator2(self, batch_tokens):
        new_tokens = []
        for one_tokens in batch_tokens:
            one_enc_tokens = self.enc_tokenizer.convert_ids_to_tokens(
                one_tokens.tolist())
            new_tokens.append(
                self.dis_tokenizer.convert_tokens_to_ids(one_enc_tokens))

        return paddle.to_tensor(new_tokens)

    @paddle.no_grad()
    def test_forward(self,
                     query_input_ids,
                     key_input_ids,
                     query_token_type_ids=None,
                     key_token_type_ids=None,
                     query_attention_mask=None,
                     key_attention_mask=None):
        # extract senmantic vector with encoder and then comput CL loss
        mlp_query_cls_embedding = self.encoder(
            query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask)
        mlp_key_cls_embedding = self.encoder(key_input_ids,
                                             token_type_ids=key_token_type_ids,
                                             attention_mask=key_attention_mask)

        if self.encoder_name != self.discriminator_name:
            query_input_ids = self.encode_by_discriminator2(query_input_ids)
            key_input_ids = self.encode_by_discriminator2(key_input_ids)
            query_input_ids = query_input_ids * query_attention_mask
            key_input_ids = key_input_ids * key_attention_mask

        dis_mlp_query_cls_embedding = self.discriminator(
            query_input_ids, cls_input=mlp_query_cls_embedding, mode="test")
        dis_mlp_key_cls_embedding = self.discriminator(
            key_input_ids, cls_input=mlp_key_cls_embedding, mode="test")

        query_cls_embedding = paddle.concat(
            [mlp_query_cls_embedding, dis_mlp_query_cls_embedding], axis=-1)
        key_cls_embedding = paddle.concat(
            [mlp_key_cls_embedding, dis_mlp_key_cls_embedding], axis=-1)

        if self.output_emb_size is not None:
            query_cls_embedding = self.emb_reduce_linear(query_cls_embedding)
            key_cls_embedding = self.emb_reduce_linear(key_cls_embedding)

        cosine_sim = self.sim(query_cls_embedding,
                              key_cls_embedding,
                              one_vs_one=True)

        return cosine_sim

    def forward(self,
                query_input_ids,
                key_input_ids,
                query_token_type_ids=None,
                key_token_type_ids=None,
                query_attention_mask=None,
                key_attention_mask=None,
                mode="train"):
        if mode == "train":
            return self.train_forward(query_input_ids,
                                      key_input_ids,
                                      query_token_type_ids=query_token_type_ids,
                                      key_token_type_ids=key_token_type_ids,
                                      query_attention_mask=query_attention_mask,
                                      key_attention_mask=key_attention_mask)
        else:
            return self.test_forward(query_input_ids,
                                     key_input_ids,
                                     query_token_type_ids=query_token_type_ids,
                                     key_token_type_ids=key_token_type_ids,
                                     query_attention_mask=query_attention_mask,
                                     key_attention_mask=key_attention_mask)
