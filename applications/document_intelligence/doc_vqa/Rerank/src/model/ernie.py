#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Ernie model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import json
import six
import logging
from io import open

import paddle.fluid as fluid

from model.transformer_encoder import encoder, pre_process_layer

log = logging.getLogger(__name__)


class ErnieConfig(object):

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing Ernie model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            log.info('%s: %s' % (arg, value))
        log.info('------------------------------------------------')


class ErnieModel(object):

    def __init__(self,
                 src_ids,
                 position_ids,
                 sentence_ids,
                 task_ids,
                 input_mask,
                 config,
                 weight_sharing=True,
                 model_name='',
                 is_noise=False):

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        if config['sent_type_vocab_size']:
            self._sent_types = config['sent_type_vocab_size']
        else:
            self._sent_types = config['type_vocab_size']

        self._use_task_id = config['use_task_id']
        if self._use_task_id:
            self._task_types = config['task_type_vocab_size']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        if is_noise:
            self._prepostprocess_dropout = 0
            self._attention_dropout = 0
        self._weight_sharing = weight_sharing
        self.checkpoints = []

        self._word_emb_name = "word_embedding"
        self._pos_emb_name = "pos_embedding"
        self._sent_emb_name = "sent_embedding"
        self._task_emb_name = "task_embedding"
        self._emb_dtype = "float32"

        # Initialize all weigths by truncated normal initializer, and all biases
        # will be initialized by constant zero by default.
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=config['initializer_range'])

        self._build_model(model_name, src_ids, position_ids, sentence_ids,
                          task_ids, input_mask)

    def _build_model(self, model_name, src_ids, position_ids, sentence_ids,
                     task_ids, input_mask):
        # padding id in vocabulary must be set to 0
        emb_out = fluid.layers.embedding(
            input=src_ids,
            size=[self._voc_size, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(name=model_name + self._word_emb_name,
                                       initializer=self._param_initializer),
            is_sparse=False)

        position_emb_out = fluid.layers.embedding(
            input=position_ids,
            size=[self._max_position_seq_len, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(name=model_name + self._pos_emb_name,
                                       initializer=self._param_initializer))

        sent_emb_out = fluid.layers.embedding(
            sentence_ids,
            size=[self._sent_types, self._emb_size],
            dtype=self._emb_dtype,
            param_attr=fluid.ParamAttr(name=model_name + self._sent_emb_name,
                                       initializer=self._param_initializer))

        emb_out = emb_out + position_emb_out
        emb_out = emb_out + sent_emb_out

        if self._use_task_id:
            task_emb_out = fluid.layers.embedding(
                task_ids,
                size=[self._task_types, self._emb_size],
                dtype=self._emb_dtype,
                param_attr=fluid.ParamAttr(name=model_name +
                                           self._task_emb_name,
                                           initializer=self._param_initializer))

            emb_out = emb_out + task_emb_out

        emb_out = pre_process_layer(emb_out,
                                    'nd',
                                    self._prepostprocess_dropout,
                                    name=model_name + 'pre_encoder')

        self_attn_mask = fluid.layers.matmul(x=input_mask,
                                             y=input_mask,
                                             transpose_y=True)

        self_attn_mask = fluid.layers.scale(x=self_attn_mask,
                                            scale=10000.0,
                                            bias=-1.0,
                                            bias_after_scale=False)
        n_head_self_attn_mask = fluid.layers.stack(x=[self_attn_mask] *
                                                   self._n_head,
                                                   axis=1)
        n_head_self_attn_mask.stop_gradient = True

        self._enc_out, self.checkpoints = encoder(
            enc_input=emb_out,
            attn_bias=n_head_self_attn_mask,
            n_layer=self._n_layer,
            n_head=self._n_head,
            d_key=self._emb_size // self._n_head,
            d_value=self._emb_size // self._n_head,
            d_model=self._emb_size,
            d_inner_hid=self._emb_size * 4,
            prepostprocess_dropout=self._prepostprocess_dropout,
            attention_dropout=self._attention_dropout,
            relu_dropout=0,
            hidden_act=self._hidden_act,
            preprocess_cmd="",
            postprocess_cmd="dan",
            param_initializer=self._param_initializer,
            model_name=model_name,
            name=model_name + 'encoder')

    def get_sequence_output(self):
        return self._enc_out

    def get_cls_output(self):
        """Get the first feature of each sequence for classification"""
        cls_output = fluid.layers.slice(input=self._enc_out,
                                        axes=[1],
                                        starts=[0],
                                        ends=[1])
        cls_output = fluid.layers.squeeze(cls_output, axes=[1])
        return cls_output

    def get_pooled_output(self):
        """Get the first feature of each sequence for classification"""
        next_sent_feat = fluid.layers.slice(input=self._enc_out,
                                            axes=[1],
                                            starts=[0],
                                            ends=[1])
        next_sent_feat = fluid.layers.fc(
            input=next_sent_feat,
            size=self._emb_size,
            act="tanh",
            param_attr=fluid.ParamAttr(name="pooled_fc.w_0",
                                       initializer=self._param_initializer),
            bias_attr="pooled_fc.b_0")
        return next_sent_feat

    def get_lm_output(self, mask_label, mask_pos):
        """Get the loss & accuracy for pretraining"""

        mask_pos = fluid.layers.cast(x=mask_pos, dtype='int32')

        # extract the first token feature in each sentence
        self.next_sent_feat = self.get_pooled_output()
        reshaped_emb_out = fluid.layers.reshape(x=self._enc_out,
                                                shape=[-1, self._emb_size])
        # extract masked tokens' feature
        mask_feat = fluid.layers.gather(input=reshaped_emb_out, index=mask_pos)

        # transform: fc
        mask_trans_feat = fluid.layers.fc(
            input=mask_feat,
            size=self._emb_size,
            act=self._hidden_act,
            param_attr=fluid.ParamAttr(name='mask_lm_trans_fc.w_0',
                                       initializer=self._param_initializer),
            bias_attr=fluid.ParamAttr(name='mask_lm_trans_fc.b_0'))

        # transform: layer norm
        mask_trans_feat = fluid.layers.layer_norm(
            mask_trans_feat,
            begin_norm_axis=len(mask_trans_feat.shape) - 1,
            param_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_scale',
                initializer=fluid.initializer.Constant(1.)),
            bias_attr=fluid.ParamAttr(
                name='mask_lm_trans_layer_norm_bias',
                initializer=fluid.initializer.Constant(1.)))
        # transform: layer norm
        #mask_trans_feat = pre_process_layer(
        #    mask_trans_feat, 'n', name='mask_lm_trans')

        mask_lm_out_bias_attr = fluid.ParamAttr(
            name="mask_lm_out_fc.b_0",
            initializer=fluid.initializer.Constant(value=0.0))
        if self._weight_sharing:
            fc_out = fluid.layers.matmul(
                x=mask_trans_feat,
                y=fluid.default_main_program().global_block().var(
                    self._word_emb_name),
                transpose_y=True)
            fc_out += fluid.layers.create_parameter(shape=[self._voc_size],
                                                    dtype=self._emb_dtype,
                                                    attr=mask_lm_out_bias_attr,
                                                    is_bias=True)

        else:
            fc_out = fluid.layers.fc(input=mask_trans_feat,
                                     size=self._voc_size,
                                     param_attr=fluid.ParamAttr(
                                         name="mask_lm_out_fc.w_0",
                                         initializer=self._param_initializer),
                                     bias_attr=mask_lm_out_bias_attr)

        mask_lm_loss = fluid.layers.softmax_with_cross_entropy(logits=fc_out,
                                                               label=mask_label)
        mean_mask_lm_loss = fluid.layers.mean(mask_lm_loss)

        return mean_mask_lm_loss

    def get_task_output(self, task, task_labels):
        task_fc_out = fluid.layers.fc(input=self.next_sent_feat,
                                      size=task["num_labels"],
                                      param_attr=fluid.ParamAttr(
                                          name=task["task_name"] + "_fc.w_0",
                                          initializer=self._param_initializer),
                                      bias_attr=task["task_name"] + "_fc.b_0")
        task_loss, task_softmax = fluid.layers.softmax_with_cross_entropy(
            logits=task_fc_out, label=task_labels, return_softmax=True)
        task_acc = fluid.layers.accuracy(input=task_softmax, label=task_labels)
        mean_task_loss = fluid.layers.mean(task_loss)
        return mean_task_loss, task_acc
