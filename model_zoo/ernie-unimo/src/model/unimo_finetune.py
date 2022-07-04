#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""Unified Visual Language model."""

import json
import six
import codecs

import paddle.nn as nn
import paddle


class UNIMOConfig(object):
    """configuration"""

    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with codecs.open(config_path, 'r', encoding='utf-8') as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing unimo model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict.get(key, None)

    def __setitem__(self, key, value):
        self._config_dict[key] = value

    def print_config(self):
        """print config"""
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class UNIMOEncoder(nn.Layer):
    '''UNIMO pre-trained encoder network'''
    def __init__(self, voc_size, pos_seq_len, image_len, loc_len, d_model, num_layers, nhead, dim_feedforward, dropout=0.1, \
        activation='relu', attn_dropout=None, act_dropout=None, normalize_before=False):
        super(UNIMOEncoder, self).__init__()
        # parameters
        self.voc_size = voc_size
        self.pos_seq_len = pos_seq_len
        self.image_len = image_len
        self.loc_len = loc_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        # image embedding
        self.image_embedding = nn.Linear(
                image_len, d_model)
        # image location embedding
        self.image_loc_embedding = nn.Linear(
                loc_len, d_model)
        # text embedding
        self.text_embedding = nn.Embedding(voc_size, d_model)
        # text position embedding
        self.text_pos_embedding = nn.Embedding(pos_seq_len, d_model)
        # pre-process norm & dropout
        self.pre_norm = nn.LayerNorm(d_model)
        self.pre_dropout = nn.Dropout(dropout, mode="upscale_in_train")
        self.vision_pre_norm = nn.LayerNorm(d_model)
        self.vision_pre_dropout = nn.Dropout(dropout, mode="upscale_in_train")

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                nhead = nhead,
                dim_feedforward = dim_feedforward,
                dropout=dropout,
                activation=activation,
                attn_dropout=attn_dropout,
                act_dropout=act_dropout,
                normalize_before=normalize_before)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    
    
    def forward(self, emb_ids=None, input_mask=None, image_input=None, emb_obj_ids=None,
    text_adv_delta=None, image_adv_delta=None):
        if emb_ids is not None and image_input is not None and emb_obj_ids is not None:
            input_type = 'vol'
        elif emb_ids is not None and image_input is not None:
            input_type = 'vl'
        elif emb_ids is not None:
            input_type = 'l'
        elif image_input is not None and emb_obj_ids is not None:
            input_type = 'vo'
        else:
            raise ValueError('input feature error')

        
        assert input_mask is not None, "input_mask should not be none"

        self_attn_mask = input_mask
        self_attn_mask = paddle.scale(self_attn_mask, scale=1e4, bias=-1.0, bias_after_scale=False)
        n_head_self_attn_mask = paddle.stack(x=[self_attn_mask] * self.nhead, axis=1)
        n_head_self_attn_mask.stop_gradient = True
        emb_feature, _v_seq_len, _o_seq_len = None, None, None

        if emb_ids is not None:
            emb_out = 0
            # text part
            for emb_name, emb_id in emb_ids.items():
                if emb_name == "sent_embedding":
                    continue  # don't use sentence embedding
                elif emb_name == "word_embedding":
                    emb = self.text_embedding(emb_id)
                    emb_out = emb_out + emb 
                elif emb_name == "pos_embedding":
                    emb = self.text_pos_embedding(emb_id)
                    emb_out = emb_out + emb
            emb_out = emb_out.squeeze()
            if text_adv_delta is not None:
                emb_out = emb_out + text_adv_delta

            emb_out = self.pre_dropout(self.pre_norm(emb_out))

        if image_input is not None:
            # visual part
            if image_adv_delta is not None:
                emb_v_in = image_input["image_embedding"]
                emb_v_in = emb_v_in + image_adv_delta
            else:
                emb_v_in = image_input["image_embedding"]

            image_embeddings = self.image_embedding(emb_v_in)
            

            loc_emb_out = self.image_loc_embedding(image_input["loc_embedding"])

            emb_v_out = image_embeddings + loc_emb_out
            emb_v_out = self.vision_pre_dropout(self.vision_pre_norm(emb_v_out))

            _v_seq_len = emb_v_out.shape[1]

        if emb_obj_ids is not None:
            emb_obj_out = 0
            # text part
            for emb_obj_name, emb_obj_id in emb_obj_ids.items():
                if emb_name == "sent_embedding":
                    continue  # don't use sentence embedding
                elif emb_name == "word_embedding":
                    emb_obj = self.text_embedding(emb_id)
                    emb_obj_out = emb_obj_out + emb_obj
                elif emb_name == "pos_embedding":
                    emb_obj = self.text_pos_embedding(emb_id)
                    emb_obj_out = emb_obj_out + emb_obj
                
            emb_obj_out = self.pre_dropout(self.pre_norm(emb_obj_out))
            _o_seq_len = emb_obj_out.shape[1]

        if input_type == 'vol':
            assert emb_ids is not None and image_input is not None and emb_obj_ids is not None, "the input is invalid"
            emb_feature = paddle.concat([emb_v_out, emb_obj_out, emb_out], axis=1)
        elif input_type == 'vl':
            assert emb_ids is not None and image_input is not None and emb_obj_ids is None, "the input is invalid"
            emb_feature = paddle.concat([emb_v_out, emb_out], axis=1)
        elif input_type == 'l':
            assert emb_ids is not None and image_input is None and emb_obj_ids is None, "the input is invalid"
            emb_feature = emb_out
        elif input_type == 'vo':
            assert emb_ids is None and image_input is not None and emb_obj_ids is not None, "the input is invalid"
            emb_feature = paddle.concat([emb_v_out, emb_obj_out], axis=1)
        else:
            raise ValueError("The input type is invalid")

        enc_out = self.encoder(emb_feature, n_head_self_attn_mask)

        if input_type == 'vol':
            assert _v_seq_len is not None and _o_seq_len is not None, "the input is invalid"
            _vol_seq_len = enc_out.shape[1]
            enc_v_out = paddle.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            enc_o_out = paddle.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len], ends=[_v_seq_len + _o_seq_len])
            enc_l_out = paddle.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len + _o_seq_len], ends=[_vol_seq_len])
            enc_vol_out = enc_out
            return enc_vol_out, enc_v_out, enc_l_out
        elif input_type == 'vl':
            assert _v_seq_len is not None and _o_seq_len is None, "the input is invalid"
            _vl_seq_len = enc_out.shape[1]
            enc_v_out = paddle.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            enc_l_out = paddle.slice(
                input=enc_out, axes=[1], starts=[_v_seq_len], ends=[_vl_seq_len])
            enc_vl_out = enc_out
            return enc_vl_out, enc_v_out, enc_l_out
        elif input_type == 'vo':
            assert _v_seq_len is not None and _o_seq_len is not None, "the input is invalid"
            enc_v_out = paddle.slice(
                input=enc_out, axes=[1], starts=[0], ends=[_v_seq_len])
            return enc_v_out
        elif input_type == 'l':
            assert _v_seq_len is None and _o_seq_len is None, "the input is invalid"
            enc_l_out = enc_out
            return enc_l_out
        else:
            raise ValueError("The input type is invalid")


class UNIMOModel(object):
    """UNIMO model for finetuning"""

    def __init__(self,
                 config=None,
                 text_adv_delta=None,
                 image_adv_delta=None,
                 weight_sharing=True,
                 task_type="normal",
                 decoding=False,
                 is_multimodal_task = False
                 ):

        self.text_adv_delta = text_adv_delta
        self.image_adv_delta = image_adv_delta

        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._voc_size = config['vocab_size']
        self._max_position_seq_len = config['max_position_embeddings']
        self._hidden_act = config['hidden_act']
        self._prepostprocess_dropout = config['hidden_dropout_prob']
        self._attention_dropout = config['attention_probs_dropout_prob']
        self._weight_sharing = weight_sharing

        self._task_type = task_type
        self._emb_vocab_size = {"word_embedding": self._voc_size,
                                "pos_embedding": self._max_position_seq_len}

        self._is_dialogue_task = (task_type == "dialog")
        self._is_img2txt_task = (task_type == "img2txt")
        self._is_multimodal_task = is_multimodal_task

        if self._is_dialogue_task:
            self._role_type_size = config["role_type_size"]
            self._turn_type_size = config["turn_type_size"]
            self._emb_vocab_size["role_embedding"] = self._role_type_size
            self._emb_vocab_size["turn_embedding"] = self._turn_type_size
        else:
            self._sent_types = config['type_vocab_size']
            self._emb_vocab_size["sent_embedding"] = self._sent_types
            if self._is_multimodal_task or self._is_img2txt_task:
                self._image_class_size = config['image_class_size']
                self._class_attr_size = config['class_attr_size']
                self._image_embedding_size = config['image_embedding_size']
                self._image_predict_feature = config['image_predict_feature']
                self._image_predict_class = config['image_predict_class']
                self._image_use_attr = config['image_use_attr']
                self._image_use_soft_label = config['image_use_soft_label']
                self._image_emb_name = "image_embedding"
                self._loc_emb_name = "loc_embedding"

        self.encoder = UNIMOEncoder(voc_size=self._voc_size, pos_seq_len=self._max_position_seq_len, image_len=2048, loc_len=5, d_model=config['hidden_size'], num_layers=config['num_hidden_layers'], 
        nhead=config['num_attention_heads'], dim_feedforward=config['hidden_size']*4,  dropout=config['hidden_dropout_prob'],
        activation=self._hidden_act, attn_dropout=config['attention_probs_dropout_prob'], act_dropout=0, normalize_before=False)

        self._emb_dtype = "float32"

    def encode(self, emb_ids=None, input_mask=None, image_input=None, emb_obj_ids=None):
        if emb_ids is not None and image_input is not None and emb_obj_ids is not None:           
            self.enc_vol_out, self.enc_v_out, self.enc_l_out = self.encoder(emb_ids=emb_ids, input_mask=input_mask, image_input=image_input, emb_obj_ids=emb_obj_ids,
    text_adv_delta=self.text_adv_delta, image_adv_delta=self.image_adv_delta)
            return self.enc_vol_out, self.enc_v_out, self.enc_l_out
        elif emb_ids is not None and image_input is not None:
            self.enc_vl_out, self.enc_v_out, self.enc_l_out = self.encoder(emb_ids=emb_ids, input_mask=input_mask, image_input=image_input, emb_obj_ids=emb_obj_ids,
    text_adv_delta=self.text_adv_delta, image_adv_delta=self.image_adv_delta)
            return self.enc_vl_out, self.enc_v_out, self.enc_l_out 
        elif emb_ids is not None:
            self.enc_l_out = self.encoder(emb_ids=emb_ids, input_mask=input_mask, image_input=image_input, emb_obj_ids=emb_obj_ids,
    text_adv_delta=self.text_adv_delta, image_adv_delta=self.image_adv_delta)
            return self.enc_l_out
        elif image_input is not None and emb_obj_ids is not None:
            self.enc_v_out = self.encoder(emb_ids=emb_ids, input_mask=input_mask, image_input=image_input, emb_obj_ids=emb_obj_ids,
    text_adv_delta=self.text_adv_delta, image_adv_delta=self.image_adv_delta)
            return self.enc_v_out
        else:
            raise ValueError('input feature error')

    def set_state_dict(self, state_dict):
        self.encoder.set_state_dict(state_dict)

    def state_dict(self):
        return self.encoder.state_dict()