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
"""XLNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np
import paddle.fluid as fluid
import modeling


def _get_initializer(args):
    if args.init == "uniform":
        param_initializer = fluid.initializer.Uniform(
            low=-args.init_range, high=args.init_range)
    elif args.init == "normal":
        param_initializer = fluid.initializer.Normal(scale=args.init_std)
    else:
        raise ValueError("Initializer {} not supported".format(args.init))
    return param_initializer


def init_attn_mask(args, place):
    """create causal attention mask."""
    qlen = args.max_seq_length
    mlen = 0 if 'mem_len' not in args else args.mem_len
    same_length = False if 'same_length' not in args else args.same_length
    dtype = 'float16' if args.use_fp16 else 'float32'
    attn_mask = np.ones([qlen, qlen], dtype=dtype)
    mask_u = np.triu(attn_mask)
    mask_dia = np.diag(np.diag(attn_mask))
    attn_mask_pad = np.zeros([qlen, mlen], dtype=dtype)
    attn_mask = np.concatenate([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = np.tril(attn_mask)
        attn_mask = np.concatenate(
            [ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    attn_mask = attn_mask[:, :, None, None]
    attn_mask_t = fluid.global_scope().find_var("attn_mask").get_tensor()
    attn_mask_t.set(attn_mask, place)


class XLNetConfig(object):
    def __init__(self, config_path):
        self._config_dict = self._parse(config_path)

    def _parse(self, config_path):
        try:
            with open(config_path) as json_file:
                config_dict = json.load(json_file)
        except Exception:
            raise IOError("Error in parsing xlnet model config file '%s'" %
                          config_path)
        else:
            return config_dict

    def __getitem__(self, key):
        return self._config_dict[key]

    def has_key(self, key):
        return self._config_dict.has_key(key)

    def print_config(self):
        for arg, value in sorted(six.iteritems(self._config_dict)):
            print('%s: %s' % (arg, value))
        print('------------------------------------------------')


class XLNetModel(object):
    def __init__(self,
                 xlnet_config,
                 input_ids,
                 seg_ids,
                 input_mask,
                 args,
                 mems=None,
                 perm_mask=None,
                 target_mapping=None,
                 inp_q=None):
        self._tie_weight = True

        self._d_head = xlnet_config['d_head']
        self._d_inner = xlnet_config['d_inner']
        self._d_model = xlnet_config['d_model']
        self._ff_activation = xlnet_config['ff_activation']
        self._n_head = xlnet_config['n_head']
        self._n_layer = xlnet_config['n_layer']
        self._n_token = xlnet_config['n_token']
        self._untie_r = xlnet_config['untie_r']
        self._xlnet_config = xlnet_config

        self._dropout = args.dropout
        self._dropatt = args.dropatt

        self._mem_len = None if 'mem_len' not in args else args.mem_len
        self._reuse_len = None if 'reuse_len' not in args else args.reuse_len
        self._bi_data = False if 'bi_data' not in args else args.bi_data
        self._clamp_len = args.clamp_len
        self._same_length = False if 'same_length' not in args else args.same_length
        # Initialize all weigths by the specified initializer, and all biases 
        # will be initialized by constant zero by default.
        self._param_initializer = _get_initializer(args)
        self.input_mask = input_mask

        tfm_args = dict(
            n_token=self._n_token,
            initializer=self._param_initializer,
            attn_type="bi",
            n_layer=self._n_layer,
            d_model=self._d_model,
            n_head=self._n_head,
            d_head=self._d_head,
            d_inner=self._d_inner,
            ff_activation=self._ff_activation,
            untie_r=self._untie_r,
            use_bfloat16=False,
            dropout=self._dropout,
            dropatt=self._dropatt,
            mem_len=self._mem_len,
            reuse_len=self._reuse_len,
            bi_data=self._bi_data,
            clamp_len=self._clamp_len,
            same_length=self._same_length,
            name='model_transformer')
        input_args = dict(
            inp_k=input_ids,
            seg_id=seg_ids,
            input_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            inp_q=inp_q)
        tfm_args.update(input_args)
        self.output, self.new_mems, self.lookup_table = modeling.transformer_xl(
            **tfm_args)
        #self._build_model(input_ids, sentence_ids, input_mask)

    def get_initializer(self):
        return self._param_initializer

    def get_debug_ret(self):
        return self.debug_ret

    def get_sequence_output(self):
        return self.output

    def get_pooled_out(self, summary_type, use_summ_proj=True):
        """
	Args:
	  summary_type: str, "last", "first", "mean", or "attn". The method
	    to pool the input to get a vector representation.
	  use_summ_proj: bool, whether to use a linear projection during pooling.
	Returns:
	  float32 Tensor in shape [bsz, d_model], the pooled representation.
	"""
        summary = modeling.summarize_sequence(
            summary_type=summary_type,
            hidden=self.output,
            d_model=self._d_model,
            n_head=self._n_head,
            d_head=self._d_head,
            dropout=self._dropout,
            dropatt=self._dropatt,
            input_mask=self.input_mask,
            initializer=self._param_initializer,
            use_proj=use_summ_proj,
            name='model_sequnece_summary')

        return summary
