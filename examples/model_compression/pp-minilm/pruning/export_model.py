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

import argparse
import logging
import os
import sys
import math
import random
import time
import json
from functools import partial
import distutils.util

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.common_ops_import import core

from paddlenlp.transformers import PPMiniLMModel
from paddlenlp.utils.log import logger

from paddleslim.nas.ofa import OFA, utils
from paddleslim.nas.ofa.convert_super import Convert, supernet
from paddleslim.nas.ofa.layers import BaseBlock

sys.path.append("../")
from data import MODEL_CLASSES, METRIC_CLASSES


def ppminilm_forward(self,
                     input_ids,
                     token_type_ids=None,
                     position_ids=None,
                     attention_mask=None):
    wtype = self.pooler.dense.fn.weight.dtype if hasattr(
        self.pooler.dense, 'fn') else self.pooler.dense.weight.dtype
    if self.use_faster_tokenizer:
        input_ids, token_type_ids = self.tokenizer(
            text=input_ids,
            text_pair=token_type_ids,
            max_seq_len=self.max_seq_len,
            pad_to_max_seq_len=self.pad_to_max_seq_len)
    if attention_mask is None:
        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id).astype(wtype) * -1e9, axis=[1, 2])
    embedding_output = self.embeddings(input_ids=input_ids,
                                       position_ids=position_ids,
                                       token_type_ids=token_type_ids)

    encoder_outputs = self.encoder(embedding_output, attention_mask)
    sequence_output = encoder_outputs
    pooled_output = self.pooler(sequence_output)
    return sequence_output, pooled_output


PPMiniLMModel.forward = ppminilm_forward


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(
            sum([
                list(classes[-1].pretrained_init_configuration.keys())
                for classes in MODEL_CLASSES.values()
            ], [])),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument(
        "--sub_model_output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the sub model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--static_sub_model",
        default=None,
        type=str,
        help=
        "The output directory where the sub static model will be written. If set to None, not export static model",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--n_gpu",
                        type=int,
                        default=1,
                        help="number of gpus to use, 0 for cpu.")
    parser.add_argument('--width_mult',
                        type=float,
                        default=1.0,
                        help="width mult you want to export")
    parser.add_argument('--depth_mult',
                        type=float,
                        default=1.0,
                        help="depth mult you want to export")
    parser.add_argument(
        "--use_faster_tokenizer",
        type=distutils.util.strtobool,
        default=True,
        help=
        "Whether to use FasterTokenizer to accelerate training or further inference."
    )
    args = parser.parse_args()
    return args


def do_export(args):
    paddle.set_device("gpu" if args.n_gpu else "cpu")
    args.model_type = args.model_type.lower()
    args.task_name = args.task_name.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config_path = os.path.join(args.model_name_or_path, 'model_config.json')
    cfg_dict = dict(json.loads(open(config_path).read()))

    kept_layers_index = {}
    if args.depth_mult < 1.0:
        depth = round(cfg_dict["init_args"][0]['num_hidden_layers'] *
                      args.depth_mult)
        cfg_dict["init_args"][0]['num_hidden_layers'] = depth
        for idx, i in enumerate(range(1, depth + 1)):
            kept_layers_index[idx] = math.floor(i / args.depth_mult) - 1

    os.rename(config_path, config_path + '_bak')
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg_dict, ensure_ascii=False))

    num_labels = cfg_dict['num_classes']

    model = model_class.from_pretrained(args.model_name_or_path,
                                        num_classes=num_labels)
    model.use_faster_tokenizer = args.use_faster_tokenizer

    origin_model = model_class.from_pretrained(args.model_name_or_path,
                                               num_classes=num_labels)

    os.rename(config_path + '_bak', config_path)

    sp_config = supernet(expand_ratio=[1.0, args.width_mult])
    model = Convert(sp_config).convert(model)

    ofa_model = OFA(model)

    sd = paddle.load(
        os.path.join(args.model_name_or_path, 'model_state.pdparams'))

    if len(kept_layers_index) == 0:
        ofa_model.model.set_state_dict(sd)
    else:
        for name, params in ofa_model.model.named_parameters():
            if 'encoder' not in name:
                params.set_value(sd[name])
            else:
                idx = int(name.strip().split('.')[3])
                mapping_name = name.replace(
                    '.' + str(idx) + '.',
                    '.' + str(kept_layers_index[idx]) + '.')
                params.set_value(sd[mapping_name])

    best_config = utils.dynabert_config(ofa_model, args.width_mult)
    for name, sublayer in ofa_model.model.named_sublayers():
        if isinstance(sublayer, paddle.nn.MultiHeadAttention):
            sublayer.num_heads = int(args.width_mult * sublayer.num_heads)

    is_text_pair = True
    if args.task_name in ('tnews', 'iflytek', 'cluewsc2020'):
        is_text_pair = False

    if args.use_faster_tokenizer:
        ofa_model.model.add_faster_tokenizer_op()
        if is_text_pair:
            origin_model_new = ofa_model.export(
                best_config,
                input_shapes=[[1], [1]],
                input_dtypes=[
                    core.VarDesc.VarType.STRINGS, core.VarDesc.VarType.STRINGS
                ],
                origin_model=origin_model)
        else:
            origin_model_new = ofa_model.export(
                best_config,
                input_shapes=[1],
                input_dtypes=core.VarDesc.VarType.STRINGS,
                origin_model=origin_model)
    else:
        origin_model_new = ofa_model.export(
            best_config,
            input_shapes=[[1, args.max_seq_length], [1, args.max_seq_length]],
            input_dtypes=['int64', 'int64'],
            origin_model=origin_model)

    for name, sublayer in origin_model_new.named_sublayers():
        if isinstance(sublayer, paddle.nn.MultiHeadAttention):
            sublayer.num_heads = int(args.width_mult * sublayer.num_heads)

    output_dir = os.path.join(args.sub_model_output_dir,
                              "model_width_%.5f" % args.width_mult)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = origin_model_new
    model_to_save.save_pretrained(output_dir)

    if args.static_sub_model != None:
        origin_model_new.to_static(
            args.static_sub_model,
            use_faster_tokenizer=args.use_faster_tokenizer,
            is_text_pair=is_text_pair)


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    do_export(args)
