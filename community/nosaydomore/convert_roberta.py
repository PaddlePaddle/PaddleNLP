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

from collections import OrderedDict
import argparse
import paddle
import torch
import os
import json

from paddle.utils.download import get_path_from_url

huggingface_to_paddle = {
    "embeddings.LayerNorm": "embeddings.layer_norm",
    "encoder.layer": "encoder.layers",
    "attention.self.query": "self_attn.q_proj",
    "attention.self.key": "self_attn.k_proj",
    "attention.self.value": "self_attn.v_proj",
    "attention.output.dense": "self_attn.out_proj",
    "intermediate.dense": "linear1",
    "output.dense": "linear2",
    "attention.output.LayerNorm": "norm1",
    "output.LayerNorm": "norm2",
    "qa_outputs": 'classifier',
    'lm_head.bias': 'lm_head.decoder.bias'
}

convert_model_name_list = [
    "roberta-base",
    "roberta-large",
    "deepset/roberta-base-squad2",
    "uer/roberta-base-finetuned-chinanews-chinese",
    "sshleifer/tiny-distilroberta-base",
    "uer/roberta-base-finetuned-cluener2020-chinese",
    "uer/roberta-base-chinese-extractive-qa",
]

link_template = "https://huggingface.co/{}/resolve/main/pytorch_model.bin"

pretrained_init_configuration = {
    "roberta-base": {
        "attention_probs_dropout_prob": 0.1,
        "layer_norm_eps": 1e-05,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 514,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265
    },
    "roberta-large": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 514,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-05,
        "vocab_size": 50265
    },
    "deepset/roberta-base-squad2": {
        "layer_norm_eps": 1e-05,
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 514,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265
    },
    "uer/roberta-base-finetuned-chinanews-chinese": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128
    },
    "sshleifer/tiny-distilroberta-base": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 2,
        "initializer_range": 0.02,
        "intermediate_size": 2,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 514,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 1,
        "type_vocab_size": 1,
        "vocab_size": 50265
    },
    "uer/roberta-base-finetuned-cluener2020-chinese": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128
    },
    "uer/roberta-base-chinese-extractive-qa": {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 21128
    }
}


def convert_pytorch_checkpoint_to_paddle(pytorch_src_base_path,
                                         paddle_dump_base_path):
    for model_name in convert_model_name_list:
        model_state_url = link_template.format(model_name)

        paddle_dump_path = os.path.join(paddle_dump_base_path,
                                        model_name.split('/')[-1])

        if os.path.exists(
                os.path.join(paddle_dump_path, 'model_state.pdparams')):
            continue
        if not os.path.exists(paddle_dump_path):
            os.makedirs(paddle_dump_path)

        with open(os.path.join(paddle_dump_path, 'model_config.json'),
                  'w') as fw:
            json.dump(pretrained_init_configuration[model_name], fw)

        _ = get_path_from_url(model_state_url, paddle_dump_path)
        pytorch_checkpoint_path = os.path.join(paddle_dump_path,
                                               'pytorch_model.bin')
        pytorch_state_dict = torch.load(
            pytorch_checkpoint_path, map_location="cpu")
        paddle_state_dict = OrderedDict()
        for k, v in pytorch_state_dict.items():
            is_transpose = False
            if k[-7:] == ".weight":
                if ".embeddings." not in k and ".LayerNorm." not in k:
                    if v.ndim == 2:
                        v = v.transpose(0, 1)
                        is_transpose = True
            oldk = k
            if k == 'lm_head.bias' and 'lm_head.decoder.bias' in pytorch_state_dict.keys(
            ):
                continue

            for huggingface_name, paddle_name in huggingface_to_paddle.items():
                k = k.replace(huggingface_name, paddle_name)
            if k[:5] == 'bert.':
                k = k.replace('bert.', 'roberta.')

            print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
            paddle_state_dict[k] = v.data.numpy()
        del pytorch_state_dict

        paddle_dump_path = os.path.join(paddle_dump_path,
                                        'model_state.pdparams')
        paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    pytorch_src_base_path = os.path.dirname(os.path.realpath(__file__))
    paddle_dump_base_path = pytorch_src_base_path
    convert_pytorch_checkpoint_to_paddle(pytorch_src_base_path,
                                         paddle_dump_base_path)
