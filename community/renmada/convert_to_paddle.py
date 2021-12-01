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
import numpy as np

huggingface_to_paddle1 = {
    "embeddings.LayerNorm": "embeddings.layer_norm",
    "transformer.layer": "encoder.layers",
    "attention.q_lin": "self_attn.q_proj",
    "attention.k_lin": "self_attn.k_proj",
    "attention.v_lin": "self_attn.v_proj",
    "attention.out_lin": "self_attn.out_proj",
    "ffn.lin1": "linear1",
    "ffn.lin2": "linear2",
    "sa_layer_norm": "norm1",
    "output_layer_norm": "norm2",
}

huggingface_to_paddle2 = {
    "bert": "distilbert",
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
    "predictions.decoder.": "predictions.decoder_",
    "predictions.transform.dense": "predictions.transform",
    "predictions.transform.LayerNorm": "predictions.layer_norm",
}


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         huggingface_to_paddle):
    import torch
    import paddle
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    Total_params = 0

    for k, v in pytorch_state_dict.items():
        mulValue = np.prod(v.shape)
        Total_params += mulValue
        is_transpose = False
        if k[-7:] == ".weight":
            if ".embeddings." not in k and ".LayerNorm." not in k:
                if v.ndim == 2:
                    v = v.transpose(0, 1)
                    is_transpose = True

        oldk = k
        for huggingface_name, paddle_name in huggingface_to_paddle.items():
            k = k.replace(huggingface_name, paddle_name)
        if k.startswith('distilbert.pooler.dense'):
            k = k.replace('distilbert.pooler.dense', 'pre_classifier')

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        paddle_state_dict[k] = v.data.numpy()
    paddle_dump_path = pytorch_checkpoint_path.replace('pytorch_model.bin',
                                                       'model_state.pdparams')
    paddle.save(paddle_state_dict, paddle_dump_path)
    print(f'Total params: {Total_params}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sshleifertiny_model_path",
        default="D:\\paddle_models\\sshleifertiny-distilbert-base-uncased-finetuned-sst-2-english\\pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.")
    parser.add_argument(
        "--base_model_path",
        default="D:\\paddle_models\\distilbert-base-multilingual-cased\\pytorch_model.bin",
        type=str,
        required=False,
        help="Path to the Pytorch checkpoint path.")
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(args.sshleifertiny_model_path,
                                         huggingface_to_paddle2)
    print()
    convert_pytorch_checkpoint_to_paddle(args.base_model_path,
                                         huggingface_to_paddle1)
