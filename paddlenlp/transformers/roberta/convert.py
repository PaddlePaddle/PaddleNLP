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
    # "roberta-large",
    "deepset/roberta-base-squad2",
    "uer/roberta-base-finetuned-chinanews-chinese",
    "sshleifer/tiny-distilroberta-base",
    "uer/roberta-base-finetuned-cluener2020-chinese",
    "uer/roberta-base-chinese-extractive-qa",
]

link_template = "https://huggingface.co/{}/resolve/main/pytorch_model.bin"


def convert_pytorch_checkpoint_to_paddle(pytorch_src_base_path,
                                         paddle_dump_base_path):
    for model_name in convert_model_name_list:
        model_state_url = link_template.format(model_name)
        pytorch_checkpoint_path = get_path_from_url(
            model_state_url, pytorch_src_base_path, None, False)
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

        paddle_dump_path = os.path.join(paddle_dump_base_path,
                                        model_name.split('/')[-1])
        if not os.path.exists(paddle_dump_path):
            os.makedirs(paddle_dump_path)

        paddle_dump_path = os.path.join(paddle_dump_path,
                                        'model_state.pdparams')
        paddle.save(paddle_state_dict, paddle_dump_path)


def compare_param(param1, param2):
    param2 = torch.tensor(param2.numpy())
    param1 = torch.tensor(param1.numpy())
    diff = (param1 - param2).abs()

    print(diff.mean())
    print(diff.max())


def view_diff_param():
    # pt_param1 = paddle.load('E:\deep_learning\model_save\\roberta_base\\test_model\\model_state.pdparams')
    pt_param1 = paddle.load(
        'E:\deep_learning\model_save\\roberta-base-finetuned-chinanews-chinese\\test_model\model_state.pdparams'
    )
    pt_param = paddle.load(
        'E:\deep_learning\model_save\\roberta-base-finetuned-chinanews-chinese\\model_state.pdparams'
    )
    # pt_param = torch.load('E:\deep_learning\model_save\\roberta_base\\pytorch_model.bin')
    # pt_param1 = torch.load('C:\\Users\C.z\.cache\huggingface\\transformers\\51ba668f7ff34e7cdfa9561e8361747738113878850a7d717dbc69de8683aaad.c7efaa30a0d80b2958b876969faa180e485944a849deee4ad482332de65365a7')
    # compare_param(pt_param['lm_head.bias'], pt_param1['lm_head.bias'])
    new_pd_param = OrderedDict()
    for k, v in pt_param1.items():
        # if 'lm_head' in k:
        print(k, ":")
        #     # k=k.replace('LayerNorm','layer_norm')
        compare_param(v, pt_param[k])
    # if k == 'lm_head.bias':
    #     k = 'lm_head.decoder.bias'
    # new_pd_param[k]=v

    # paddle.save(new_pd_param,'E:\deep_learning\model_save\\roberta_base\\model_state.pdparams')


if __name__ == "__main__":
    pytorch_src_base_path = os.path.dirname(os.path.realpath(__file__))
    paddle_dump_base_path = pytorch_src_base_path
    convert_pytorch_checkpoint_to_paddle(pytorch_src_base_path,
                                         paddle_dump_base_path)

    # param = paddle.load("E:\deep_learning\model_save\\roberta-base-finetuned-chinanews-chinese\model_state.pdparams")
    # param = torch.load("E:\deep_learning\model_save\\tiny_disll_roberta\\pytorch_model.bin")
    # for k,v in param.items():
    #     print("{}: {}".format(k,v.shape))
    #     if k=='lm_head.bias':
    #         par1=v
    #     if k=='lm_head.decoder.bias':
    #         par2=v

    # compare_param(par1, par2)
