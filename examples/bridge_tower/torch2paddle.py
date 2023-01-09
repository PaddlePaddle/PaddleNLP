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

from collections import OrderedDict

import numpy as np
import paddle
import torch
from transformers import BertForMaskedLM as PTBertForMaskedLM

from paddlenlp.transformers import BertForPretraining as PDBertForMaskedLM


def convert_pytorch_checkpoint_to_paddle(
    pytorch_checkpoint_path="pytorch_model.bin",
    paddle_dump_path="model_state.pdparams",
    version="old",
):
    hf_to_paddle = {
        "embeddings.LayerNorm": "embeddings.layer_norm",
        "encoder.layer": "encoder.layers",
        "attn.out_proj": "self_attn.out_proj",
        "crossattention.self.query": "cross_attn.q_proj",
        "crossattention.self.key": "cross_attn.k_proj",
        "crossattention.self.value": "cross_attn.v_proj",
        "crossattention.output.dense": "cross_attn.out_proj",
        "crossattention.output.LayerNorm": "norm2",
        "cross_modal_text_layers.0.output.LayerNorm": "cross_modal_text_layers.0.norm3",
        "cross_modal_text_layers.1.output.LayerNorm": "cross_modal_text_layers.1.norm3",
        "cross_modal_text_layers.2.output.LayerNorm": "cross_modal_text_layers.2.norm3",
        "cross_modal_text_layers.3.output.LayerNorm": "cross_modal_text_layers.3.norm3",
        "cross_modal_text_layers.4.output.LayerNorm": "cross_modal_text_layers.4.norm3",
        "cross_modal_text_layers.5.output.LayerNorm": "cross_modal_text_layers.5.norm3",
        "cross_modal_image_layers.0.output.LayerNorm": "cross_modal_image_layers.0.norm3",
        "cross_modal_image_layers.1.output.LayerNorm": "cross_modal_image_layers.1.norm3",
        "cross_modal_image_layers.2.output.LayerNorm": "cross_modal_image_layers.2.norm3",
        "cross_modal_image_layers.3.output.LayerNorm": "cross_modal_image_layers.3.norm3",
        "cross_modal_image_layers.4.output.LayerNorm": "cross_modal_image_layers.4.norm3",
        "cross_modal_image_layers.5.output.LayerNorm": "cross_modal_image_layers.5.norm3",
        "attention.self.query": "self_attn.q_proj",
        "attention.self.key": "self_attn.k_proj",
        "attention.self.value": "self_attn.v_proj",
        "attention.output.dense": "self_attn.out_proj",
        "intermediate.dense": "linear1",
        "output.dense": "linear2",
        "attention.output.LayerNorm": "norm1",
        "output.LayerNorm": "norm2",
        # "vit_model.visual.positional_embedding":"vit_model.vision_model.positional_embedding.weight",
        # "predictions.decoder.": "predictions.decoder_",
        # "predictions.transform.dense": "predictions.transform",
        # "text_transformer.encoder.layer": "text_transformer.encoder.layers",
        "vit_model.visual.transformer.resblocks": "vit_model.vision_model.transformer.layers",
        "ln_1": "norm1",
        "ln_2": "norm2",
        "mlp.c_fc": "linear1",
        "mlp.c_proj": "linear2",
        "vit_model.visual": "vit_model.vision_model",
        "attn.in_proj_": "self_attn.in_proj_",
        "visual.transformer.resblocks": "vision_model.transformer.layers",
        "visual.": "vision_model.",
        "positional_embedding": "positional_embedding.weight",
    }
    do_not_transpose = []
    if version == "old":
        hf_to_paddle.update(
            {
                "predictions.bias": "predictions.decoder_bias",
                ".gamma": ".weight",
                ".beta": ".bias",
            }
        )
        do_not_transpose = do_not_transpose + ["predictions.decoder.weight"]

    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")["state_dict"]
    # pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    with open("raw_torch_param.txt", "w") as f:
        for k, v in pytorch_state_dict.items():
            f.write(k + "\n")
    paddle_state_dict = OrderedDict()
    for k, v in pytorch_state_dict.items():
        is_transpose = False
        if k[-7:] == ".weight":
            # embeddings.weight and LayerNorm.weight do not transpose
            if all(d not in k for d in do_not_transpose):
                if ".embeddings." not in k and ".LayerNorm." not in k:
                    if v.ndim == 2:
                        if "embeddings" not in k:
                            v = v.transpose(0, 1)
                            is_transpose = True
                        is_transpose = False
        oldk = k
        for hf_name, pd_name in hf_to_paddle.items():
            k = k.replace(hf_name, pd_name)

        # add prefix `bert.`
        if "bert." not in k and "cls." not in k and "classifier" not in k:
            k = k
        key = k
        if "in_proj" in key:
            # breakpoint()
            weight = v.data.numpy()
            # 分别给q、k、v赋值
            q_name = key.replace("in_proj_", "q_proj.")
            k_name = key.replace("in_proj_", "k_proj.")
            v_name = key.replace("in_proj_", "v_proj.")
            first_index = 2304 // 3
            second_index = 2304 // 3 * 2
            if "weight" in key:
                # weight=weight.transpose()
                query, key, value = (
                    weight[:first_index, :].transpose(),
                    weight[first_index:second_index, :].transpose(),
                    weight[second_index:, :].transpose(),
                )
                # query, key, value = weight[ :,:first_index], weight[ :,first_index:second_index], weight[ :,second_index:]
            elif "bias" in key:
                query, key, value = weight[:first_index], weight[first_index:second_index], weight[second_index:]
            paddle_state_dict[q_name] = query
            paddle_state_dict[k_name] = key
            paddle_state_dict[v_name] = value
        else:
            paddle_state_dict[k] = v.data.numpy()

        print(f"Converting: {oldk} => {k} | is_transpose {is_transpose}")
        # if('visual' in k):
        #     k = k.replace('visual','vision_model')
        # if('resblocks' in k):
        #     k = k.replace('resblocks','layers')

    # print(paddle_state_dict.keys())
    paddle.save(paddle_state_dict, paddle_dump_path)


def compare(out_torch, out_paddle):
    out_torch = out_torch.detach().numpy()
    out_paddle = out_paddle.detach().numpy()
    assert out_torch.shape == out_paddle.shape
    abs_dif = np.abs(out_torch - out_paddle)
    mean_dif = np.mean(abs_dif)
    max_dif = np.max(abs_dif)
    min_dif = np.min(abs_dif)
    print("mean_dif:{}".format(mean_dif))
    print("max_dif:{}".format(max_dif))
    print("min_dif:{}".format(min_dif))


def test_forward():
    paddle.set_device("cpu")
    model_torch = PTBertForMaskedLM.from_pretrained("./bert-base-uncased")
    model_paddle = PDBertForMaskedLM.from_pretrained("./bert-base-uncased")
    model_torch.eval()
    model_paddle.eval()
    np.random.seed(42)
    x = np.random.randint(1, model_paddle.bert.config["vocab_size"], size=(4, 64))
    input_torch = torch.tensor(x, dtype=torch.int64)
    out_torch = model_torch(input_torch)[0]

    input_paddle = paddle.to_tensor(x, dtype=paddle.int64)
    out_paddle = model_paddle(input_paddle)[0]

    print("torch result shape:{}".format(out_torch.shape))
    print("paddle result shape:{}".format(out_paddle.shape))
    compare(out_torch, out_paddle)


if __name__ == "__main__":
    convert_pytorch_checkpoint_to_paddle(
        "checkpoints/BridgeTower_pt_base.ckpt", "../BridgeTower_pd/pretrained/BridgeTower_pt_base.pdparams"
    )
    # convert_pytorch_checkpoint_to_paddle("checkpoints/BridgeTower_ftfpt_base_snlive.ckpt", "../BridgeTower_pd/pretrained/BridgeTower_ftfpt_base_snlive_base.pdparams")
    # convert_pytorch_checkpoint_to_paddle("checkpoints/BridgeTower_ftfpt_base_vqav2.ckpt", "./paddle_weight.pdparams")
    # convert_pytorch_checkpoint_to_paddle("data/corss_layer.pth","data/paddle_cross_layer.paparams")
    # convert_pytorch_checkpoint_to_paddle("data/vit_models.pth",'data/paddle_vit_model.paparams')
