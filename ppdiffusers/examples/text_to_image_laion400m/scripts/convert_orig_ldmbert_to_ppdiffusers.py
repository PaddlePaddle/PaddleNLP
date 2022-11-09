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

from ppdiffusers.pipelines.latent_diffusion import LDMBertModel
import paddle
import torch

paddle.set_device("cpu")
# 12层的ldmbert
encoder_layers = 12
d_model = 768
# 32层的ldmbert
# encoder_layers = 32
# d_model = 1280

pt_ldmbert_weights = "init_weights/ldmbert.pt"
output_dir = "./pretrained_paddle_model/ldmbert"

######################
text_encoder = LDMBertModel(vocab_size=30522,
                            max_position_embeddings=77,
                            encoder_layers=encoder_layers,
                            encoder_ffn_dim=d_model * 4,
                            encoder_attention_heads=8,
                            head_dim=64,
                            activation_function="gelu",
                            d_model=d_model,
                            dropout=0.0,
                            attention_dropout=0.0,
                            activation_dropout=0.0,
                            init_std=0.02,
                            pad_token_id=0)
old = torch.load(pt_ldmbert_weights, map_location="cpu")
new = {}
new["embeddings.word_embeddings.weight"] = old[
    "transformer.token_emb.weight"].numpy()
new["embeddings.position_embeddings.weight"] = old[
    "transformer.pos_emb.emb.weight"].numpy()
for i in range(encoder_layers):
    double_i = 2 * i
    double_i_plus1 = 2 * i + 1
    # convert norm
    new[f"encoder.layers.{i}.norm1.weight"] = old[
        f"transformer.attn_layers.layers.{double_i}.0.weight"].numpy()
    new[f"encoder.layers.{i}.norm1.bias"] = old[
        f"transformer.attn_layers.layers.{double_i}.0.bias"].numpy()

    new[f"encoder.layers.{i}.self_attn.q_proj.weight"] = old[
        f"transformer.attn_layers.layers.{double_i}.1.to_q.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.k_proj.weight"] = old[
        f"transformer.attn_layers.layers.{double_i}.1.to_k.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.v_proj.weight"] = old[
        f"transformer.attn_layers.layers.{double_i}.1.to_v.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.out_proj.weight"] = old[
        f"transformer.attn_layers.layers.{double_i}.1.to_out.weight"].t().numpy(
        )
    new[f"encoder.layers.{i}.self_attn.out_proj.bias"] = old[
        f"transformer.attn_layers.layers.{double_i}.1.to_out.bias"].numpy()

    new[f"encoder.layers.{i}.norm2.weight"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.0.weight"].numpy()
    new[f"encoder.layers.{i}.norm2.bias"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.0.bias"].numpy()
    new[f"encoder.layers.{i}.linear1.weight"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.weight"].t(
        ).numpy()
    new[f"encoder.layers.{i}.linear1.bias"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.bias"].numpy(
        )
    new[f"encoder.layers.{i}.linear2.weight"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.weight"].t(
        ).numpy()
    new[f"encoder.layers.{i}.linear2.bias"] = old[
        f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.bias"].t(
        ).numpy()

new["final_layer_norm.weight"] = old["transformer.norm.weight"].numpy()
new["final_layer_norm.bias"] = old["transformer.norm.bias"].numpy()

text_encoder.load_dict(new)
text_encoder.eval()
text_encoder.save_pretrained(output_dir)
