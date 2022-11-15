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

import torch
import os

dt = torch.load("ldm_1p4b_init0.ckpt", map_location="cpu")
unet = {}
vqvae = {}
ldmbert = {}
for k, v in dt['state_dict'].items():
    unet_key = "model.diffusion_model."
    if unet_key in k:
        unet[k.replace(unet_key, "")] = v

    vqvae_key = "first_stage_model."
    if vqvae_key in k:
        vqvae[k.replace(vqvae_key, "")] = v

    ldmbert_key = "cond_stage_model."
    if ldmbert_key in k:
        ldmbert[k.replace(ldmbert_key, "")] = v

os.makedirs("init_weights")
torch.save(unet, "init_weights/unet.pt")
torch.save(vqvae, "init_weights/vqvae.pt")
torch.save(ldmbert, "init_weights/ldmbert.pt")
