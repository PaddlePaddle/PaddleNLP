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

import os
from collections import OrderedDict

import numpy as np
import paddle
import torch
from transformers import BartForConditionalGeneration as hf_BartForConditionalGeneration

from paddlenlp.transformers import (
    BartForConditionalGeneration as pp_BartForConditionalGeneration,
)
from paddlenlp.utils import load_torch
from paddlenlp.utils.downloader import get_path_from_url_with_filelock
from paddlenlp.utils.log import logger

# Download huggingface models
hf_hub_repo = "fnlp/bart-base-chinese"
base_url = f"https://huggingface.co/{hf_hub_repo}/resolve/main/"

pp_hf_checkpoint = hf_hub_repo.replace("/", "_")
os.makedirs(pp_hf_checkpoint, exist_ok=True)

for i in [
    "config.json",
    "vocab.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "added_tokens.json",
    "spiece.model",
]:
    try:
        get_path_from_url_with_filelock(f"{base_url}{i}", pp_hf_checkpoint)
    except RuntimeError:
        logger.warning(f"{base_url}{i} not found.")

use_torch = False
try:
    hf_model = load_torch(os.path.join(pp_hf_checkpoint, "pytorch_model.bin"))
except ValueError:
    # Some models coming from pytorch_lighting
    use_torch = True
    hf_model = torch.load(os.path.join(pp_hf_checkpoint, "pytorch_model.bin"), map_location="cpu")

huggingface_to_paddle_encoder = {
    "model.encoder.embed_tokens": "bart.encoder.embed_tokens",
    "model.encoder.embed_positions": "bart.encoder.encoder_embed_positions",
    "model.encoder.layernorm_embedding": "bart.encoder.encoder_layernorm_embedding",
    ".self_attn_layer_norm.": ".norm1.",
    ".fc1.": ".linear1.",
    ".fc2.": ".linear2.",
    ".final_layer_norm.": ".norm2.",
    "model.encoder": "bart.encoder.encoder",
}

huggingface_to_paddle_decoder = {
    "model.decoder.embed_tokens": "bart.decoder.embed_tokens",
    "model.decoder.embed_positions": "bart.decoder.decoder_embed_positions",
    "model.decoder.layernorm_embedding": "bart.decoder.decoder_layernorm_embedding",
    ".self_attn_layer_norm.": ".norm1.",
    ".encoder_attn.": ".cross_attn.",
    ".encoder_attn_layer_norm.": ".norm2.",
    ".fc1.": ".linear1.",
    ".fc2.": ".linear2.",
    ".final_layer_norm.": ".norm3.",
    "model.decoder": "bart.decoder.decoder",
}

skip_weights = []

dont_transpose = [
    ".embed_positions.weight",
    ".embed_tokens.weight",
    "layernorm_embedding.weight",
    "norm.weight",
    ".shared.weight",
    "lm_head.weight",
]

paddle_state_dict = OrderedDict()

# Convert parameters
for k, v in hf_model.items():
    transpose = False
    if k in skip_weights:
        continue
    if k[-7:] == ".weight":
        if not any([w in k for w in dont_transpose]):
            if v.ndim == 2:
                v = v.transpose(0, 1) if use_torch else v.transpose()
                transpose = True
    oldk = k

    if "model.encoder." in k:
        for huggingface_name, paddle_name in huggingface_to_paddle_encoder.items():
            k = k.replace(huggingface_name, paddle_name)
    elif "model.decoder." in k:
        for huggingface_name, paddle_name in huggingface_to_paddle_decoder.items():
            k = k.replace(huggingface_name, paddle_name)

    if oldk == "model.shared.weight":
        k = "bart.shared.weight"

    if oldk == "lm_head.weight":
        k = "lm_head_weight"

    logger.info(f"Converting: {oldk} => {k} | is_transpose {transpose}")

    paddle_state_dict[k] = v.data.numpy() if use_torch else v

# Save to .pdparams
paddle.save(paddle_state_dict, os.path.join(pp_hf_checkpoint, "model_state.pdparams"))

# Compare ppnlp with hf
paddle.set_grad_enabled(False)
torch.set_grad_enabled(False)
pp_model = pp_BartForConditionalGeneration.from_pretrained(pp_hf_checkpoint)
pp_model.eval()
hf_model = hf_BartForConditionalGeneration.from_pretrained(pp_hf_checkpoint)
hf_model.eval()

input_ids = np.random.randint(1, 10000, size=(2, 10))
pp_inputs = paddle.to_tensor(input_ids)
hf_inputs = torch.tensor(input_ids)

pp_output = pp_model(pp_inputs)
hf_output = hf_model(hf_inputs)

diff = abs(hf_output.logits.detach().numpy() - pp_output.numpy())
logger.info(f"max diff: {np.max(diff)}, min diff: {np.min(diff)}")
