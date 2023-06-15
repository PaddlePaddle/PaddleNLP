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

import argparse
import json

import paddle

from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LDMBertModel,
    LDMTextToImagePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.latent_diffusion import LDMBertConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", type=str, default="./model_state.pdparams", help="path to pretrained model_state.pdparams"
    )
    parser.add_argument("--output_path", type=str, default="./ldm_pipelines", help="the output path of pipeline.")
    parser.add_argument(
        "--vae_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4/vae",
        help="pretrained_vae_name_or_path.",
    )
    parser.add_argument(
        "--text_encoder_config_file", type=str, default="./config/ldmbert.json", help="text_encoder_config_file."
    )
    parser.add_argument("--unet_config_file", type=str, default="./config/unet.json", help="unet_config_file.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Pretrained tokenizer name or path if not the same as model_name.",
    )
    parser.add_argument("--model_max_length", type=int, default=77, help="Pretrained tokenizer model_max_length.")

    return parser.parse_args()


def extract_paramaters(model_file="model_state.pdparams", dtype="float32"):
    state_dict = paddle.load(model_file)
    unet = {}
    vae = {}
    bert = {}
    for k, v in state_dict.items():
        unet_key = "unet."
        if k.startswith(unet_key):
            unet[k.replace(unet_key, "")] = v.astype(dtype)

        vae_key = "vae."
        vqvae_key = "vqvae."
        if k.startswith(vae_key):
            vae[k.replace(vae_key, "")] = v.astype(dtype)
        elif k.startswith(vqvae_key):
            vae[k.replace(vqvae_key, "")] = v.astype(dtype)

        bert_key = "text_encoder."
        if k.startswith(bert_key):
            bert[k.replace(bert_key, "")] = v.astype(dtype)

    return unet, vae, bert


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def check_keys(model, state_dict):
    cls_name = model.__class__.__name__
    missing_keys = []
    mismatched_keys = []
    for k, v in model.state_dict().items():
        if k not in state_dict.keys():
            missing_keys.append(k)
        if list(v.shape) != list(state_dict[k].shape):
            mismatched_keys.append(k)
    if len(missing_keys):
        missing_keys_str = ", ".join(missing_keys)
        print(f"{cls_name} Found missing_keys {missing_keys_str}!")
    if len(mismatched_keys):
        mismatched_keys_str = ", ".join(mismatched_keys)
        print(f"{cls_name} Found mismatched_keys {mismatched_keys_str}!")


def build_pipelines(
    model_file,
    output_path,
    vae_name_or_path,
    unet_config_file,
    text_encoder_config_file,
    tokenizer_name_or_path="bert-base-uncased",
    model_max_length=77,
):
    vae = AutoencoderKL.from_config(vae_name_or_path)
    unet = UNet2DConditionModel(**read_json(unet_config_file))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=model_max_length)
    text_encoder_config = read_json(text_encoder_config_file)
    vocab_size = text_encoder_config["vocab_size"]
    max_position_embeddings = text_encoder_config["max_position_embeddings"]
    if tokenizer.vocab_size != vocab_size:
        logger.info(
            f"The tokenizer has a vocab size of {tokenizer.vocab_size}, while the text encoder has a vocab size of {vocab_size}, we will use {tokenizer.vocab_size} as vocab_size!"
        )
        text_encoder_config["vocab_size"] = tokenizer.vocab_size

    if tokenizer.model_max_length != max_position_embeddings:
        logger.info(
            f"The tokenizer's model_max_length {tokenizer.model_max_length}, while the text encoder's max_position_embeddings is {max_position_embeddings}, we will use {tokenizer.model_max_length} as max_position_embeddings!"
        )
        text_encoder_config["max_position_embeddings"] = tokenizer.model_max_length
    cofnig = LDMBertConfig(**text_encoder_config)
    text_encoder = LDMBertModel(cofnig)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        # Make sure the scheduler compatible with DDIM
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    unet_dict, vae_dict, text_encoder_dict = extract_paramaters(model_file)
    check_keys(unet, unet_dict)
    check_keys(vae, vae_dict)
    check_keys(text_encoder, text_encoder_dict)
    unet.load_dict(unet_dict)
    vae.load_dict(vae_dict)
    text_encoder.load_dict(text_encoder_dict)
    pipe = LDMTextToImagePipeline(bert=text_encoder, tokenizer=tokenizer, scheduler=scheduler, vqvae=vae, unet=unet)
    pipe.save_pretrained(output_path)


if __name__ == "__main__":
    args = parse_args()
    build_pipelines(
        model_file=args.model_file,
        output_path=args.output_path,
        vae_name_or_path=args.vae_name_or_path,
        unet_config_file=args.unet_config_file,
        text_encoder_config_file=args.text_encoder_config_file,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        model_max_length=args.model_max_length,
    )
