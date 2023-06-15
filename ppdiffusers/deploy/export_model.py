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
import os
from pathlib import Path
from types import MethodType

import paddle

from ppdiffusers import (
    FastDeployRuntimeModel,
    FastDeployStableDiffusionInpaintPipeline,
    FastDeployStableDiffusionMegaPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)


def convert_ppdiffusers_pipeline_to_fastdeploy_pipeline(
    model_path: str, output_path: str, sample: bool = False, height: int = None, width: int = None
):
    # specify unet model with unet pre_temb_act opt enabled.
    unet_model = UNet2DConditionModel.from_pretrained(model_path, resnet_pre_temb_non_linearity=True, subfolder="unet")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, unet=unet_model, safety_checker=None, feature_extractor=None
    )
    output_path = Path(output_path)
    # calculate latent's H and W
    latent_height = height // 8 if height is not None else None
    latent_width = width // 8 if width is not None else None
    # get arguments
    cross_attention_dim = pipeline.unet.config.cross_attention_dim  # 768 or 1024 or 1280
    unet_channels = pipeline.unet.config.in_channels  # 4 or 9
    vae_in_channels = pipeline.vae.config.in_channels  # 3
    vae_latent_channels = pipeline.vae.config.latent_channels  # 4
    print(
        f"cross_attention_dim: {cross_attention_dim}\n",
        f"unet_in_channels: {unet_channels}\n",
        f"vae_encoder_in_channels: {vae_in_channels}\n",
        f"vae_decoder_latent_channels: {vae_latent_channels}",
    )
    # 1. Convert text_encoder
    text_encoder = paddle.jit.to_static(
        pipeline.text_encoder,
        input_spec=[paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")],  # input_ids
    )
    save_path = os.path.join(args.output_path, "text_encoder", "inference")
    paddle.jit.save(text_encoder, save_path)
    print(f"Save text_encoder model in {save_path} successfully.")
    del pipeline.text_encoder

    # 2. Convert unet
    unet = paddle.jit.to_static(
        pipeline.unet,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, unet_channels, latent_height, latent_width], dtype="float32", name="sample"
            ),  # sample
            paddle.static.InputSpec(shape=[1], dtype="float32", name="timestep"),  # timestep
            paddle.static.InputSpec(
                shape=[None, None, cross_attention_dim], dtype="float32", name="encoder_hidden_states"
            ),  # encoder_hidden_states
        ],
    )
    save_path = os.path.join(args.output_path, "unet", "inference")
    paddle.jit.save(unet, save_path)
    print(f"Save unet model in {save_path} successfully.")
    del pipeline.unet

    def forward_vae_encoder_mode(self, z):
        return self.encode(z, True).latent_dist.mode()

    def forward_vae_encoder_sample(self, z):
        return self.encode(z, True).latent_dist.sample()

    # 3. Convert vae encoder
    vae_encoder = pipeline.vae
    if sample:
        vae_encoder.forward = MethodType(forward_vae_encoder_sample, vae_encoder)
    else:
        vae_encoder.forward = MethodType(forward_vae_encoder_mode, vae_encoder)

    vae_encoder = paddle.jit.to_static(
        vae_encoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, vae_in_channels, height, width],
                dtype="float32",
                name="sample",  # N, C, H, W
            ),  # latent
        ],
    )
    # Save vae_encoder in static graph model.
    save_path = os.path.join(args.output_path, "vae_encoder", "inference")
    paddle.jit.save(vae_encoder, save_path)
    print(f"Save vae_encoder model in {save_path} successfully.")

    # 4. Convert vae encoder
    vae_decoder = pipeline.vae

    def forward_vae_decoder(self, z):
        return self.decode(z, True).sample

    vae_decoder.forward = MethodType(forward_vae_decoder, vae_decoder)
    vae_decoder = paddle.jit.to_static(
        vae_decoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, vae_latent_channels, latent_height, latent_width], dtype="float32", name="latent_sample"
            ),  # latent_sample
        ],
    )
    # Save vae_decoder in static graph model.
    save_path = os.path.join(args.output_path, "vae_decoder", "inference")
    paddle.jit.save(vae_decoder, save_path)
    print(f"Save vae_decoder model in {save_path} successfully.")
    del pipeline.vae

    if "inpainting" in model_path:
        fd_pipe_cls = FastDeployStableDiffusionInpaintPipeline
    else:
        fd_pipe_cls = FastDeployStableDiffusionMegaPipeline

    fastdeploy_pipeline = fd_pipe_cls(
        vae_encoder=FastDeployRuntimeModel.from_pretrained(output_path / "vae_encoder"),
        vae_decoder=FastDeployRuntimeModel.from_pretrained(output_path / "vae_decoder"),
        text_encoder=FastDeployRuntimeModel.from_pretrained(output_path / "text_encoder"),
        unet=FastDeployRuntimeModel.from_pretrained(output_path / "unet"),
        tokenizer=pipeline.tokenizer,
        scheduler=pipeline.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    fastdeploy_pipeline.save_pretrained(output_path)
    print("FastDeploy pipeline saved to", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to the `ppdiffusers` checkpoint to convert (either a local directory or on the bos).",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")
    parser.add_argument(
        "--sample", action="store_true", default=False, help="Export the vae encoder in mode or sample"
    )
    parser.add_argument("--height", type=int, default=None, help="The height of output images. Default: None")
    parser.add_argument("--width", type=int, default=None, help="The width of output images. Default: None")
    args = parser.parse_args()

    convert_ppdiffusers_pipeline_to_fastdeploy_pipeline(
        args.pretrained_model_name_or_path, args.output_path, args.sample, args.height, args.width
    )
