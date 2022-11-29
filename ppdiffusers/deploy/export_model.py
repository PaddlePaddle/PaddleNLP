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

import distutils.util
import os

import paddle

from paddlenlp.transformers import CLIPTextModel
from ppdiffusers import AutoencoderKL, UNet2DConditionModel


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpaint", type=distutils.util.strtobool, default=False, help="Wheter to export inpaint model"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="CompVis/stable-diffusion-v1-4",
        help="The pretrained diffusion model.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="The pretrained diffusion model.")
    return parser.parse_args()


class VAEDecoder(AutoencoderKL):
    def forward(self, z):
        return self.decode(z, True).sample


class VAEEncoder(AutoencoderKL):
    def forward(self, z):
        return self.encode(z, True).latent_dist.mode()


if __name__ == "__main__":
    paddle.set_device("cpu")
    args = parse_arguments()
    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.pretrained_model_name_or_path, "text_encoder"))
    vae_decoder = VAEDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae_encoder = VAEEncoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Convert to static graph with specific input description
    text_encoder = paddle.jit.to_static(
        text_encoder,
        input_spec=[paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")],  # input_ids
    )

    # Save text_encoder in static graph model.
    save_path = os.path.join(args.output_path, "text_encoder", "inference")
    paddle.jit.save(text_encoder, save_path)
    print(f"Save text_encoder model in {save_path} successfully.")

    # Convert to static graph with specific input description
    vae_decoder = paddle.jit.to_static(
        vae_decoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 4, None, None], dtype="float32", name="latent_sample"
            ),  # latent_sample
        ],
    )
    # Save vae_decoder in static graph model.
    save_path = os.path.join(args.output_path, "vae_decoder", "inference")
    paddle.jit.save(vae_decoder, save_path)
    print(f"Save vae_decoder model in {save_path} successfully.")

    # Convert to static graph with specific input description
    vae_encoder = paddle.jit.to_static(
        vae_encoder,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype="float32", name="sample"  # N, C, H, W
            ),  # latent
        ],
    )
    # Save vae_encoder in static graph model.
    save_path = os.path.join(args.output_path, "vae_encoder", "inference")
    paddle.jit.save(vae_encoder, save_path)
    print(f"Save vae_encoder model in {save_path} successfully.")

    # Convert to static graph with specific input description
    unet_channels = 9 if args.inpaint else 4
    unet = paddle.jit.to_static(
        unet,
        input_spec=[
            paddle.static.InputSpec(shape=[None, unet_channels, None, None], dtype="float32", name="sample"),  # sample
            paddle.static.InputSpec(shape=[1], dtype="int64", name="timestep"),  # timesteps
            paddle.static.InputSpec(
                shape=[None, None, 768], dtype="float32", name="encoder_hidden_states"
            ),  # encoder_hidden_states
        ],
    )
    save_path = os.path.join(args.output_path, "unet", "inference")
    paddle.jit.save(unet, save_path)
    print(f"Save unet model in {save_path} successfully.")
