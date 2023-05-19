# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import glob
import os
import os.path as osp

import click
import paddle
from paddle.vision import transforms
from PIL import Image
from tqdm import tqdm

from ppdiffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline

image_processing = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)


def decode_image(image):
    image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]).cast("float32").numpy()
    image = StableDiffusionImg2ImgPipeline.numpy_to_pil(image)
    return image


@click.command()
@click.option("--vae_path", type=str)
@click.option("--src_size", type=int)
@click.option("--tgt_size", type=int)
@click.option("--imgs", type=str)
@click.option("--outdir", type=str)
def main(vae_path, src_size, tgt_size, imgs, outdir):
    imgs = sorted(glob.glob(imgs))
    model = AutoencoderKL.from_pretrained(vae_path)
    model.eval()
    with paddle.no_grad():
        os.makedirs(outdir, exist_ok=True)
        for img_path in tqdm(imgs):
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            assert w == tgt_size and h == tgt_size
            img = img.resize([src_size, src_size])

            img = image_processing(img).unsqueeze(0)

            z = model.encode(img).latent_dist.sample()
            recon = model.decode(z).sample

            decode_image(recon)[0].save(osp.join(outdir, osp.basename(img_path)))


if __name__ == "__main__":
    main()
