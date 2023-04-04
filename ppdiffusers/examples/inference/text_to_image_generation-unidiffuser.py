# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
from ppdiffusers import UniDiffuserTextToImagePipeline
from ppdiffusers.models import AutoencoderKL, CaptionDecoder, UViTModel

generator = paddle.Generator().manual_seed(0)

pipe = UniDiffuserTextToImagePipeline(
    text_encoder=CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14"),
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    unet=UViTModel(pretrained_path="models/uvit_v1.pdparams"),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams"),
    vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4/vae"),
    scheduler=None,
)

prompt = "an elephant under the sea"
image = pipe(prompt=prompt, generator=generator).images[0]
image.save("./unidiffuser-t2i.png")
