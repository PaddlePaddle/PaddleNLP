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

from paddlenlp.transformers import (
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from ppdiffusers import DPMSolverMultistepScheduler, UniDiffuserPipeline

scheduler = DPMSolverMultistepScheduler()
from ppdiffusers.models import AutoencoderKL, CaptionDecoder, UViTModel
from ppdiffusers.utils import load_image

generator = paddle.Generator().manual_seed(0)

# mode = "t2i"
mode = "i2t"
# mode = "t2i2t"
# mode = "i2t2i"
# mode = "joint"
# mode = "i"
# mode = "t"

pipe = UniDiffuserPipeline(
    image_encoder=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32"),
    image_feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
    text_encoder=CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14"),
    tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
    unet=UViTModel.from_pretrained("thu-ml/unidiffuser/unet"),
    vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4/vae"),
    caption_decoder=CaptionDecoder.from_pretrained("thu-ml/unidiffuser/caption_decoder"),
    scheduler=scheduler,
)

image = load_image("https://bj.bcebos.com/v1/paddlenlp/models/community/thu-ml/data/space.jpg")
prompt = "an elephant under the sea"
result = pipe(mode=mode, image=image, prompt=prompt, generator=generator)

if result.images:
    image = result.images[0]
    image.save("./unidiffuser-{}.png".format(mode))

if result.texts:
    text = result.texts[0]
    print(text)
    with open("./unidiffuser-{}.txt".format(mode), "w") as f:
        print("{}\n".format(text), file=f)
