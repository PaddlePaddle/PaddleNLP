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
from ppdiffusers.utils import load_image
from paddlenlp.transformers import CLIPModel, CLIPProcessor
from ppdiffusers import UniDiffuserImageToTextPipeline
from ppdiffusers.models import (
    UViT,
    FrozenAutoencoderKL,
    CaptionDecoder,
)

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserImageToTextPipeline(
    image_encoder=CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    image_feature_extractor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    unet=UViT(pretrained_path="models/uvit_v1.pdparams"),
    vae=FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams"),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams"),
    scheduler=None,
)
pipe.vae = FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams")
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams")

url = "https://bj.bcebos.com/v1/paddlenlp/models/community/thu-ml/data/space.jpg"
image = load_image(url)
text = pipe(image=image, generator=generator).texts[0]
with open("./unidiffuser-i2t.txt", "w") as f:
    print("{}\n".format(text), file=f)
