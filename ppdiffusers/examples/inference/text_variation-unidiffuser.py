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
from ppdiffusers import UniDiffuserTextVariationPipeline
from ppdiffusers.models import FrozenCLIPEmbedder, UViT, CaptionDecoder

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserTextVariationPipeline(
    clip_text_model=FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14"),
    unet=UViT(pretrained_path="models/uvit_v1.pdparams"),
    caption_decoder=CaptionDecoder(pretrained_path="models/caption_decoder.pdparams"),
    scheduler=None,
)
pipe.clip_text_model = FrozenCLIPEmbedder(version="openai/clip-vit-large-patch14")
pipe.caption_decoder = CaptionDecoder(pretrained_path="models/caption_decoder.pdparams")

prompt = "an elephant under the sea"
text = pipe(prompt=prompt, generator=generator).texts[0]
with open("./unidiffuser-t2i2t.txt", "w") as f:
    print("{}\n".format(text), file=f)
