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
from ppdiffusers import UniDiffuserImageGenerationPipeline
from ppdiffusers.models import FrozenAutoencoderKL, UViT

generator = paddle.Generator().manual_seed(0)
pipe = UniDiffuserImageGenerationPipeline(
    unet=UViT(pretrained_path="models/uvit_v1.pdparams"),
    vae=FrozenAutoencoderKL(pretrained_path="models/autoencoder_kl.pdparams"),
    scheduler=None,
)
image = pipe(generator=generator).images[0]
image.save("./unidiffuser-i.png")
