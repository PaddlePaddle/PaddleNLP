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

from ppdiffusers import UniDiffuserJointGenerationPipeline
from ppdiffusers.models import AutoencoderKL, CaptionDecoder, UViTModel

generator = paddle.Generator().manual_seed(0)

pipe = UniDiffuserJointGenerationPipeline(
    unet=UViTModel.from_pretrained("thu-ml/unidiffuser/unet"),
    vae=AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4/vae"),
    caption_decoder=CaptionDecoder.from_pretrained("thu-ml/unidiffuser/caption_decoder"),
    scheduler=None,
)

result = pipe(generator=generator)
image = result.images[0]
image.save("./unidiffuser-joint_i.png")
text = result.texts[0]
print(text)
with open("./unidiffuser-joint_t.txt", "w") as f:
    print("{}\n".format(text), file=f)
