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

import paddle

from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

# 加载pipeline
pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")

# 下载初始图片
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"

init_image = load_image(url).resize((128, 128))
init_image.save("original-image.png")

# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(init_image, num_inference_steps=100, eta=1).images[0]

image.save("super_resolution-latent_diffusion-result.png")
