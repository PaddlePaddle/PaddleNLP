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

from ppdiffusers import StableDiffusionUpscalePipeline
from ppdiffusers.utils import load_image

pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"
low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"
upscaled_image = pipe(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("text_guided_image_upscaling-stable_diffusion_2-result.png")
