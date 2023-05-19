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

from ppdiffusers import VersatileDiffusionDualGuidedPipeline
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)
text = "a red car in the sun"

pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained("shi-labs/versatile-diffusion")
pipe.remove_unused_weights()

text_to_image_strength = 0.75
image = pipe(prompt=text, image=image, text_to_image_strength=text_to_image_strength).images[0]
image.save("dual_text_and_image_guided_generation-versatile_diffusion-result.png")
