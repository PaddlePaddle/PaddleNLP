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

from ppdiffusers import SemanticStableDiffusionPipeline

pipe = SemanticStableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.set_progress_bar_config(disable=None)
prompt = "a photo of a cat"
edit = {
    "editing_prompt": ["sunglasses"],
    "reverse_editing_direction": [False],
    "edit_warmup_steps": 10,
    "edit_guidance_scale": 6,
    "edit_threshold": 0.95,
    "edit_momentum_scale": 0.5,
    "edit_mom_beta": 0.6,
}
seed = 3
guidance_scale = 7
generator = paddle.Generator().manual_seed(seed)
output = pipe(
    [prompt],
    generator=generator,
    guidance_scale=guidance_scale,
    num_inference_steps=50,
    width=512,
    height=512,
)
image = output.images[0]
image.save("text_guided_generation-semantic_stable_diffusion-result.png")
