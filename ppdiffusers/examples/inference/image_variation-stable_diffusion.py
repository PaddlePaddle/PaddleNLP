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


from paddle.vision import transforms

from ppdiffusers import StableDiffusionImageVariationPipeline
from ppdiffusers.utils import load_image

sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    from_diffusers=True,
    from_hf_hub=True,
)

im = load_image("https://bj.bcebos.com/v1/paddlenlp/models/community/thu-ml/data/space.jpg")

tform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation="bicubic",
        ),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ]
)
inp = tform(im)

out = sd_pipe(im, guidance_scale=3)
out["images"][0].save("image_variation-stable_diffusion-result.jpg")
