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
import cv2
import numpy as np
from PIL import Image

from ppdiffusers import ControlNetModel, StableDiffusionControlNetPipeline
from ppdiffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None
)

resolution = 512
image = np.array(
    load_image("https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/control_bird_canny_demo.png")
)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image = canny_image.resize((resolution, resolution))


prompt = "bird"
image = pipe(
    prompt=prompt,
    image=canny_image,
    num_inference_steps=50,
    height=resolution,
    width=resolution,
    controlnet_conditioning_scale=1.0,
).images[0]
image.save("text_to_image_generation-controlnet-result-bird_canny.png")
