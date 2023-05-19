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

from ppdiffusers import RePaintPipeline, RePaintScheduler
from ppdiffusers.utils import load_image

img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/celeba_hq_256.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/mask_256.png"

# Load the original image and the mask as PIL images
original_image = load_image(img_url).resize((256, 256))
mask_image = load_image(mask_url).resize((256, 256))

scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256", subfolder="scheduler")
pipe = RePaintPipeline.from_pretrained("google/ddpm-ema-celebahq-256", scheduler=scheduler)

output = pipe(
    image=original_image,
    mask_image=mask_image,
    num_inference_steps=250,
    eta=0.0,
    jump_length=10,
    jump_n_sample=10,
)
inpainted_image = output.images[0]

inpainted_image.save("image_inpainting-repaint-result.png")
