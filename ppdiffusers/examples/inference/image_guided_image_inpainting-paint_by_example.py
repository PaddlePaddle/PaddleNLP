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

from ppdiffusers import PaintByExamplePipeline
from ppdiffusers.utils import load_image

img_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/image_example_1.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/mask_example_1.png"
example_url = "https://paddlenlp.bj.bcebos.com/models/community/Fantasy-Studio/data/reference_example_1.jpeg"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))
example_image = load_image(example_url).resize((512, 512))

pipe = PaintByExamplePipeline.from_pretrained("Fantasy-Studio/Paint-by-Example")

# 使用fp16加快生成速度
with paddle.amp.auto_cast(True):
    image = pipe(image=init_image, mask_image=mask_image, example_image=example_image).images[0]
image.save("image_guided_image_inpainting-paint_by_example-result.png")
