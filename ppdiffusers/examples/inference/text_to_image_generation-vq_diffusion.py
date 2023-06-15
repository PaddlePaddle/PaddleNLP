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

from ppdiffusers import VQDiffusionPipeline

pipe = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq")

output = pipe("teddy bear playing in the pool", truncation_rate=1.0)

image = output.images[0]
image.save("text_to_image_generation-vq_diffusion-result.png")
