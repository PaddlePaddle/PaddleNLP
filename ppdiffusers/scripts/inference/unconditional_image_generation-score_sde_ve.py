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

from ppdiffusers import DiffusionPipeline, ScoreSdeVePipeline

# 加载模型和scheduler
sde_ve = ScoreSdeVePipeline.from_pretrained("google/ncsnpp-ffhq-1024")

# 执行pipeline进行推理
image = pipe().images[0]

# 保存图片
image[0].save("sde_ve_generated_image.png")
