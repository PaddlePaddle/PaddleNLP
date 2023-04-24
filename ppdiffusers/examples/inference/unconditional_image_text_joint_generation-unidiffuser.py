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


from ppdiffusers import UniDiffuserPipeline

pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser")
result = pipe(mode="joint", image=None, prompt=None)
image = result.images[0]
image.save("unconditional_image_text_generation-unidiffuser-result.png")
text = result.texts[0]
with open("unconditional_image_text_generation-unidiffuser-result.txt", "w") as f:
    print("{}\n".format(text), file=f)
