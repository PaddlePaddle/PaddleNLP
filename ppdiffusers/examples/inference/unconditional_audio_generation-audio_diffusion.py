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
from scipy.io.wavfile import write

from ppdiffusers import AudioDiffusionPipeline

# 加载模型和scheduler
pipe = AudioDiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256")
pipe.set_progress_bar_config(disable=None)
generator = paddle.Generator().manual_seed(42)

output = pipe(generator=generator)
audio = output.audios[0]
image = output.images[0]

# 保存音频到本地
for i, audio in enumerate(audio):
    write(f"audio_diffusion_test{i}.wav", pipe.mel.sample_rate, audio.transpose())

# 保存图片
image.save("unconditional_audio_generation-audio_diffusion-result.png")
