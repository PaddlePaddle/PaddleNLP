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

from scipy.io.wavfile import write

from ppdiffusers import DanceDiffusionPipeline

# 加载模型和scheduler
pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k")

# 生成4s钟的音频
audios = pipe(audio_length_in_s=4.0).audios

# 保存音频到本地
for i, audio in enumerate(audios):
    write(f"unconditional_audio_generation-dance_diffusion-result_{i}.wav", pipe.unet.sample_rate, audio.transpose())
