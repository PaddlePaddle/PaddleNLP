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
import scipy
from IPython.display import Audio, display

from ppdiffusers import AudioLDMPipeline

pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm", paddle_dtype=paddle.float16)

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

output_path = "text_to_audio_generation-audio_ldm-techno.wav"
# save the audio sample as a .wav file
scipy.io.wavfile.write(output_path, rate=16000, data=audio)

# 可以直接使用 IPython.display.Audio 来显示音频文件
display(Audio(output_path))
