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

from ppdiffusers import MidiProcessor, SpectrogramDiffusionPipeline
from ppdiffusers.utils.download_utils import ppdiffusers_url_download

# Download MIDI from: wget https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/beethoven_hammerklavier_2.mid
mid_file_path = ppdiffusers_url_download(
    "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/beethoven_hammerklavier_2.mid", cache_dir="."
)
pipe = SpectrogramDiffusionPipeline.from_pretrained("google/music-spectrogram-diffusion", paddle_dtype=paddle.float16)
processor = MidiProcessor()
output = pipe(processor(mid_file_path))
audio = output.audios[0]

output_path = "unconditional_audio_generation-spectrogram_diffusion-result-beethoven_hammerklavier_2.wav"
# save the audio sample as a .wav file
scipy.io.wavfile.write(output_path, rate=16000, data=audio)

# 可以直接使用 IPython.display.Audio 来显示音频文件
display(Audio(output_path))
