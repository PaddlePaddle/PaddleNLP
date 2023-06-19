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

import imageio

from ppdiffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline

pipe = TextToVideoSDPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "An astronaut riding a horse."
video_frames = pipe(prompt, num_inference_steps=25).frames
imageio.mimsave("text_to_video_generation-synth-result-astronaut_riding_a_horse.mp4", video_frames, fps=8)
