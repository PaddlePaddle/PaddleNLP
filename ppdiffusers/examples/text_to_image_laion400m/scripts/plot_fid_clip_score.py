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

import matplotlib.pyplot as plt

clip_pd = [26.421875, 27.000000, 27.359375, 27.593750, 27.734375, 27.843750]
fid_pd = [
    61.63832793539143,
    60.8674158133241,
    61.06873714387473,
    61.611543101882205,
    62.41915003785908,
    63.73414302529454,
]

clip_pt = [22.312500, 22.890625, 23.265625, 23.484375, 23.703125, 23.781250]
fid_pt = [
    84.44459421090801,
    80.3668421393279,
    78.9310124831315,
    77.22472126942046,
    76.33773728759894,
    75.6021109021998,
]

plt.plot(clip_pd, fid_pd, label="Paddle line", linewidth=3, color="r", marker="o", markerfacecolor="blue")
plt.plot(clip_pt, fid_pt, label="Pytorch line", linewidth=3, color="b", marker="o", markerfacecolor="red")
plt.xlabel("CLIP Score")
plt.ylabel("FID@1k")
plt.title("12W Globel Step Pareto Curves - DDIM")
plt.legend()
plt.savefig("ddim-12w.png")
plt.show()
