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

import sys
import matplotlib.pyplot as plt

mode = sys.argv[1]
batch_size = sys.argv[2]

ylabel_name = "CLUE Avg Score"
title_name = 'PaddleNLP Chinese Models'

if mode == 'gpu':
    picture_name = './gpu_bs' + batch_size + '.png'
    xlabel_name = "Latency (ms) under FP16 on Tesla T4"
elif mode == 'cpu1':
    picture_name = './cpu_thread1_bs' + batch_size + '.png'
    xlabel_name = "Latency (ms) under FP32 on Intel(R) Xeon(R) Gold 6271C, num_threads=1"
elif mode == 'cpu8':
    picture_name = './cpu_thread8_bs' + batch_size + '.png'
    xlabel_name = "Latency (ms) under FP32 on Intel(R) Xeon(R) Gold 6271C, num_threads=8"
else:
    raise ValueError("Only supports gpu, cpu1, cpu8.")

# Each element has model_name, model_param_num, latency(ms), clue avg score,
# color, the size of circle.
# Models of the same series are best represented by colors of the same color
# system. https://zhuanlan.zhihu.com/p/65220518 is for reference.

data = [
    [
        'ERNIE 3.0-Base-zh', '117.95M', 2.69, 226.43, 33.08, 76.05, '#F08080',
        11.8
    ],
    [
        'ERNIE 3.0-Medium-zh', '75.43M', 1.42, 113.35, 17.32, 72.49, '#A52A2A',
        7.5
    ],
    ['ERNIE 3.0-Mini-zh', '26.95M', 0.75, 38.24, 5.54, 66.90, '#CD5C5C', 2.7],
    ['ERNIE 3.0-Micro-zh', '23.40M', 0.62, 26.44, 5.26, 64.21, '#FF6347', 2.3],
    ['ERNIE 3.0-Nano-zh', '17.91M', 0.57, 20.93, 3.22, 62.97, '#FF0000', 1.8],
    [
        'UER/RoBERTa-Base', '102.27M', 2.69, 226.16, 32.18, 71.71, '#4169E1',
        10.2
    ],
    [
        'UER/RoBERTa-6L768H', '59.74M', 1.36, 112.55, 16.21, 66.67, '#6495ED',
        6.0
    ],
    ['UER/RoBERTa-Medium', '36.56M', 1.02, 71.23, 10.84, 68.17, '#87CEFA', 3.7],
    # ['UER/RoBERTa-Small', '23.95M', 0.63, 36.33, 5.61, 59.69, '#B0E0E6', 2.4],
    # ['UER/RoBERTa-Mini','8.77M', 0.59, 10.61, 2.02, 53.85, '#40E0D0', 0.9],
    # ['UER/RoBERTa-Tiny','3.18M', 0.37, 2.08, 0.72, 49.19, '#4682B4', 0.3],
    [
        'TinyBERT6, Chinese', '64.47M', 1.44, 113.90, 16.37, 69.58, '#008000',
        6.5
    ],
    ['TinyBERT4, Chinese', '12.86M', 0.54, 16.53, 2.93, 60.83, '#8FBC8F', 1.3],
    [
        'HFL/RBTL3, Chinese', '61.00M', 1.34, 113.27, 16.02, 66.79, '#FFA500',
        6.1
    ],
    ['HFL/RBT6, Chinese', '59.74M', 1.43, 114.24, 16.35, 69.74, '#FFDEAD', 6.0],
    ['HFL/RBT4, Chinese', '46.56M', 1.03, 76.19, 11.08, 67.45, '#FFD700', 4.7],
    ['HFL/RBT3, Chinese', '38.43M', 0.86, 58.65, 8.28, 65.72, '#FFE4B5', 3.8],
]

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for i in range(len(data)):
    if mode == 'gpu':
        ln, = plt.plot(data[i][2],
                       data[i][5],
                       color=data[i][6],
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=data[i][7] * 2)
        xytext = (data[i][2] + 0.1, data[i][5] - 0.1)
        if data[i][0] in ('TinyBERT6, Chinese', 'UER/RoBERTa-6L768H'):
            xytext = (data[i][2] + 0.1, data[i][5] - 0.6)
        plt.annotate(data[i][0],
                     xy=(data[i][2], data[i][5]),
                     xytext=xytext,
                     size=5,
                     alpha=0.8)
        plt.annotate(data[i][1],
                     xy=(data[i][2], data[i][5]),
                     xytext=(xytext[0], xytext[1] - 0.3),
                     size=5,
                     alpha=0.8)

    elif mode == 'cpu1':
        ln, = plt.plot(data[i][3],
                       data[i][5],
                       color=data[i][6],
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=data[i][7] * 2)
        xytext = (data[i][3] + 8.0, data[i][5] - 0.1)
        if data[i][0] in ('TinyBERT6, Chinese', 'UER/RoBERTa-6L768H'):
            xytext = (data[i][3] + 8.0, data[i][5] - 0.6)
        plt.annotate(data[i][0],
                     xy=(data[i][3], data[i][5]),
                     xytext=xytext,
                     size=5,
                     alpha=0.8)
        plt.annotate(data[i][1],
                     xy=(data[i][3], data[i][5]),
                     xytext=(xytext[0], xytext[1] - 0.3),
                     size=5,
                     alpha=0.8)
    else:
        ln, = plt.plot(data[i][4],
                       data[i][5],
                       color=data[i][6],
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=data[i][7] * 2)
        xytext = (data[i][4] + 1.2, data[i][5] - 0.1)
        if data[i][0] in ('TinyBERT6, Chinese', 'UER/RoBERTa-6L768H'):
            xytext = (data[i][4] + 1.2, data[i][5] - 0.6)
        plt.annotate(data[i][0],
                     xy=(data[i][4], data[i][5]),
                     xytext=xytext,
                     size=5,
                     alpha=0.8)
        plt.annotate(data[i][1],
                     xy=(data[i][4], data[i][5]),
                     xytext=(xytext[0], xytext[1] - 0.3),
                     size=5,
                     alpha=0.8)

plt.title(title_name)
plt.xlabel(xlabel_name)
plt.ylabel(ylabel_name)

plt.savefig(picture_name, dpi=1000)
