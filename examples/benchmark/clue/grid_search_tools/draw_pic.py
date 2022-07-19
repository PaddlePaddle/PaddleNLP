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
xlabel_name += ", batch_size=" + batch_size

# Each element has model_name, model_param_num, latency(ms), clue avg score,
# color, the size of circle.
# Models of the same series are best represented by colors of the same color
# system. https://zhuanlan.zhihu.com/p/65220518 is for reference.
data = [
    [
        [
            'ERNIE 3.0-Base', '117.95M', 2.69, 226.43, 33.08, 3.43, 205.57,
            34.10, 76.05, '#F08080', 11.8
        ],  # #F08080
        [
            'ERNIE 3.0-Medium', '75.43M', 1.42, 113.35, 17.32, 2.11, 104.06,
            17.50, 72.49, '#A52A2A', 7.5
        ],
        [
            'ERNIE 3.0-Mini', '26.95M', 0.75, 38.24, 5.54, 1.59, 30.28, 8.18,
            66.90, '#CD5C5C', 2.7
        ],
        [
            'ERNIE 3.0-Micro', '23.40M', 0.62, 26.44, 3.76, 1.33, 20.06, 5.46,
            64.21, '#FF6347', 2.3
        ],
        [
            'ERNIE 3.0-Nano', '17.91M', 0.57, 20.93, 3.22, 1.25, 15.24, 4.89,
            62.97, '#FF0000', 1.8
        ]
    ],
    [
        [
            'RoBERTa-Base', '102.27M', 2.69, 226.16, 32.18, 3.44, 204.27, 34.10,
            71.78, 'royalblue', 10.2
        ],  #'#4169E1'
        [
            'RoBERTa-6L768H', '59.74M', 1.43, 112.55, 16.21, 2.14, 102.95,
            18.55, 67.09, '#6495ED', 6.0
        ],
        [
            'RoBERTa-Medium', '36.56M', 1.02, 71.23, 10.84, 1.91, 65.74, 13.26,
            67.06, '#87CEFA', 3.7
        ],
        [
            'RoBERTa-Small', '23.95M', 0.63, 36.33, 5.61, 1.41, 33.26, 7.01,
            63.25, '#B0E0E6', 2.4
        ],
        # ['RoBERTa-Mini','8.77M', 0.59, 10.61, 2.02, 1.41, 10.03, 3.60, 53.40, '#40E0D0', 0.9],
        # ['RoBERTa-Tiny','3.18M', 0.37, 2.08, 0.72, 1.03, 2.25, 1.30, 44.45, '#4682B4', 0.3],
    ],
    [
        [
            'TinyBERT6', '59.74M', 1.44, 113.90, 16.37, 2.14, 104.06, 17.44,
            69.62, 'gold', 6.5
        ],  # '#008000'
        [
            'TinyBERT4', '11.46M', 0.54, 16.53, 2.93, 1.22, 14.02, 4.64, 60.82,
            '#8FBC8F', 1.3
        ],
    ],
    [
        # [
        #     'RBTL3', '61.00M', 1.34, 113.27, 16.02, 1.69, 101.59,
        #     15.47, 66.79, '#FFA500', 6.1
        # ],
        [
            'RBT6', '59.74M', 1.43, 114.24, 16.35, 2.14, 103.53, 17.27, 70.06,
            'mediumseagreen', 6.0
        ],  #'#FFDEAD'
        [
            'RBT4', '46.56M', 1.03, 76.19, 11.08, 1.60, 69.90, 12.60, 67.42,
            '#FFD700', 4.7
        ],
        [
            'RBT3', '38.43M', 0.86, 58.65, 8.28, 1.40, 52.12, 10.63, 65.72,
            '#FFE4B5', 3.8
        ],
    ]
]

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ln_list = []
size = 7
for i in range(len(data)):
    model_name_list = [model_info[0] for model_info in data[i]]
    clue_res_list = [model_info[-3] for model_info in data[i]]
    color = data[i][0][-2]
    num_param_list = [model_info[1] for model_info in data[i]]

    if mode == 'gpu':
        if batch_size == '32':
            latency_list = [model_info[2] for model_info in data[i]]
        else:
            latency_list = [model_info[5] for model_info in data[i]]

        ln, = plt.plot(latency_list,
                       clue_res_list,
                       color=color,
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=5)
        ln_list.append(ln)
        for j, model in enumerate(data[i]):
            xytext = (latency_list[j] + 0.05, clue_res_list[j] - 0.1)
            model_name = model_name_list[j]
            clue_res = clue_res_list[j]
            num_param = num_param_list[j]
            latency = latency_list[j]
            if model_name in ('RoBERTa-Medium', 'TinyBERT6', 'ERNIE 3.0-Nano'):
                xytext = (latency + 0.05, clue_res - 0.6)
            if model_name in ("RBT4"):
                xytext = (latency + 0.05, clue_res + 0.1)
            plt.annotate(model_name,
                         xy=(latency, clue_res),
                         xytext=xytext,
                         size=size,
                         alpha=1.0)
            plt.annotate(num_param,
                         xy=(latency, clue_res),
                         xytext=(xytext[0], xytext[1] - 0.3),
                         size=5,
                         alpha=1.0)

    elif mode == 'cpu1':
        if batch_size == '32':
            latency_list = [model_info[3] for model_info in data[i]]
        else:
            latency_list = [model_info[6] for model_info in data[i]]
        ln, = plt.plot(latency_list,
                       clue_res_list,
                       color=color,
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=5)
        ln_list.append(ln)
        for j, model in enumerate(data[i]):
            xytext = (latency_list[j] + 5.0, clue_res_list[j] - 0.1)
            model_name = model_name_list[j]
            clue_res = clue_res_list[j]
            num_param = num_param_list[j]
            latency = latency_list[j]
            if model_name in ('RoBERTa-Medium', 'TinyBERT6', 'ERNIE 3.0-Nano'):
                xytext = (latency + 5.0, clue_res - 0.6)
            plt.annotate(model_name,
                         xy=(latency, clue_res),
                         xytext=xytext,
                         size=size,
                         alpha=1.0)
            plt.annotate(num_param,
                         xy=(latency, clue_res),
                         xytext=(xytext[0], xytext[1] - 0.3),
                         size=5,
                         alpha=1.0)
    else:
        if batch_size == '32':
            latency_list = [model_info[4] for model_info in data[i]]
        else:
            latency_list = [model_info[7] for model_info in data[i]]
        ln, = plt.plot(latency_list,
                       clue_res_list,
                       color=color,
                       linewidth=2.0,
                       linestyle='-',
                       marker='o',
                       ms=5)
        ln_list.append(ln)
        for j, model in enumerate(data[i]):
            xytext = (latency_list[j] + 0.8, clue_res_list[j] - 0.1)
            model_name = model_name_list[j]
            clue_res = clue_res_list[j]
            num_param = num_param_list[j]
            latency = latency_list[j]
            if model_name in ('RoBERTa-Medium', 'TinyBERT6', 'ERNIE 3.0-Nano'):
                xytext = (latency + 0.8, clue_res - 0.6)
            plt.annotate(model_name,
                         xy=(latency, clue_res),
                         xytext=xytext,
                         size=size,
                         alpha=1.0)
            plt.annotate(num_param,
                         xy=(latency, clue_res),
                         xytext=(xytext[0], xytext[1] - 0.3),
                         size=5,
                         alpha=1.0)
    plt.legend(
        handles=ln_list,
        labels=['Baidu/ERNIE 3.0', 'UER/RoBERTa', 'Huawei/TinyBERT', 'HFL/RBT'],
        loc='best')

plt.title(title_name)
plt.xlabel(xlabel_name)
plt.ylabel(ylabel_name)

plt.savefig(picture_name, dpi=500)
