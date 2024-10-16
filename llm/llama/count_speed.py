# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# 检查命令行参数数量是否正确
if len(sys.argv) != 2:
    print("Usage: python script.py <logfile>")
    sys.exit(1)

# 从命令行参数中获取日志文件路径
logfile = sys.argv[1]


# 打开日志文件
with open(logfile, "r") as file:
    lines = file.readlines()

# 初始化变量用于存储 interval_samples_per_second 的值和计数
total_samples_per_second = 0
count = 0

skip_row = 2
row = 0
# 遍历每一行日志
for line in lines:
    row = row + 1
    if row <= skip_row:
        continue
    # 检查当前行是否包含 interval_samples_per_second 字段
    if "interval_samples_per_second" in line:
        # 使用字符串分割找到 interval_samples_per_second 字段的值
        samples_per_second = float(line.split("interval_samples_per_second: ")[1].split(",")[0])
        # 将值加到总和中
        total_samples_per_second += samples_per_second
        # 增加计数
        count += 1

# 计算均值
if count > 0:
    average_samples_per_second = total_samples_per_second / count
    print("Average interval_samples_per_second:", average_samples_per_second)
else:
    print("No interval_samples_per_second found in the log file.")

