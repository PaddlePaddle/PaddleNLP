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

shift = int(sys.argv[1])
top = int(sys.argv[2])
total_part = int(sys.argv[3])

f_list = []
for part in range(total_part):
    f0 = open('output/res.top%s-part%s' % (top, part))
    f_list.append(f0)

line_list = []
for part in range(total_part):
    line = f_list[part].readline()
    line_list.append(line)

out = open('output/dev.res.top%s' % top, 'w')
last_q = ''
ans_list = {}
while line_list[-1]:
    cur_list = []
    for line in line_list:
        sub = line.strip().split('\t')
        cur_list.append(sub)

    if last_q == '':
        last_q = cur_list[0][0]
    if cur_list[0][0] != last_q:
        rank = sorted(ans_list.items(), key=lambda a: a[1], reverse=True)
        for i in range(top):
            out.write("%s\t%s\t%s\t%s\n" %
                      (last_q, rank[i][0], i + 1, rank[i][1]))
        ans_list = {}
    for i, sub in enumerate(cur_list):
        ans_list[int(sub[1]) + shift * i] = float(sub[-1])
    last_q = cur_list[0][0]

    line_list = []
    for f0 in f_list:
        line = f0.readline()
        line_list.append(line)

rank = sorted(ans_list.items(), key=lambda a: a[1], reverse=True)
for i in range(top):
    out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i + 1, rank[i][1]))
out.close()

print('output/dev.res.top%s' % top)
