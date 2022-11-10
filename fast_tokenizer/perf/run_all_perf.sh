# !/bin/sh

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

for seq_len in 32 64 128 256 512; do
for batch_size in 1 2 4 8 16 32 64; do
mkdir -p seq_len_$seq_len/batch_size_$batch_size
for thread_num in 1 2 4 8 16 32 64; do
echo "Experiment setting: thread_num=$thread_num, batch_size=$batch_size, sequence_length=$seq_len"
export OMP_NUM_THREADS=$thread_num
export RAYON_RS_NUM_CPUS=$thread_num
python perf.py --batch_size $batch_size --max_seq_length $seq_len >seq_len_$seq_len/batch_size_$batch_size/parallel$thread_num.log 2>nohup.out
done 
done
done