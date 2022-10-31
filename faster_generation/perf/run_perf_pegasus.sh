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

GPU_ID=4
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for model_name in IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese;
    do
        for batch_size in 1 4 8 16;
            do   
                for num_beams in 2 4 6 8;
                    do
                        for input_len in 60;
                            do
                                for generate_len in 20;
                                    do
                                        for perf_type in pd_faster_fp16 pd_faster_fp32 pd hf;
                                            do 
                                                echo model_name: $model_name, perf_type: $perf_type, batch_size:$batch_size, num_beams: $num_beams, input_len: $input_len, generate_len: $generate_len
                                                python pegasus_perf.py \
                                                    --model_name_or_path=$model_name \
                                                    --perf_type=$perf_type \
                                                    --batch_size=$batch_size \
                                                    --num_beams=$num_beams \
                                                    --input_len=$input_len \
                                                    --generate_len=$generate_len \
                                                    --gpu_id ${GPU_ID}
                                                sleep 3s
                                            done
                                    done
                            done
                    done
            done
    done