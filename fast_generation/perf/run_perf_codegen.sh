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

GPU_ID=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for model_name in Salesforce/codegen-350M-mono Salesforce/codegen-2B-mono Salesforce/codegen-6B-mono; 
    do   
        for top_k in 1 4 8 16;
            do
                for input_len in 60;
                    do
                        for generate_len in 20;
                            do
                                for perf_type in pd pd_faster_fp32 pd_faster_fp16 hf;
                                    do 
                                        echo model_name: $model_name, perf_type: $perf_type, top_k: $top_k, top_p: 1.0, input_len: $input_len, generate_len: $generate_len
                                        python codegen_perf.py \
                                            --model_name_or_path=$model_name \
                                            --perf_type=$perf_type \
                                            --top_k=$top_k \
                                            --top_p=1.0 \
                                            --input_len=$input_len \
                                            --generate_len=$generate_len \
                                            --gpu_id ${GPU_ID}
                                        sleep 3s
                                    done
                            done
                    done
            done
        for top_p in 0.4;
            do
                for input_len in 60;
                    do
                        for generate_len in 20;
                            do
                                for perf_type in pd pd_faster_fp32 pd_faster_fp16 hf;
                                    do 
                                        echo model_name: $model_name, perf_type: $perf_type, top_k: 0, top_p: $top_p, input_len: $input_len, generate_len: $generate_len
                                        python codegen_perf.py \
                                            --model_name_or_path=$model_name \
                                            --perf_type=$perf_type \
                                            --top_k=0 \
                                            --top_p=$top_p \
                                            --input_len=$input_len \
                                            --generate_len=$generate_len \
                                            --gpu_id ${GPU_ID}
                                        sleep 3s
                                    done
                            done
                    done
            done
    done