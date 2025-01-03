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

export PADDLE_RUN_ASYNC=true
export FLAGS_use_stride_kernel=false
export FLAGS_auto_growth_chunk_size_in_mb=512
export FLAGS_use_stream_safe_cuda_allocator=false
export CUSTOM_DEVICE_BLACK_LIST="softmax_with_cross_entropy"

export PYTHONPATH=../../../:$PYTHONPATH

echo 'run llama wiki_text eval, log: wikitext_eval_gcu.log'
python ../../../../slm/examples/benchmark/wiki_lambada/eval.py \
    --model_name_or_path "__internal_testing__/sci-benchmark-llama-13b-5k" \
    --device gcu \
    --batch_size 4 \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --tensor_parallel_degree 1 \
    --logging_steps 10 \
    --use_flash_attention True \
    --dtype float16 &> wikitext_eval_gcu.log &

