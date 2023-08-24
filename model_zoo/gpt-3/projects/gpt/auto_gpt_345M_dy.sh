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

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1

log_dir=log/dy_single
rm -rf $log_dir

python -m paddle.distributed.launch --log_dir=$log_dir --devices=0,1 ./tools/train.py \
    -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
    -o Model.module="GPTModule" \
    -o Model.hidden_dropout_prob=0 \
    -o Model.attention_probs_dropout_prob=0 \
    -o Model.use_recompute=True \
    -o Global.local_batch_size=8 \
    -o Global.micro_batch_size=4 \
    -o Distributed.dp_degree=1 \
    -o Distributed.mp_degree=1 \
    -o Distributed.pp_degree=2 \
    -o Distributed.sharding.sharding_degree=1 \
    -o Distributed.sharding.sharding_stage=1 \
    -o Engine.mix_precision.enable=False \
    -o Engine.max_steps=20 \
    -o Engine.eval_freq=10 \
    -o Engine.logging_freq=1 \
    -o Engine.verbose=3 \
    -o Engine.save_load.output_dir="" \
    -o Profiler_auto.memory_stats=True \
    # -o Profiler_auto.nvprof_start=5 \
    # -o Profiler_auto.nvprof_end=10
