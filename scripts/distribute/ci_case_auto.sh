#!/usr/bin/env bash

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

set -e

export log_path=/workspace/case_logs
export root_path=/workspace/PaddleNLP

export gpt_case_path=$root_path/model_zoo/gpt-3
export gpt_data_path=/fleetx_data

export llama_case_path=$root_path/llm/llama/auto_parallel
export llama_data_path=/llama_data

unset CUDA_VISIBLE_DEVICES

function gpt_case_list_auto() {
    gpt_auto_recompute_bs16_fp32_DP1-MP1-PP1
    gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8
    gpt_auto_recompute_bs16_fp16_o2_DP1-MP2-PP4
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage3
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage3
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage1
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3
    gpt_auto_sp_acc_check
}

function llama_case_list_auto() {
    llama_dygraph_auto_bs8_fp32_DP2
    llama_dygraph_auto_bs8_fp32_DP2-MP2
    llama_dygraph_auto_bs8_fp32_DP2-MP2-PP2
    llama_dygraph_auto_bs8_fp16_DP2-MP2-PP2

    llama_static_auto_recompute_bs8_fp32_DP1-MP1-PP1
    llama_static_auto_recompute_bs16_fp32_DP2-MP1-PP1
    llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP1
    llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP2
    llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP2-VPP2-Sharding2_stage2
    llama_static_auto_recompute_bs16_fp16_DP2-MP2-PP2-VPP2-Sharding2_stage2
}

function gpt_case_list_auto_pir() {
    gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8_pir
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_pir
    gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1_pir
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1_pir
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2_pir
    gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3_pir
}

############ case start ############

function gpt_auto_recompute_bs16_fp32_DP1-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=$log_dir --devices=0 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=16 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=False \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.507633305
    ips_base=3518
    mem_base=11750.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=8 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.7 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.570028400
    ips_base=35050
    mem_base=1988.9
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP1-MP1-PP8_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=8 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.7 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.570028400
    ips_base=35050
    mem_base=1988.9
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP1-MP2-PP4() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=16 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=1 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.7 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.700293922
    ips_base=32518
    mem_base=1535.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.672543240
    ips_base=18681
    mem_base=2135.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.672543240
    ips_base=18681
    mem_base=2135.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.720068359
    ips_base=15232
    mem_base=1999.2
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage1_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.720068359
    ips_base=15232
    mem_base=1999.2
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.720078850
    ips_base=15571
    mem_base=1999.2
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP4-MP2-Sharding4_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=4 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=4 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=1 \
        -o Distributed.sharding.sharding_degree=4 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.681921577
    ips_base=13813
    mem_base=1747.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.579057693
    ips_base=19822
    mem_base=1709.8
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage1_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.579057693
    ips_base=19822
    mem_base=1709.8
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.579057693
    ips_base=20170
    mem_base=1709.8
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP1-PP4_Sharding2_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=2 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=1 \
        -o Distributed.pp_degree=4 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.3 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.585316849
    ips_base=15742
    mem_base=1591.6
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.672568035
    ips_base=19461
    mem_base=1384.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.672568035
    ips_base=19652
    mem_base=1384.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage2_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.672568035
    ips_base=19652
    mem_base=1384.7
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.696336079
    ips_base=16613
    mem_base=1280.5
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_recompute_bs16_fp16_o2_DP2-MP2-PP2_Sharding2_stage3_pir() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=mylog
    rm -rf $log_dir
    export FLAGS_enable_pir_in_executor=true
    python -m paddle.distributed.launch --log_dir=./mylog --devices=0,1,2,3,4,5,6,7 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Global.global_batch_size=16 \
        -o Global.local_batch_size=8 \
        -o Global.micro_batch_size=4 \
        -o Distributed.dp_degree=2 \
        -o Distributed.mp_degree=2 \
        -o Distributed.pp_degree=2 \
        -o Distributed.sharding.sharding_degree=2 \
        -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=True \
        -o Engine.mix_precision.level="o2" \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.logging_freq=10 \
        -o Profiler_auto.memory_stats=True \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $log_dir/workerlog.2 | grep '29/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'ips: ' '{print $2}' | awk -F ' tokens/s,' '{print $1}'`
    mem=`cat $log_dir/workerlog.0 | grep '29/30' | awk -F 'max_memory_reserved: ' '{print $2}' | awk -F ' MB,' '{print $1}'`
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.696336079
    ips_base=16613
    mem_base=1280.5
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_auto_sp_acc_check() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_infer_spmd_enable=true
    export FLAGS_call_stack_level=2
    mp_degree=2
    dp_degree=1
    pp_degree=1
    local_batch_size=1

    # sp on
    sp=True
    log_dir_spTrue=./${FUNCNAME}_mp${mp_degree}_sp${sp}
    rm -rf ./${log_dir_spTrue}/*
    python -m paddle.distributed.launch --log_dir=${log_dir_spTrue} --devices=0,1 --rank 0 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
        -o Model.hidden_size=1024 \
        -o Model.num_layers=12 \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Optimizer.grad_clip.clip_norm=0 \
        -o Global.local_batch_size=$(($local_batch_size / $dp_degree)) \
        -o Global.micro_batch_size=$(($local_batch_size / $dp_degree)) \
        -o Distributed.dp_degree=${dp_degree} \
        -o Distributed.mp_degree=${mp_degree} \
        -o Distributed.pp_degree=${pp_degree} \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=False \
        -o Engine.mix_precision.level=o2 \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.verbose=3 \
        -o Engine.logging_freq=1 \
        -o Engine.save_load.output_dir="" \
        -o Model.sequence_parallel=${sp} \
        >>${log_path}/$FUNCNAME 2>&1

    # sp off
    sp=False
    log_dir_spFalse=./${FUNCNAME}_mp${mp_degree}_sp${sp}
    rm -rf ./${log_dir_spFalse}/*
    python -m paddle.distributed.launch --log_dir=${log_dir_spFalse} --devices=0,1 --rank 0 tools/auto.py \
        -c ppfleetx/configs/nlp/gpt/auto/pretrain_gpt_1.3B_dp8.yaml \
        -o Model.hidden_size=1024 \
        -o Model.num_layers=12 \
        -o Model.hidden_dropout_prob=0 \
        -o Model.attention_probs_dropout_prob=0 \
        -o Model.use_recompute=True \
        -o Optimizer.grad_clip.clip_norm=0 \
        -o Global.local_batch_size=$(($local_batch_size / $dp_degree)) \
        -o Global.micro_batch_size=$(($local_batch_size / $dp_degree)) \
        -o Distributed.dp_degree=${dp_degree} \
        -o Distributed.mp_degree=${mp_degree} \
        -o Distributed.pp_degree=${pp_degree} \
        -o Distributed.sharding.sharding_degree=1 \
        -o Distributed.sharding.sharding_stage=1 \
        -o Distributed.pipeline.schedule_mode=1F1B \
        -o Engine.mix_precision.enable=False \
        -o Engine.mix_precision.level=o2 \
        -o Engine.max_steps=30 \
        -o Engine.eval_freq=100000 \
        -o Engine.verbose=3 \
        -o Engine.logging_freq=1 \
        -o Engine.save_load.output_dir="" \
        -o Model.sequence_parallel=${sp} \
        >>${log_path}/$FUNCNAME 2>&1
    
    # loss diff
    loss=`cat ${log_dir_spTrue}/workerlog.0 |  grep '30/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    loss_base=`cat ${log_dir_spFalse}/workerlog.0 |  grep '30/30' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips_base=-1
    mem_base=-1
    echo "result: loss_spTrue=$loss loss_spFasle=$loss_base"
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs8_fp32_DP1-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs8_dp1mp1pp1"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2"  \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --sharding "stage1" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.52110565
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs16_fp32_DP2-MP1-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs16_dp2mp1pp1"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2"  \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --sharding "stage1" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.42011833
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs16_dp2mp2pp1"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2"  \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --sharding "stage1" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.44299471
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs16_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2"  \
        --scale_loss 1024 \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --sharding "stage1" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.4 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.45936012
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs16_fp32_DP2-MP2-PP2-VPP2-Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs16_dp2mp2pp2vpp2sharding2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2"  \
        --scale_loss 1024 \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --virtual_pp_degree 2 \
        --pipeline_schedule_mode "VPP" \
        --sharding_parallel_degree 2 \
        --sharding "stage2" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.4 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.46707726
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_static_auto_recompute_bs16_fp16_DP2-MP2-PP2-VPP2-Sharding2_stage2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=2

    task_name="llama_auto_bs16_fp16_dp2mp2pp2vpp2sharding2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto_static.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 1 \
        --fp16_opt_level "O2"  \
        --amp_master_grad 1 \
        --scale_loss 1024 \
        --tensor_parallel_degree 2 \
        --pipeline_parallel_degree 2 \
        --virtual_pp_degree 2 \
        --pipeline_schedule_mode "VPP" \
        --sharding_parallel_degree 2 \
        --sharding "stage2" \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --max_grad_norm 1.0 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --eval_steps 1000 \
        --report_to "visualdl" \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 1 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.4 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=10.0859375
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp32_DP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 1 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.53389835
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp32_DP2-MP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2mp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 1 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.39066124
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp32_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 0 \
        --fp16_opt_level "O2" \
        --scale_loss 1024 \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.38235474
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}

function llama_dygraph_auto_bs8_fp16_DP2-MP2-PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    export PYTHONPATH=$root_path/:$PYTHONPATH
    export FLAGS_call_stack_level=3
    export NVIDIA_TF32_OVERRIDE=0

    task_name="llama_auto_bs8_fp16_dp2mp2pp2"
    case_out_dir="output/$task_name"
    case_log_dir="output/$task_name""_log"
    rm -rf $case_out_dir
    rm -rf $case_log_dir

    python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" --log_dir $case_log_dir run_pretrain_auto.py \
        --model_type "llama" \
        --model_name_or_path "facebook/llama-7b" \
        --tokenizer_name_or_path "facebook/llama-7b" \
        --input_dir "./data" \
        --output_dir $case_out_dir \
        --split 949,50,1 \
        --max_seq_length 2048 \
        --hidden_size 1024 \
        --intermediate_size 3072 \
        --num_hidden_layers 8 \
        --num_attention_heads 32 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --use_flash_attention 0 \
        --use_fused_rms_norm 0 \
        --fp16 1 \
        --fp16_opt_level "O2" \
        --amp_master_grad 1 \
        --scale_loss 1024 \
        --pipeline_parallel_degree 2 \
        --tensor_parallel_degree 2 \
        --sharding_parallel_degree 1 \
        --learning_rate 0.0001 \
        --min_learning_rate 0.00001 \
        --max_steps 10 \
        --save_steps 5000 \
        --weight_decay 0.01 \
        --warmup_ratio 0.01 \
        --logging_steps 1 \
        --dataloader_num_workers 1 \
        --sharding "" \
        --eval_steps 1000000 \
        --disable_tqdm true \
        --continue_training 0 \
        --recompute 0 \
        --do_train \
        --do_eval \
        --device "gpu" \
        --data_impl "mmap" \
        --enable_auto_parallel 1 \
        --to_static 0 \
        --max_grad_norm 1.0 \
        >>${log_path}/$FUNCNAME 2>&1
    loss=`cat $case_log_dir/workerlog.0 | grep 'global_step: 10' | awk -F 'loss: ' '{print $2}' | awk -F ',' '{print $1}'`
    ips=-1
    mem=-1
    echo "result: loss=$loss ips=$ips mem=$mem"
    loss_base=9.38256836
    ips_base=-1
    mem_base=-1
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}
############ case end ############

function check_result() {
    echo -e "$1" >> ${log_path}/result.log
    if [ $? -ne 0 ];then
        echo -e "\033[31m $1 run failed! \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi

    if [ $# -ne 7 ]; then
        echo -e "\033[31m $1 parameter transfer failed: $@ \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi

    diff_loss=$(echo $2 $3|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "loss_base: $2 loss_test: $3 loss_diff: $diff_loss%" | tee -a ${log_path}/result.log
    if [ $2 != $3 ];then
        echo -e "\033[31m $1 loss diff check failed! \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi

    diff_ips=$(echo $4 $5|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "ips_base: $4 ips_test: $5 ips_diff: $diff_ips% " | tee -a $log_path/result.log
    v1=$(echo $diff_ips 5.0|awk '{print($1>=$2)?"0":"1"}')
    v2=$(echo $diff_ips -5.0|awk '{print($1<=$2)?"0":"1"}')
    if [[ $v1 == 0 ]];then
        echo -e "$1 IPS increase greater than 5%, not exit " | tee -a $log_path/result.log
    fi
    if [[ $v2 == 0 ]];then
        echo -e "\033[31m $1 IPS diff check failed! \033[0m" | tee -a $log_path/result.log
        exit -1
    fi

    diff_mem=$(echo $6 $7|awk '{printf "%0.2f\n", ($2-$1)/$1*100}')
    echo -e "mem_base: $6 mem_test: $7 mem_diff: $diff_mem% " | tee -a $log_path/result.log
    w1=$(echo $diff_mem 5.0|awk '{print($1>=$2)?"0":"1"}')
    w2=$(echo $diff_mem -5.0|awk '{print($1<=$2)?"0":"1"}')
    if [[ $w1 == 0 ]];then
        echo -e "\033[31m $1 MEM diff check failed! \033[0m" | tee -a $log_path/result.log
        exit -1
    fi
    if [[ $w2 == 0 ]];then
        echo -e "$1 MEM decreases greater than 5%, not exit " | tee -a $log_path/result.log
    fi
}

function before_hook_for_gpt() {
    echo -e "\033[31m ---- Set FLAGS for GPT auto cases  \033[0m"
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    if [[ $FLAGS_install_deps == 0 ]];then
        echo -e "\033[31m ---- Install requirements for GPT auto cases  \033[0m"
        python -m pip install -r requirements.txt --force-reinstall
        python -m pip install --no-cache-dir https://paddlenlp.bj.bcebos.com/wheels/paddlenlp-ci-py3-none-any.whl --force-reinstall --no-dependencies
        python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
    else
        echo -e "\033[31m ---- Skip install requirements for GPT auto cases  \033[0m"
    fi
    if [[ ! $FLAGS_download_data =~ "gpt" ]];then
        echo -e "\033[31m ---- Download GPT data  \033[0m"
        rm -rf data
        if [[ -e ${gpt_data_path}/data ]]; then
            echo "GPT data downloaded"
        else
            # download data for gpt
            mkdir ${gpt_data_path}/data;
            wget -O ${gpt_data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
            wget -O ${gpt_data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
        fi
        cp -r ${gpt_data_path}/data ${gpt_case_path}/
    else
        echo -e "\033[31m ---- Skip download gpt data \033[0m"
    fi
}

function before_hook_for_llama() {
    echo -e "\033[31m ---- Set FLAGS for LLaMA auto cases  \033[0m"
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    export FLAGS_program_topo_reorder=1            # 1: 反向对齐动手拓扑排序
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    if [[ ! $FLAGS_download_data =~ "llama" ]];then
        echo -e "\033[31m ---- Download LLaMA data  \033[0m"
        rm -rf data
        if [[ -e ${llama_data_path}/data ]]; then
            echo "LLaMA data downloaded"
        else
            # download data for llama
            mkdir ${llama_data_path}/data;
            wget -O ${llama_data_path}/data/llama_openwebtext_100k_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_ids.npy;
            wget -O ${llama_data_path}/data/llama_openwebtext_100k_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k_idx.npz;
        fi
        cp -r ${llama_data_path}/data ${llama_case_path}/
    else
        echo -e "\033[31m ---- Skip download LLaMA data \033[0m"
    fi
}

echo -e "\033[31m ---- Start executing $1 \033[0m"
export exec_case=$1
export FLAGS_install_deps=$2
export FLAGS_download_data=$3

if [[ $exec_case =~ "gpt" ]];then
    cd ${gpt_case_path}
    before_hook_for_gpt
elif [[ $exec_case =~ "llama" ]];then
    cd ${llama_case_path}
    before_hook_for_llama
else
    echo -e "\033[31m ---- Invalid exec_case $exec_case \033[0m"
fi

$1
