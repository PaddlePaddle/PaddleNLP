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

export nlp_dir=/workspace/PaddleNLP/
export log_path=/workspace/PaddleNLP/model_logs
export case_path=/workspace/PaddleNLP/model_zoo/gpt-3
export data_path=/fleetx_data

unset CUDA_VISIBLE_DEVICES

function case_list_auto() {
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
    loss_base=10.570029736
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
    loss_base=10.700303650
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
    loss_base=10.672552681
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
    loss_base=10.720057583
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
    loss_base=10.720069408
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
    loss_base=10.681925678
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
    loss_base=10.579048920
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
    loss_base=10.579052830
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
    loss_base=10.585315990
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
    loss_base=10.672563076
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
    loss_base=10.672573280
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
    loss_base=10.696338844
    ips_base=16613
    mem_base=1280.5
    check_result $FUNCNAME ${loss_base} ${loss} ${ips_base} ${ips} ${mem_base} ${mem}
    echo "=========== $FUNCNAME run  end ==========="
}
############ case end ############

function check_result() {
    echo -e "$1" | tee -a ${log_path}/result.log
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
        echo -e " $1 IPS increase greater than 5%, not exit " | tee -a $log_path/result.log
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
        echo -e " $1 MEM decreases greater than 5%, not exit " | tee -a $log_path/result.log
    fi
}

main() {
    echo -e "\033[31m ---- Start executing auto_parallel case \033[0m"
    cd ${case_path}
    case_list_auto
}

main$@
