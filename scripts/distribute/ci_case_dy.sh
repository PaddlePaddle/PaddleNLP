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
export case_path=/workspace/PaddleNLP/model_zoo/gpt-3
export data_path=/fleetx_data

unset CUDA_VISIBLE_DEVICES

function case_list_dygraph(){
    gpt_preprocess_data
    gpt_345M_single
    gpt_1.3B_dp
    gpt_6.7B_stage2_dp2_sharding4
    gpt_6.7B_stage3_dp2_sharding4
    gpt_6.7B_stage2_sharding8
    gpt_175B_DP1_MP4_PP2
    gpt_175B_DP1_MP4_PP2_sp
    gpt_175B_DP1_MP8_PP1
    gpt_175B_DP1_MP8_PP1_sp
    gpt_175B_DP1_MP1_PP8
    gpt_generation_345M_single
    gpt_generation_345M_hybrid
    gpt_345M_mp8_qat
    gpt_export_345M_mp1
    # gpt_export_345M_mp2
    # gpt_export_qat_345M
    gpt_inference_345M_single
    gpt_inference_345M_dp8
    gpt_345M_single_finetune
    gpt_eval_WikiText
    gpt_eval_LAMBADA
}


############ case start ############
function gpt_preprocess_data() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ppfleetx/data/data_tools/gpt/raw_trans_to_json.py  \
        --input_path ./dataset/wikitext_103_en \
        --output_path ./dataset/wikitext_103_en/wikitext_103_en \
        >>${log_path}/$FUNCNAME 2>&1
    python ppfleetx/data/data_tools/gpt/preprocess_data.py \
        --model_name gpt2 \
        --tokenizer_name GPTTokenizer \
        --data_format JSON \
        --input_path ./dataset/wikitext_103_en/wikitext_103_en.jsonl \
        --append_eos \
        --output_prefix ./dataset/wikitext_103_en/wikitext_103_en  \
        --workers 40 \
        --log_interval 1000 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python tools/train.py \
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_345M_single_card.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_1.3B_dp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_1.3B_dp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage2_dp2_sharding4() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage3_dp2_sharding4() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=4 -o Distributed.sharding.sharding_stage=3 \
        -o Distributed.sharding.reduce_overlap=False -o Distributed.sharding.broadcast_overlap=False \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_6.7B_stage2_sharding8() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" \
        tools/train.py -c ppfleetx/configs/nlp/gpt/pretrain_gpt_6.7B_sharding16.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=20 -o Engine.eval_freq=20 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Distributed.sharding.sharding_degree=8 -o Distributed.sharding.sharding_stage=2 \
        -o Distributed.sharding.reduce_overlap=True -o Distributed.sharding.broadcast_overlap=True \
        -o Engine.logging_freq=5 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP4_PP2() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP4_PP2_sp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=4 -o Model.num_attention_heads=4 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=4 -o Distributed.pp_degree=2 -o Model.sequence_parallel=True \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP8_PP1() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP8_PP1_sp() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=16 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=2 \
        -o Distributed.mp_degree=8 -o Distributed.pp_degree=1 -o Model.sequence_parallel=True \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_175B_DP1_MP1_PP8() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml \
        -o Model.hidden_size=1024 -o Model.num_layers=32 -o Model.num_attention_heads=16 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        -o Global.local_batch_size=16 -o Global.micro_batch_size=1 \
        -o Distributed.mp_degree=1 -o Distributed.pp_degree=8 \
        -o Model.virtual_pp_degree=2 -o Distributed.pp_recompute_interval=2 \
        -o Model.fused_linear=True -o Model.use_recompute=True \
        -o Model.sequence_parallel=False \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_mp8_qat() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" tools/train.py\
        -c ppfleetx/configs/nlp/gpt/qat_gpt_345M_mp8.yaml \
        -o Model.num_layers=4 -o Model.num_attention_heads=8 \
        -o Engine.max_steps=10 -o Engine.eval_freq=10 \
        -o Engine.eval_iters=5 -o Engine.save_load.save_steps=10 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_generation_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_generation_345M_hybrid() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python -m paddle.distributed.launch --devices "0" tasks/gpt/generation.py \
        -c ppfleetx/configs/nlp/gpt/generation_gpt_345M_dp8.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_345M_mp1() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    export PYTHONPATH=/workspace/PaddleNLP/model_zoo/gpt-3:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=1
    python -m paddle.distributed.launch --log_dir $log_dir --devices "1" \
        ./tools/auto_export.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./pretrained/inference_model \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "1" \
        projects/gpt/inference.py --mp_degree 1 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    unset CUDA_VISIBLE_DEVICES
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_345M_mp2() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    export PYTHONPATH=/workspace/PaddleNLP/model_zoo/gpt-3:$PYTHONPATH
    export CUDA_VISIBLE_DEVICES=0,1
    python -m paddle.distributed.launch --devices "0,1" \
        ./tools/auto_export.py \
        -c ./ppfleetx/configs/nlp/gpt/auto/generation_gpt_345M_mp2.yaml \
        -o Generation.use_topp_sampling=False \
        -o Engine.save_load.ckpt_dir=./pretrained/inference_model \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0,1" \
        projects/gpt/inference.py --mp_degree 2 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    unset CUDA_VISIBLE_DEVICES
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_export_qat_345M() {
    echo "=========== $FUNCNAME run begin ==========="
    log_dir=log_export
    rm -rf $log_dir
    rm -rf output

    python ./tools/export.py \
        -c ./ppfleetx/configs/nlp/gpt/generation_qat_gpt_345M_single_card.yaml \
        -o Model.hidden_dropout_prob=0.0 \
        -o Model.attention_probs_dropout_prob=0.0 \
        -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_wo_analysis/' \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0" \
        projects/gpt/inference.py --mp_degree 1 --model_dir output \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_inference_345M_single() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    rm -rf output
    python tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    python tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_inference_345M_dp8() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    rm -rf output
    python -m paddle.distributed.launch --devices "0" tools/export.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826/ \
        >>${log_path}/$FUNCNAME 2>&1
    python -m paddle.distributed.launch --devices "0" \
        tasks/gpt/inference.py \
        -c ppfleetx/configs/nlp/gpt/inference_gpt_345M_single_card.yaml \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_345M_single_finetune() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/train.py \
        -c ./ppfleetx/configs/nlp/gpt/finetune_gpt_345M_single_card_glue.yaml \
        -o Engine.num_train_epochs=1 \
        -o Data.Train.dataset.name=WNLI \
        -o Data.Train.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.name=WNLI \
        -o Data.Eval.dataset.root=./dataset/WNLI/ \
        -o Data.Eval.dataset.split=dev \
        -o Model.num_classes=2 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_eval_WikiText() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./wikitext-103/wiki.valid.tokens \
        -o Offline_Eval.overlapping_eval=32 \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}

function gpt_eval_LAMBADA() {
    echo "=========== $FUNCNAME run begin ==========="
    rm -rf log
    python ./tools/eval.py \
        -c ./ppfleetx/configs/nlp/gpt/eval_gpt_345M_single_card.yaml \
        -o Engine.save_load.ckpt_dir=./ckpt/PaddleFleetX_GPT_345M_220826 \
        -o Offline_Eval.eval_path=./lambada_test.jsonl \
        -o Offline_Eval.cloze_eval=True \
        -o Offline_Eval.batch_size=16 \
        -o Engine.max_steps=20 \
        >>${log_path}/$FUNCNAME 2>&1
    check_result $FUNCNAME
    echo "=========== $FUNCNAME run  end ==========="
}
############ case end ############

function check_result() {
    if [ $? -ne 0 ];then
        echo -e "\033[31m $1 run failed! \033[0m" | tee -a ${log_path}/result.log
        exit -1
    fi
}

function before_hook() {
    echo -e "\033[31m ---- Set FLAGS  \033[0m"
    env | grep FLAGS
    if [[ $FLAGS_before_hook == 0 ]];then
        echo -e "\033[31m ---- Install requirements  \033[0m"
        export http_proxy=${proxy}
        export https_proxy=${proxy}
        python -m pip install -r requirements.txt --force-reinstall
        python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";
        
        echo -e "\033[31m ---- download data  \033[0m"
        rm -rf data
        if [[ -e ${data_path}/data ]]; then
            echo "data downloaded"
        else
            # download data for gpt
            mkdir ${data_path}/data;
            wget -O ${data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
            wget -O ${data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
        fi
        cp -r ${data_path}/data ${case_path}/
    else
        echo -e "\033[31m ---- Skip install requirements \033[0m"
        echo -e "\033[31m ---- Install ppfleetx/ops  \033[0m"
        cd ppfleetx/ops && python setup_cuda.py install && cd ../..

        echo -e "\033[31m ---- download other data  \033[0m"
        rm -rf ckpt
        if [[ -e ${data_path}/ckpt/PaddleFleetX_GPT_345M_220826 ]]; then
            echo "ckpt/PaddleFleetX_GPT_345M_220826 downloaded"
        else
            # download ckpt for gpt
            mkdir -p ${data_path}/ckpt
            wget -O ${data_path}/ckpt/GPT_345M.tar.gz \
                https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
            tar -xzf ${data_path}/ckpt/GPT_345M.tar.gz -C ${data_path}/ckpt
            rm -rf ${data_path}/ckpt/GPT_345M.tar.gz
        fi

        rm -rf dataset
        if [[ -e ${data_path}/dataset/wikitext_103_en ]]; then
            echo "dataset/wikitext_103_en downloaded"
        else
            # download dataset/wikitext_103_en
            mkdir -p ${data_path}/dataset/wikitext_103_en;
            wget -O ${data_path}/dataset/wikitext_103_en/wikitext-103-en.txt http://fleet.bj.bcebos.com/datasets/gpt/wikitext-103-en.txt
        fi

        rm -rf wikitext-103
        if [[ -e ${data_path}/wikitext-103 ]]; then
            echo "wikitext-103 downloaded"
        else
            # download wikitext-103 for gpt eval
            wget -O ${data_path}/wikitext-103-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
            unzip -q ${data_path}/wikitext-103-v1.zip -d ${data_path}/
            rm -rf ${data_path}/wikitext-103-v1.zip
        fi

        rm -rf lambada_test.jsonl
        if [[ -e ${data_path}/lambada_test.jsonl ]]; then
            echo "lambada_test.jsonl downloaded"
        else
            # download lambada_test.jsonl for gpt eval
            wget -O ${data_path}/lambada_test.jsonl https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
        fi

        rm -rf pretrained
        if [[ -e ${data_path}/pretrained ]]; then
            echo "GPT_345M_FP16 downloaded"
        else
            # download GPT_345M_FP16 for gpt export
            wget -O ${data_path}/GPT_345M_FP16.tar.gz https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_FP16.tar.gz
            tar -zxvf ${data_path}/GPT_345M_FP16.tar.gz -C ${data_path}/
            rm -rf ${data_path}/GPT_345M_FP16.tar.gz
        fi

        rm -rf GPT_345M_QAT_wo_analysis
        if [[ -e ${data_path}/GPT_345M_QAT_wo_analysis ]]; then
            echo "GPT_345M_QAT_wo_analysis downloaded"
        else
            # download GPT_345M_QAT_wo_analysis for gpt qat
            wget -O ${data_path}/GPT_345M_QAT_wo_analysis.tar https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_wo_analysis.tar
            tar xf ${data_path}/GPT_345M_QAT_wo_analysis.tar -C ${data_path}/
            rm -rf ${data_path}/GPT_345M_QAT_wo_analysis.tar
        fi

        ln -s ${data_path}/ckpt ${case_path}/ckpt
        cp -r ${data_path}/dataset ${case_path}/
        ln -s ${data_path}/wikitext-103 ${case_path}/wikitext-103
        cp ${data_path}/lambada_test.jsonl ${case_path}/
        ln -s ${data_path}/pretrained ${case_path}/pretrained
        ln -s ${data_path}/GPT_345M_QAT_wo_analysis ${case_path}/GPT_345M_QAT_wo_analysis
    fi
}

echo -e "\033[31m ---- Start executing gpt-3 $1 \033[0m"
cd ${case_path}
export FLAGS_before_hook=$2
before_hook
$1
