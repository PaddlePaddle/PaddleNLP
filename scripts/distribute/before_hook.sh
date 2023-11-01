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


function gpt-3() {
    export case_path=/workspace/PaddleNLP/model_zoo/gpt-3
    export data_path=/fleetx_data

    cd ${case_path}
    echo -e "\033[31m ---- Set FLAGS  \033[0m"
    export FLAGS_new_executor_micro_batching=True  # True：打开新执行器
    export FLAGS_embedding_deterministic=1         # 1：关闭随机性
    export FLAGS_cudnn_deterministic=1             # 1：关闭随机性
    unset CUDA_MODULE_LOADING
    env | grep FLAGS
    echo -e "\033[31m ---- Install requirements  \033[0m"
    export http_proxy=${proxy}
    export https_proxy=${proxy}
    python -m pip install -r requirements.txt --force-reinstall
    
    cd ppfleetx/ops && python setup_cuda.py install && cd ../..
    python -m pip install numpy==1.22.4 --force-reinstall
    python -c "import paddlenlp; print('paddlenlp commit:',paddlenlp.version.commit)";

    echo -e "\033[31m ---- download data  \033[0m"
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

    rm -rf data
    if [[ -e ${data_path}/data ]]; then
        echo "data downloaded"
    else
        # download data for gpt
        mkdir ${data_path}/data;
        wget -O ${data_path}/data/gpt_en_dataset_300m_ids.npy https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy;
        wget -O ${data_path}/data/gpt_en_dataset_300m_idx.npz https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz;
    fi

    rm -rf dataset
    if [[ -e ${data_path}/dataset/wikitext_103_en ]]; then
        echo "dataset/wikitext_103_en downloaded"
    else
        # download dataset/wikitext_103_en
        mkdir ${data_path}/dataset/wikitext_103_en;
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
    cp -r ${data_path}/data ${case_path}/
    cp -r ${data_path}/dataset ${case_path}/
    ln -s ${data_path}/wikitext-103 ${case_path}/wikitext-103
    cp ${data_path}/lambada_test.jsonl ${case_path}/
    ln -s ${data_path}/pretrained ${case_path}/pretrained
    ln -s ${data_path}/GPT_345M_QAT_wo_analysis ${case_path}/GPT_345M_QAT_wo_analysis
}

$1
