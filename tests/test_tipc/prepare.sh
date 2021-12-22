#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1

# MODE be one of ['lite_train_lite_infer' 'lite_train_whole_infer' 'whole_train_whole_infer',  
#                 'whole_infer', 'klquant_whole_infer',
#                 'cpp_infer', 'serving_infer']
# PaddleNLP supports 'lite_train_lite_infer', 'lite_train_whole_infer', 'whole_train_whole_infer' and 
# 'whole_infer' mode now.

MODE=$2

dataline=$(cat ${FILENAME})

# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd .. 
    fi
elif [ ${MODE} = "whole_train_whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
    fi
elif [ ${MODE} = "lite_train_whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
    fi
elif [ ${MODE} = "whole_infer" ];then
    if [ ${model_name} == "bigru_crf" ]; then
        rm -rf ./data/lexical_analysis_dataset_tiny ./data/lexical_analysis_dataset_tiny.tar.gz
        wget -nc -P ./data/ https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz --no-check-certificate
        cd ./data/ && tar xfz lexical_analysis_dataset_tiny.tar.gz && cd ..
        # Download static model
        rm -rf ./test_tipc/bigru_crf/infer_model
        wget -nc -P ./test_tipc/bigru_crf/ https://bj.bcebos.com/paddlenlp/models/bigru_crf_infer_model.tgz  --no-check-certificate
        cd ./test_tipc/bigru_crf && tar xfz bigru_crf_infer_model.tgz && cd ../..
    fi
fi
