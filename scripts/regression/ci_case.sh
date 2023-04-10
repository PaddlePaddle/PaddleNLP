#!/usr/bin/env bash
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

export nlp_dir=${PWD}
export log_path=${nlp_dir}/model_logs
export cudaid1=$2
export cudaid2=$3
export PATH=${PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
if [ ! -d "model_logs" ];then
    mkdir model_logs
fi
if [ ! -d "unittest_logs" ];then
    mkdir model_logs
fi

print_info(){
if [ $1 -ne 0 ];then
    if [[ $2 =~ 'tests' ]];then
        mv ${nlp_dir}/unittest_logs/$3.log ${nlp_dir}/unittest_logs/$3_FAIL.log
        echo -e "\033[31m ${nlp_dir}/unittest_logs/$3_FAIL \033[0m"
        cat ${nlp_dir}/unittest_logs/$3_FAIL.log
    else
        mv ${log_path}/$2 ${log_path}/$2_FAIL.log
        echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
        cat ${log_path}/$2_FAIL.log
    fi
elif [[ $2 =~ 'tests' ]];then
    echo -e "\033[32m ${log_path}/$3_SUCCESS \033[0m"
else
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
# case list
# 1 waybill_ie (无可控参数，数据集外置)
waybill_ie(){
cd ${nlp_dir}/examples/information_extraction/waybill_ie/
export CUDA_VISIBLE_DEVICES=${cudaid1}
# BiGRU +CRF star training
time (
python download.py --data_dir ./waybill_ie
python run_bigru_crf.py >${log_path}/waybill_ie_bigru_crf) >>${log_path}/waybill_ie_bigru_crf 2>&1
print_info $? waybill_ie_bigru_crf
# ERNIE +RF star training
time (python run_ernie.py >${log_path}/waybill_ie_ernie) >>${log_path}/waybill_ie_ernie 2>&1
print_info $? waybill_ie_ernie
# ERNIE +CRF star training
time (python run_ernie_crf.py >${log_path}/waybill_ie_ernie_crf) >>${log_path}/waybill_ie_ernie_crf 2>&1
print_info $? waybill_ie_ernie_crf
}
# 2 msra_ner （不可控，内置）
msra_ner(){
cd ${nlp_dir}/examples/information_extraction/msra_ner/
export CUDA_VISIBLE_DEVICES=${cudaid2}
## train
time (python -m paddle.distributed.launch  ./train.py \
    --model_type bert  \
    --model_name_or_path bert-base-multilingual-uncased \
    --dataset msra_ner \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --max_steps 2 \
    --save_steps 2 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu >${log_path}/msra_ner_train) >>${log_path}/msra_ner_train 2>&1
print_info $? msra_ner_train
## eval
time (python -u ./eval.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 16 \
    --device gpu \
    --init_checkpoint_path ./tmp/msra_ner/model_2.pdparams >${log_path}/msra_ner_eval) >>${log_path}/msra_ner_eval 2>&1
print_info $? msra_ner_eval
## predict
time (python -u ./predict.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 16 \
    --device gpu \
    --init_checkpoint_path ./tmp/msra_ner/model_2.pdparams >${log_path}/msra_ner_predict) >>${log_path}/msra_ner_predict 2>&1
print_info $? msra_ner_predict
}
# 3 glue
glue() {
cd ${nlp_dir}/examples/benchmark/glue/
export CUDA_VISIBLE_DEVICES=${cudaid2}
##  TASK_SST-2
export TASK_NAME=SST-2
time (python -u run_glue.py \
    --model_type bert    \
    --model_name_or_path bert-base-uncased    \
    --task_name $TASK_NAME \
    --max_seq_length 128   \
    --batch_size 128    \
    --learning_rate 3e-5    \
    --max_steps 1    \
    --logging_steps 1    \
    --save_steps 1   \
    --output_dir ./$TASK_NAME/    \
    --device gpu  >${log_path}/glue_${TASK_NAME}_train) >>${log_path}/glue_${TASK_NAME}_train 2>&1
print_info $? glue_${TASK_NAME}_train
}
# 4 bert
bert() {
export CUDA_VISIBLE_DEVICES=${cudaid2}
# cd ${nlp_dir}/model_zoo/bert/
# wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/bert.tar.gz
# tar -xzvf bert.tar.gz
cd ${nlp_dir}/model_zoo/bert/data/
wget -q https://bj.bcebos.com/paddlenlp/models/transformers/bert/data/training_data.hdf5
cd ../
# pretrain
time (python -m paddle.distributed.launch run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 16  \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --device gpu \
    --use_amp False >${log_path}/bert_pretrain) >>${log_path}/bert_pretrain 2>&1
print_info $? bert_pretrain
time (python -m paddle.distributed.launch run_glue_trainer.py \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --max_seq_length 128 \
    --per_device_train_batch_size 32   \
    --per_device_eval_batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --output_dir ./tmp/ \
    --device gpu \
    --fp16 False\
    --do_train \
    --do_eval >${log_path}/bert_fintune) >>${log_path}/bert_fintune 2>&1
print_info $? bert_fintune
time (python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model >${log_path}/bert_export) >>${log_path}/bert_export 2>&1
print_info $? bert_export
 }
# 5 skep (max save 不可控 内置)
skep () {
cd ${nlp_dir}/examples/sentiment_analysis/skep/
export CUDA_VISIBLE_DEVICES=${cudaid2}
## train_sentence
time ( python -m paddle.distributed.launch train_sentence.py --batch_size 16 --epochs 1 --model_name "skep_ernie_1.0_large_ch" --device gpu --save_dir ./checkpoints >${log_path}/skep_train_sentence) >>${log_path}/skep_train_sentence 2>&1
print_info $? skep_train_sentence
## train_aspect
time ( python -m paddle.distributed.launch train_aspect.py --batch_size 4 --epochs 1  --device gpu --save_dir ./aspect_checkpoints  >${log_path}/skep_train_aspect) >>${log_path}/skep_train_aspect 2>&1
print_info $? skep_train_aspect
# # train_opinion
time ( python -m paddle.distributed.launch train_opinion.py  --batch_size 4 --epochs 1 --device gpu --save_dir ./opinion_checkpoints >${log_path}/skep_train_opinion) >>${log_path}/skep_train_opinion 2>&1
print_info $? skep_train_opinion
# predict_sentence
time (python predict_sentence.py --model_name "skep_ernie_1.0_large_ch"  --ckpt_dir checkpoints/model_100 >${log_path}/skep_predict_sentence) >>${log_path}/skep_predict_sentence 2>&1
print_info $? skep_predict_sentence
## predict_aspect
time (python predict_aspect.py --device 'gpu' --ckpt_dir ./aspect_checkpoints/model_100  >${log_path}/skep_predict_aspect) >>${log_path}/skep_predict_aspect 2>&1
print_info $? skep_predict_aspect
# # predict_opinion
time (python predict_opinion.py --device 'gpu' --ckpt_dir ./opinion_checkpoints/model_100 >${log_path}/skep_predict_opinion) >>${log_path}/skep_predict_opinion 2>&1
print_info $? skep_predict_opinion
}
# 6 bigbird
bigbird(){
cd ${nlp_dir}/examples/language_model/bigbird/
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  --log_dir log  run_pretrain.py --model_name_or_path bigbird-base-uncased \
    --input_dir "./data" \
    --output_dir "output" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --max_encoder_length 512 \
    --max_pred_length 75 >${log_path}/bigbird_pretrain) >>${log_path}/bigbird_pretrain 2>&1
    print_info $? bigbird_pretrain
}
# 7 electra
electra(){
cd ${nlp_dir}/model_zoo/electra/
export CUDA_VISIBLE_DEVICES=${cudaid2}
export DATA_DIR=./BookCorpus/
wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/BookCorpus.tar.gz && tar -xzvf BookCorpus.tar.gz
time (python -u ./run_pretrain.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --input_dir ./BookCorpus/ \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 4 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --device gpu >${log_path}/electra_pretrain) >>${log_path}/electra_pretrain 2>&1
print_info $? electra_pretrain
}
fast_gpt(){
# FT
cd ${nlp_dir}/
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz
tar -zxf paddle_inference.tgz
cd ${nlp_dir}/paddlenlp/ops
export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++
#python
mkdir build_gpt_so
cd build_gpt_so/
cmake ..  -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python -DWITH_GPT=ON
make -j >${log_path}/GPT_python_FT >>${log_path}/gpt_python_FT 2>&1
print_info $? gpt_python_FT
cd ../
#c++
mkdir build_gpt_cc
cd build_gpt_cc/
cmake ..  -DWITH_GPT=ON -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=${nlp_dir}/paddle_inference/ -DDEMO=${nlp_dir}/paddlenlp/ops/fast_transformer/src/demo/gpt.cc -DON_INFER=ON -DWITH_MKL=ON -DWITH_ONNXRUNTIME=ON
make -j >${log_path}/GPT_C_FT >>${log_path}/gpt_C_FT 2>&1
print_info $? gpt_C_FT
#depoly python
cd ${nlp_dir}/model_zoo/gpt/fast_gpt/
python infer.py \
    --model_name_or_path gpt2-medium-en \
    --batch_size 1 \
    --topk 4 \
    --topp 0.0 \
    --max_length 32 \
    --start_token "<|endoftext|>" \
    --end_token "<|endoftext|>" \
    --temperature 1.0  >${log_path}/gpt_deploy_P_FT >>${log_path}/gpt_deploy_P_FT 2>&1
print_info $? gpt_deploy_P_FT
#depoly C++
python export_model.py \
    --model_name_or_path gpt2-medium-en \
    --decoding_lib ${nlp_dir}/paddlenlp/ops/build_gpt_so/lib/libdecoding_op.so \
    --topk 4 \
    --topp 0.0 \
    --max_out_len 32 \
    --temperature 1.0 \
    --inference_model_dir ./infer_model/
mv infer_model/ ${nlp_dir}/paddlenlp/ops/build_gpt_cc/bin/
cd ${nlp_dir}/paddlenlp/ops/build_gpt_cc/bin/
./gpt -batch_size 1 -gpu_id 0 -model_dir ./infer_model -vocab_file ./infer_model/vocab.txt -start_token "<|endoftext|>" -end_token "<|endoftext|>"  >${log_path}/gpt_deploy_C_FT >>${log_path}/gpt_deploy_C_FT 2>&1
print_info $? gpt_deploy_C_FT
}
# 8 gpt
gpt(){

# TODO(wj-Mcat): revert the gpt run_pretrain.py code, remove it later.

cd ${nlp_dir}/model_zoo/ernie-1.0/data_tools
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
#pretrain
cd ${nlp_dir}/model_zoo/gpt/
mkdir pre_data
cd ./pre_data
wget -q https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget -q https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
cd ../
time (python -m paddle.distributed.launch run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./pre_data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 2\
    --save_steps 2\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 2 \
    --device gpu >${log_path}/gpt_pretrain) >>${log_path}/gpt_pretrain 2>&1
print_info $? gpt_pretrain
time (
python export_model.py --model_type=gpt \
    --model_path=gpt2-medium-en \
    --output_path=./infer_model/model >${log_path}/gpt_export) >>${log_path}/gpt_export 2>&1
print_info $? gpt_export
time (
python deploy/python/inference.py \
    --model_type gpt \
    --model_path ./infer_model/model >${log_path}/gpt_p_depoly) >>${log_path}/gpt_p_depoly 2>&1
print_info $? gpt_p_depoly

echo 'run gpt test with pytest'
cd ${nlp_dir}
python -m pytest ./tests/model_zoo/test_gpt.py >${log_path}/gpt >>${log_path}/gpt 2>&1
print_info $? gpt

fast_gpt
cd ${nlp_dir}/fast_generation/samples
python gpt_sample.py >${log_path}/fast_generation_gpt >>${log_path}/fast_generation_gpt 2>&1
print_info $? fast_generation_gpt
}
# 9 ernie
ernie(){
#data process
cd ${nlp_dir}/model_zoo/ernie-1.0/data_tools
sed -i "s/python3/python/g" Makefile
sed -i "s/python-config/python3.7m-config/g" Makefile
export CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/model_zoo/ernie-1.0/
mkdir data && cd data
wget -q https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/ernie_wudao_0903_92M_ids.npy
wget -q https://paddlenlp.bj.bcebos.com/models/transformers/data_tools/ernie_wudao_0903_92M_idx.npz
cd ../
mkdir data_ernie_3.0 && cd data_ernie_3.0
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/wudao_200g_sample_ernie-3.0-base-zh_idx.npz
cd ../
# pretrain_trainer
python -u -m paddle.distributed.launch \
    --log_dir "output/trainer_log" \
    run_pretrain_trainer.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-3.0-base-zh" \
    --tokenizer_name_or_path "ernie-3.0-base-zh" \
    --input_dir "./data_ernie_3.0" \
    --output_dir "output/trainer_log" \
    --split 949,50,1 \
    --max_seq_length 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 2 \
    --save_steps 2 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 1\
    --dataloader_num_workers 4 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --do_train \
    --device "gpu" >${log_path}/ernie_1.0_pretrain_trainer >>${log_path}/ernie_1.0_pretrain_trainer 2>&1
    print_info $? ernie_1.0_pretrain_trainer
# pretrain_static
python -u -m paddle.distributed.launch \
    --log_dir "./log" \
    run_pretrain_static.py \
    --model_type "ernie" \
    --model_name_or_path "ernie-1.0-base-zh" \
    --tokenizer_name_or_path "ernie-1.0-base-zh" \
    --input_dir "./data/" \
    --output_dir "./output/" \
    --max_seq_len 512 \
    --micro_batch_size 16 \
    --global_batch_size 32 \
    --sharding_degree 1 \
    --dp_degree 2 \
    --use_sharding false \
    --use_amp true \
    --use_recompute false \
    --max_lr 0.0001 \
    --min_lr 0.00001 \
    --max_steps 4 \
    --save_steps 2 \
    --checkpoint_steps 5000 \
    --decay_steps 3960000 \
    --weight_decay 0.01 \
    --warmup_rate 0.0025 \
    --grad_clip 1.0 \
    --logging_freq 2\
    --num_workers 2 \
    --eval_freq 1000 \
    --device "gpu" >${log_path}/ernie_1.0_pretrain_static >>${log_path}/ernie_1.0_pretrain_static 2>&1
    print_info $? ernie_1.0_pretrain_static
}
# 10 xlnet
xlnet(){
cd ${nlp_dir}/examples/language_model/xlnet/
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./xlnet/ >${log_path}/xlnet_train) >>${log_path}/xlnet_train 2>&1
print_info $? xlnet_train
}
# 11 ofa
ofa(){
cd ${nlp_dir}/examples/model_compression/ofa/
cd ../../benchmark/glue/
export CUDA_VISIBLE_DEVICES=${cudaid2}
# finetuing
time (python -u run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./ \
    --device gpu  >${log_path}/ofa_pretrain) >>${log_path}/ofa_pretrain 2>&1
print_info $? ofa_pretrain
mv sst-2_ft_model_1.pdparams/  ${nlp_dir}/examples/model_compression/ofa/
cd -
#model slim
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch run_glue_ofa.py  \
          --model_type bert \
          --model_name_or_path ./sst-2_ft_model_1.pdparams/ \
          --task_name SST-2 --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 1     \
          --max_steps 1 \
          --logging_steps 1    \
          --save_steps 1     \
          --output_dir ./ofa/SST-2 \
          --device gpu  \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 >${log_path}/ofa_slim) >>${log_path}/ofa_slim 2>&1
print_info $? ofa_slim
}
# 12 albert
albert (){
cd ${nlp_dir}/examples/benchmark/glue/
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  run_glue.py \
        --model_type albert    \
        --model_name_or_path albert-base-v2    \
        --task_name SST-2 \
        --max_seq_length 128   \
        --batch_size 32    \
        --learning_rate 1e-5    \
        --max_steps 1    \
        --warmup_steps 1256    \
        --logging_steps 1    \
        --save_steps 1   \
        --output_dir ./albert/SST-2/    \
        --device gpu >${log_path}/albert_sst-2_train) >>${log_path}/albert_sst-2_train 2>&1
print_info $? albert_sst-2_train
}
# 13 squad
squad (){
cd ${nlp_dir}/examples/machine_reading_comprehension/SQuAD/
export CUDA_VISIBLE_DEVICES=${cudaid1}
# finetune
time (python -m paddle.distributed.launch run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict >${log_path}/squad_train) >>${log_path}/squad_train 2>&1
print_info $? squad_train
# export model
time (python  -u ./export_model.py \
    --model_type bert \
    --model_path ./tmp/squad/model_1/ \
    --output_path ./infer_model/model >${log_path}/squad_export) >>${log_path}/squad_export 2>&1
print_info $? squad_export
# predict
time (python -u deploy/python/predict.py \
    --model_type bert \
    --model_name_or_path ./infer_model/model \
    --batch_size 2 \
    --max_seq_length 384 >${log_path}/squad_predict) >>${log_path}/squad_predict 2>&1
print_info $? squad_predict
}
# 14 tinybert
tinybert() {
export CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${nlp_dir}/model_zoo/tinybert/
cp -r /ssd1/paddlenlp/download/tinybert/pretrained_models/ ./
#中间层蒸馏
time (python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path tinybert-6l-768d-v2 \
    --task_name SST-2 \
    --intermediate_distill \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path ./pretrained_models/SST-2/best_model_610/ \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./mid/SST-2/ \
    --device gpu >${log_path}/tinybert_midslim) >>${log_path}/tinybert_midslim 2>&1
print_info $? tinybert_midslim
#预测层蒸馏
time (python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path ./mid/SST-2/intermediate_distill_model_final.pdparams \
    --task_name SST-2 \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path ./pretrained_models/SST-2/best_model_610/  \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --max_steps 1 \
    --save_steps 1 \
    --output_dir ./ped/SST-2/ \
    --device gpu >${log_path}/tinybert_predslim) >>${log_path}/tinybert_predslim 2>&1
print_info $? tinybert_predslim
}
# 15 lexical_analysis
lexical_analysis(){
export CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/lexical_analysis/
#train
time (python download.py --data_dir ./ )
time (python -m paddle.distributed.launch train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 1 \
        --save_steps 15 \
        --logging_steps 1\
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_train) >>${log_path}/lexical_analysis_train 2>&1
print_info $? lexical_analysis_train
#export
time (python export_model.py \
    --data_dir=./lexical_analysis_dataset_tiny \
    --params_path=./save_dir/model_15.pdparams \
    --output_path=./infer_model/static_graph_params >${log_path}/lexical_analysis_export) >>${log_path}/lexical_analysis_export 2>&1
print_info $? lexical_analysis_export
# predict
time (python predict.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/model_15.pdparams \
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_predict) >>${log_path}/lexical_analysis_predict 2>&1
print_info $? lexical_analysis_predict
# deploy
time (python deploy/predict.py \
    --model_file=infer_model/static_graph_params.pdmodel \
    --params_file=infer_model/static_graph_params.pdiparams \
    --data_dir lexical_analysis_dataset_tiny >${log_path}/lexical_analysis_deploy) >>${log_path}/lexical_analysis_deploy 2>&1
print_info $? lexical_analysis_deploy
}
# 16 seq2seq
seq2seq() {
export CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/machine_translation/seq2seq/
# train  (1041/steps) 5min
time (python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --max_epoch 1 \
    --log_freq 1 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --device gpu \
    --model_path ./attention_models >${log_path}/seq2seq_train) >>${log_path}/seq2seq_train 2>&1
print_info $? seq2seq_train
# predict
time (python predict.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/0 \
     --infer_output_file infer_output.txt \
     --beam_size 10 \
     --device gpu  >${log_path}/seq2seq_predict) >>${log_path}/seq2seq_predict 2>&1
print_info $? seq2seq_predict
# export
time (python export_model.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/0.pdparams \
     --beam_size 10 \
     --export_path ./infer_model/model >${log_path}/seq2seq_export) >>${log_path}/seq2seq_export 2>&1
print_info $? seq2seq_export
# depoly
time (cd deploy/python
python infer.py \
    --export_path ../../infer_model/model \
    --device gpu \
    --batch_size 128 \
    --infer_output_file infer_output.txt  >${log_path}/seq2seq_depoly) >>${log_path}/seq2seq_deploy 2>&1
print_info $? seq2seq_depoly
}
# 18 word_embedding 5min
word_embedding(){
export CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${nlp_dir}/examples/word_embedding/
# 使用paddlenlp.embeddings.TokenEmbedding
time (python train.py --device='gpu' \
                --lr=5e-4 \
                --batch_size=32 \
                --epochs=1 \
                --use_token_embedding=True \
                --vdl_dir='./vdl_paddlenlp_dir'  >${log_path}/word_embedding_paddlenlp_train) >>${log_path}/word_embedding_paddlenlp_train 2>&1
print_info $? word_embedding_paddlenlp_train
# 使用paddle.nn.Embedding
time (python train.py --device='gpu' \
                --lr=1e-4 \
                --batch_size=32 \
                --epochs=1 \
                --use_token_embedding=False \
                --vdl_dir='./vdl_paddle_dir' >${log_path}/word_embedding_paddle_train) >>${log_path}/word_embedding_paddle_train 2>&1
print_info $? word_embedding_paddle_train
}
# 19 ernie-ctm
ernie-ctm(){
export CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${nlp_dir}/examples/text_to_knowledge/ernie-ctm/
wget https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/wordtag_dataset_v2.tar.gz && tar -zxvf wordtag_dataset_v2.tar.gz
time (python -m paddle.distributed.launch  train.py \
    --max_seq_len 128 \
    --batch_size 8   \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir ./output/ \
    --device "gpu"   >${log_path}/ernie-ctm_train) >>${log_path}/ernie-ctm_train 2>&1
print_info $? ernie-ctm_train
export CUDA_VISIBLE_DEVICES=${cudaid1}
time (python -m paddle.distributed.launch predict.py \
    --batch_size 32   \
    --params_path ./output/model_125/model_state.pdparams \
    --device "gpu"   >${log_path}/ernie-ctm_eval) >>${log_path}/ernie-ctm_eval 2>&1
print_info $? ernie-ctm_eval
}
# 20 distilbert
distilbert (){
cd ${nlp_dir}/examples/model_compression/distill_lstm/
wget -q https://paddle-qa.bj.bcebos.com/SST-2_GLUE.tar
tar -xzvf SST-2_GLUE.tar 
time (
    python small.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 1 \
    --batch_size 64 \
    --lr 1.0 \
    --dropout_prob 0.4 \
    --output_dir small_models/SST-2 \
    --save_steps 10000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en >${log_path}/distilbert_small_train) >>${log_path}/distilbert_small_train 2>&1
print_info $? distilbert_small_train
time (
    python bert_distill.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 1 \
    --lr 1.0 \
    --task_name sst-2 \
    --dropout_prob 0.2 \
    --batch_size 128 \
    --model_name bert-base-uncased \
    --output_dir distilled_models/SST-2 \
    --teacher_dir ./SST-2/sst-2_ft_model_1.pdparams/ \
    --save_steps 1000 \
    --n_iter 1 \
    --embedding_name w2v.google_news.target.word-word.dim300.en >${log_path}/distilbert_teacher_train) >>${log_path}/distilbert_teacher_train 2>&1
print_info $? distilbert_teacher_train
}
# 21 stacl
stacl() {
cd ${nlp_dir}/examples/simultaneous_translation/stacl/
cp -r /ssd1/paddlenlp/download/stacl/* ./
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (sed -i "s/save_step: 10000/save_step: 1/g" config/transformer.yaml
sed -i "s/p print_step: 100/print_step: 1/g" config/transformer.yaml
sed -i "s/epoch: 30/epoch: 1/g" config/transformer.yaml
sed -i "s/max_iter: None/max_iter: 3/g" config/transformer.yaml
sed -i "s/batch_size: 4096/batch_size: 500/g" config/transformer.yaml
python -m paddle.distributed.launch train.py --config ./config/transformer.yaml  >${log_path}/stacl_wk-1) >>${log_path}/stacl_wk-1 2>&1
print_info $? stacl_wk-1

time (sed -i "s/batch_size: 500/batch_size: 100/g" config/transformer.yaml
sed -i 's#init_from_params: "trained_models/step_final/"#init_from_params: "./trained_models/step_1/"#g' config/transformer.yaml
python predict.py --config ./config/transformer.yaml >${log_path}/stacl_predict) >>${log_path}/stacl_predict 2>&1
print_info $? stacl_predict
}
fast_transformer(){
# FT
cd ${nlp_dir}/
export PYTHONPATH=$PWD/PaddleNLP/:$PYTHONPATH
wget -q https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-Centos-Gcc82-Cuda102-Cudnn81-Trt7234-Py38-Compile/latest/paddle_inference.tgz
tar -zxf paddle_inference.tgz
export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++
cd ${nlp_dir}/paddlenlp/ops
#python op
mkdir build_tr_so
cd build_tr_so/
cmake ..  -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python
make -j >${log_path}/transformer_python_FT >>${log_path}/transformer_python_FT 2>&1
print_info $? transformer_python_FT
cd ../
#C++ op
mkdir build_tr_cc
cd build_tr_cc/
cmake .. -DCMAKE_BUILD_TYPE=Release -DPADDLE_LIB=${nlp_dir}/paddle_inference -DDEMO=${nlp_dir}/paddlenlp/ops/fast_transformer/src/demo/transformer_e2e.cc -DON_INFER=ON -DWITH_MKL=ON -DWITH_ONNXRUNTIME=ON
make -j >${log_path}/transformer_C_FT >>${log_path}/transformer_C_FT 2>&1
print_info $? transformer_C_FT
#deploy python
cd ${nlp_dir}/examples/machine_translation/transformer/fast_transformer/
sed -i "s#./trained_models/step_final/#./base_trained_models/step_final/#g" ../configs/transformer.base.yaml
wget -q https://paddlenlp.bj.bcebos.com/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
tar -zxf transformer-base-wmt_ende_bpe.tar.gz
export FLAGS_fraction_of_gpu_memory_to_use=0.1
cp -rf ${nlp_dir}/paddlenlp/ops/build_tr_so/third-party/build/fastertransformer/bin/decoding_gemm ./
./decoding_gemm 8 4 8 64 38512 32 512 0
#beam_search
python encoder_decoding_predict.py \
    --config ../configs/transformer.base.yaml \
    --decoding_lib ${nlp_dir}/paddlenlp/ops/build_tr_so/lib/libdecoding_op.so \
    --decoding_strategy beam_search \
    --beam_size 5 >${log_path}/transformer_deploy_P_FT >>${log_path}/transformer_deploy_P_FT 2>&1
print_info $? transformer_deploy_P_FT
#topk
python encoder_decoding_predict.py \
    --config ../configs/transformer.base.yaml \
    --decoding_lib ${nlp_dir}/paddlenlp/ops/build_tr_so/lib/libdecoding_op.so \
    --decoding_strategy topk_sampling \
    --topk 3 >topk.log
#topp
python encoder_decoding_predict.py \
    --config ../configs/transformer.base.yaml \
    --decoding_lib ${nlp_dir}/paddlenlp/ops/build_tr_so/lib/libdecoding_op.so \
    --decoding_strategy topp_sampling \
    --topk 0 \
    --topp 0.1 >topp.log
#deploy c++
python export_model.py  \
    --config ../configs/transformer.base.yaml  \
    --decoding_lib ${nlp_dir}/paddlenlp/ops/build_tr_so/lib/libdecoding_op.so   \
    --decoding_strategy beam_search --beam_size 5
./decoding_gemm 8 5 8 64 38512 256 512 0
${nlp_dir}/paddlenlp/ops/build_tr_cc/bin/./transformer_e2e -batch_size 8 -gpu_id 0 -model_dir ./infer_model/ -vocab_file ${PPNLP_HOME}/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 \
-data_file ${PPNLP_HOME}/datasets/WMT14ende/WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en  >${log_path}/transformer_deploy_C_FT >>${log_path}/transformer_deploy_C_FT 2>&1
print_info $? transformer_deploy_C_FT
}
# 22 transformer
transformer (){
cd ${nlp_dir}/examples/machine_translation/transformer/
wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/WMT14.en-de.partial.tar.gz
tar -xzvf WMT14.en-de.partial.tar.gz
time (
sed -i "s/save_step: 10000/save_step: 1/g" configs/transformer.base.yaml
sed -i "s/print_step: 100/print_step: 1/g" configs/transformer.base.yaml
sed -i "s/epoch: 30/epoch: 1/g" configs/transformer.base.yaml
sed -i "s/max_iter: None/max_iter: 2/g" configs/transformer.base.yaml
sed -i "s/batch_size: 4096/batch_size: 1000/g" configs/transformer.base.yaml

python train.py --config ./configs/transformer.base.yaml \
    --train_file ${PWD}/WMT14.en-de.partial/train.tok.clean.bpe.en ${PWD}/WMT14.en-de.partial/train.tok.clean.bpe.de \
    --dev_file ${PWD}/WMT14.en-de.partial/dev.tok.bpe.en ${PWD}/WMT14.en-de.partial/dev.tok.bpe.de \
    --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
    --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>"  >${log_path}/transformer_train) >>${log_path}/transformer_train 2>&1
print_info $? transformer_train
#predict
time (
sed -i 's#init_from_params: "./trained_models/step/"#init_from_params: "./trained_models/step_final/"#g' configs/transformer.base.yaml
python predict.py --config ./configs/transformer.base.yaml  \
    --test_file ${PWD}/WMT14.en-de.partial/test.tok.bpe.en ${PWD}/WMT14.en-de.partial/test.tok.bpe.de \
    --without_ft \
    --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
    --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>"  >${log_path}/transformer_predict) >>${log_path}/transformer_predict 2>&1
print_info $? transformer_predict
#export
time (
python export_model.py --config ./configs/transformer.base.yaml \
    --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
    --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_export) >>${log_path}/transformer_export 2>&1
print_info $? transformer_export
#infer
time (
python ./deploy/python/inference.py --config ./configs/transformer.base.yaml \
    --profile \
    --test_file ${PWD}/WMT14.en-de.partial/test.tok.bpe.en ${PWD}/WMT14.en-de.partial/test.tok.bpe.de  \
    --vocab_file ${PWD}/WMT14.en-de.partial/vocab_all.bpe.33708 \
    --unk_token "<unk>" --bos_token "<s>" --eos_token "<e>" >${log_path}/transformer_infer) >>${log_path}/transformer_infer 2>&1
print_info $? transformer_infer

fast_transformer
}
# 23 pet
pet (){
path="examples/few_shot/pet"
python scripts/regression/ci_normal_case.py ${path}
}
efl(){
path="examples/few_shot/efl"
python scripts/regression/ci_normal_case.py ${path}
}
p-tuning(){
path="examples/few_shot/p-tuning"
python scripts/regression/ci_normal_case.py ${path}
}
#24 simbert
simbert(){
cd ${nlp_dir}/examples/text_matching/simbert/
cp -r /ssd1/paddlenlp/download/simbert/dev.tsv ./
time (
python predict.py --input_file ./dev.tsv >${log_path}/simbert) >>${log_path}/simbert 2>&1
print_info $? simbert
}
#25 ernie-doc
ernie-doc(){
cd ${nlp_dir}/model_zoo/ernie-doc/
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  --log_dir hyp run_classifier.py --epochs 15 --layerwise_decay 0.7 --learning_rate 5e-5 --batch_size 4 --save_steps 100 --max_steps 100  --dataset hyp --output_dir hyp >${log_path}/ernie-doc_hyp) >>${log_path}/ernie-doc_hyp 2>&1
print_info $? ernie-doc_hyp
time (python -m paddle.distributed.launch  --log_dir cmrc2018 run_mrc.py --batch_size 4 --layerwise_decay 0.8 --dropout 0.2 --learning_rate 4.375e-5 --epochs 1 --save_steps 100 --max_steps 100  --dataset cmrc2018 --output_dir cmrc2018  >${log_path}/ernie-doc_cmrc2018) >>${log_path}/ernie-doc_cmrc2018 2>&1
print_info $?  ernie-doc_cmrc2018
time (python -m paddle.distributed.launch  --log_dir c3 run_mcq.py --learning_rate 6.5e-5 --epochs 1 --save_steps 100 --max_steps 100  --output_dir c3 >${log_path}/ernie-doc_c3) >>${log_path}/ernie-doc_c3 2>&1
print_info $? ernie-doc_c3
time (python -m paddle.distributed.launch  --log_dir cail/ run_semantic_matching.py --epochs 1 --layerwise_decay 0.8 --learning_rate 1.25e-5 --batch_size 4  --save_steps 100 --max_steps 100 --output_dir cail >${log_path}/ernie-doc_cail) >>${log_path}/ernie-doc_cail 2>&1
print_info $? ernie-doc_cail
time (python -m paddle.distributed.launch  --log_dir msra run_sequence_labeling.py --learning_rate 3e-5 --epochs 1 --save_steps 100 --max_steps 100  --output_dir msra  >${log_path}/ernie-doc_msar) >>${log_path}/ernie-doc_msar 2>&1
print_info $? ernie-doc_msar
time (python run_mrc.py  --model_name_or_path ernie-doc-base-zh  --dataset dureader_robust  --batch_size 8 --learning_rate 2.75e-4 --epochs 1 --save_steps 10 --max_steps 2 --logging_steps 10 --device gpu >${log_path}/ernie-doc_dureader_robust) >>${log_path}/ernie-doc_dureader_robust 2>&1
print_info $? ernie-doc_dureader_robust
}
#26 transformer-xl
transformer-xl (){
cd ${nlp_dir}/examples/language_model/transformer-xl/
mkdir gen_data && cd gen_data
wget https://paddle-qa.bj.bcebos.com/paddlenlp/enwik8.tar.gz && tar -zxvf enwik8.tar.gz
cd ../
export CUDA_VISIBLE_DEVICES=${cudaid2}
time (sed -i 's/print_step: 100/print_step: 1/g' configs/enwik8.yaml
sed -i 's/save_step: 10000/save_step: 3/g' configs/enwik8.yaml
sed -i 's/batch_size: 16/batch_size: 8/g' configs/enwik8.yaml
sed -i 's/max_step: 400000/max_step: 3/g' configs/enwik8.yaml
python -m paddle.distributed.launch  train.py --config ./configs/enwik8.yaml >${log_path}/transformer-xl_train_enwik8) >>${log_path}/transformer-xl_train_enwik8 2>&1
print_info $? transformer-xl_train_enwik8
time (sed -i 's/batch_size: 8/batch_size: 1/g' configs/enwik8.yaml
sed -i 's#init_from_params: "./trained_models/step_final/"#init_from_params: "./trained_models/step_3/"#g' configs/enwik8.yaml
python eval.py --config ./configs/enwik8.yaml >${log_path}/transformer-xl_eval_enwik8) >>${log_path}/transformer-xl_eval_enwik8 2>&1
print_info $? transformer-xl_eval_enwik8
}
#27 pointer_summarizer
pointer_summarizer() {
cd ${nlp_dir}/examples/text_summarization/pointer_summarizer/
cp -r /ssd1/paddlenlp/download/pointer_summarizer/* ./
export CUDA_VISIBLE_DEVICES=${cudaid1}
time (sed -i 's/max_iterations = 100000/max_iterations = 5/g' config.py
sed -i 's/if iter % 5000 == 0 or iter == 1000:/if iter % 5 == 0 :/g' train.py
python train.py >${log_path}/pointer_summarizer_train) >>${log_path}/pointer_summarizer_train 2>&1
print_info $? pointer_summarizer_train
}
#28 question_matching
question_matching() {
cd ${nlp_dir}/examples/text_matching/question_matching/
wget -q https://paddle-qa.bj.bcebos.com/paddlenlp/data_v4.tar.gz
tar -xvzf data_v4.tar.gz
export CUDA_VISIBLE_DEVICES=${cudaid2}
#train
time (
python -u -m paddle.distributed.launch train.py \
       --train_set ./data_v4/train/ALL/train \
       --dev_set ./data_v4/train/ALL/dev \
       --device gpu \
       --eval_step 10 \
       --max_steps 10 \
       --save_dir ./checkpoints \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --epochs 1 \
       --rdrop_coef 0.0 >${log_path}/question_matching_train) >>${log_path}/question_matching_train 2>&1
print_info $? question_matching_train
#predict
time (
export CUDA_VISIBLE_DEVICES=${cudaid1}
python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_10/model_state.pdparams" \
    --batch_size 128 \
    --input_file ./data_v4/test/public_test_A \
    --result_file 0.0_predict_public_result_test_A_re >${log_path}/question_matching_predict) >>${log_path}/question_matching_predict 2>&1
print_info $? question_matching_predict
}
# 29 ernie-csc
ernie-csc() {
export CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/text_correction/ernie-csc
#dowdnload data
python download.py --data_dir ./extra_train_ds/ --url https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml
#trans xml txt
python change_sgml_to_txt.py -i extra_train_ds/train.sgml -o extra_train_ds/train.txt
#2卡训练
python -m paddle.distributed.launch  train.py --batch_size 32 --logging_steps 100 --epochs 1 --learning_rate 5e-5 --model_name_or_path ernie-1.0-base-zh --output_dir ./checkpoints/ --extra_train_ds_dir ./extra_train_ds/  >${log_path}/ernie-csc_train >>${log_path}/ernie-csc_train 2>&1
print_info $? ernie-csc_train
#predict
sh run_sighan_predict.sh >${log_path}/ernie-csc_predict >>${log_path}/ernie-csc_predict 2>&1
print_info $? ernie-csc_predict
#export model
python export_model.py --params_path ./checkpoints/best_model.pdparams --output_path ./infer_model/static_graph_params >${log_path}/ernie-csc_export >>${log_path}/ernie-csc_export 2>&1
print_info $? ernie-csc_export
#python deploy
python predict.py --model_file infer_model/static_graph_params.pdmodel --params_file infer_model/static_graph_params.pdiparams >${log_path}/ernie-csc_deploy >>${log_path}/ernie-csc_deploy 2>&1
print_info $? ernie-csc_deploy
}
#30 nptag
nptag() {
cd ${nlp_dir}/examples/text_to_knowledge/nptag/
wget -q https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/nptag_dataset.tar.gz && tar -zxvf nptag_dataset.tar.gz
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch  train.py \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 100 \
    --output_dir ./output \
    --device "gpu" >${log_path}/nptag_train >>${log_path}/nptag_train 2>&1
print_info $? nptag_train
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch  predict.py \
    --device=gpu \
    --params_path ./output/model_100/model_state.pdparams >${log_path}/nptag_predict >>${log_path}/nptag_predict 2>&1
print_info $? nptag_predict
python export_model.py --params_path=./output/model_100/model_state.pdparams --output_path=./export >${log_path}/nptag_export >>${log_path}/nptag_export 2>&1
print_info $? nptag_export
python deploy/python/predict.py --model_dir=./export >${log_path}/nptag_depoly >>${log_path}/nptag_deploy 2>&1
print_info $? nptag_depoly
}
#31 ernie-m
ernie-m() {
export CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/model_zoo/ernie-m
# TODO(ouyanghongyu): remove the following scripts later.
if [ ! -f 'test.py' ];then
    echo '模型测试文件不存在！'
    # finetuned for cross-lingual-transfer
    python -m paddle.distributed.launch --log_dir output_clt run_classifier.py \
        --do_train \
        --do_eval \
        --do_export \
        --device gpu \
        --task_type cross-lingual-transfer \
        --model_name_or_path __internal_testing__/ernie-m \
        --use_test_data True \
        --test_data_path ../../tests/fixtures/tests_samples/xnli/xnli.jsonl \
        --output_dir output_clt \
        --export_model_dir output_clt \
        --per_device_train_batch_size 8 \
        --save_steps 1 \
        --eval_steps 1  \
        --max_steps 2 \
        --overwrite_output_dir \
        --remove_unused_columns False >${log_path}/ernie-m_clt >>${log_path}/ernie-m_clt 2>&1
    print_info $? ernie-m_clt
    # finetuned for translate-train-all
    python -m paddle.distributed.launch --log_dir output_tta run_classifier.py \
        --do_train \
        --do_eval \
        --do_export \
        --device gpu \
        --task_type translate-train-all \
        --model_name_or_path __internal_testing__/ernie-m \
        --use_test_data True \
        --test_data_path ../../tests/fixtures/tests_samples/xnli/xnli.jsonl \
        --output_dir output_tta \
        --export_model_dir output_tta \
        --per_device_train_batch_size 8 \
        --save_steps 1 \
        --eval_steps 1  \
        --max_steps 2 \
        --overwrite_output_dir \
        --remove_unused_columns False >${log_path}/ernie-m_tta >>${log_path}/ernie-m_tta 2>&1
    print_info $? ernie-m_tta
else
    python -m pytest ${nlp_dir}/tests/model_zoo/test_ernie_m.py >${log_path}/ernie-m >>${log_path}/ernie-m 2>&1
    print_info $? ernie-m
fi
}
#32 clue
clue (){
cd ${nlp_dir}/examples/benchmark/clue/classification
python -u ./run_clue_classifier_trainer.py \
    --model_name_or_path ernie-3.0-base-zh \
    --dataset "clue afqmc" \
    --max_seq_length 128 \
    --per_device_train_batch_size 32   \
    --per_device_eval_batch_size 32   \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --seed 42  \
    --save_steps 3 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ./tmp \
    --device gpu  \
    --do_train \
    --do_eval \
    --metric_for_best_model "eval_accuracy" \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --max_steps 1 >${log_path}/clue-trainer_api >>${log_path}/clue-trainer_api 2>&1
print_info $? clue-tranier_api
python -u run_clue_classifier.py  \
    --model_name_or_path ernie-3.0-base-zh \
    --task_name afqmc \
    --max_seq_length 128 \
    --batch_size 16   \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --seed 42  \
    --save_steps 1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --output_dir ./output/afqmc \
    --device gpu \
    --max_steps 1 \
    --do_train  >${log_path}/clue-class >>${log_path}/clue-class 2>&1
print_info $? clue-class
cd ${nlp_dir}/examples/benchmark/clue/mrc
export CUDA_VISIBLE_DEVICES=${cudaid1}
python -m paddle.distributed.launch run_cmrc2018.py \
    --model_name_or_path ernie-3.0-base-zh \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --num_train_epochs 2 \
    --do_train \
    --do_predict \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --gradient_accumulation_steps 2 \
    --max_steps 1 \
    --output_dir ./tmp >${log_path}/clue-mrc >>${log_path}/clue-mrc 2>&1
print_info $? clue-mrc
}
#32 textcnn
textcnn(){
cd ${nlp_dir}/examples/sentiment_analysis/textcnn
wget https://bj.bcebos.com/paddlenlp/datasets/RobotChat.tar.gz
tar xvf RobotChat.tar.gz
wget https://bj.bcebos.com/paddlenlp/robot_chat_word_dict.txt
wget https://bj.bcebos.com/paddlenlp/models/textcnn.pdparams
python -m paddle.distributed.launch train.py \
    --vocab_path=./robot_chat_word_dict.txt \
    --init_from_ckpt=./textcnn.pdparams \
    --device=gpu \
    --lr=5e-5 \
    --batch_size=64 \
    --epochs=1 \
    --save_dir=./checkpoints \
    --data_path=./RobotChat >${log_path}/textcnn_train >>${log_path}/textcnn_train 2>&1
print_info $? textcnn_train
python export_model.py --vocab_path=./robot_chat_word_dict.txt --params_path=./checkpoints/final.pdparams \
    --output_path=./static_graph_params >${log_path}/textcnn_export >>${log_path}/textcnn_export 2>&1
print_info $? export_export
python deploy/python/predict.py --model_file=static_graph_params.pdmodel \
    --params_file=static_graph_params.pdiparams >${log_path}/textcnn_depoly >>${log_path}/textcnn_depoly 2>&1
print_info $? textcnn_deploy
python predict.py --vocab_path=./robot_chat_word_dict.txt \
    --device=gpu \
    --params_path=./checkpoints/final.pdparams >${log_path}/textcnn_predict >>${log_path}/textcnn_predict 2>&1
print_info $? textcnn_predict
}
#33 taskflow
taskflow (){
cd ${nlp_dir}
python -m pytest tests/taskflow/test_*.py >${nlp_dir}/unittest_logs/taskflow_unittest >>${nlp_dir}/unittest_logs/taskflow_unittest 2>&1
print_info $? taskflow_unittest
python -m pytest scripts/regression/test_taskflow.py >${log_path}/taskflow >>${log_path}/taskflow 2>&1
print_info $? taskflow
}
transformers(){
echo ' RUN all transformers unittest'
cd ${nlp_dir}/tests/transformers/
for apicase in `ls`;do
    if [[ ${apicase##*.} == "py" ]];then
            continue
    else
        cd ${nlp_dir}
        python -m pytest tests/transformers/${apicase}/test_*.py  >${nlp_dir}/unittest_logs/${apicase}_unittest.log 2>&1
        print_info $? tests ${apicase}_unittest
    fi
done
}
fast_generation(){

export CC=/usr/local/gcc-8.2/bin/gcc
export CXX=/usr/local/gcc-8.2/bin/g++

cd ${nlp_dir}/fast_generation/samples
python codegen_sample.py >${log_path}/fast_generation_codegen >>${log_path}/fast_generation_codegen 2>&1
print_info $? fast_generation_codegen

python gpt_sample.py >${log_path}/fast_generation_gpt >>${log_path}/fast_generation_gpt 2>&1
print_info $? fast_generation_gpt

python mbart_sample.py >${log_path}/fast_generation_mbart >>${log_path}/fast_generation_mbart 2>&1
print_info $? fast_generation_mbart

python plato_sample.py >${log_path}/fast_generation_plato >>${log_path}/fast_generation_plato 2>&1
print_info $? fast_generation_plato

python t5_sample.py --use_faster >${log_path}/fast_generation_t5 >>${log_path}/fast_generation_t5 2>&1
print_info $? fast_generation_t5

cd ${nlp_dir}/paddlenlp/ops/fast_transformer/sample/
python bart_decoding_sample.py >${log_path}/fast_generation_bart >>${log_path}/fast_generation_bart 2>&1
print_info $? fast_generation_bart

python t5_export_model_sample.py >${log_path}/t5_export_model_sample >>${log_path}/t5_export_model_sample 2>&1
print_info $? t5_export_model_sample

python t5_export_model_sample.py >${log_path}/t5_export_model_sample >>${log_path}/t5_export_model_sample 2>&1
print_info $? t5_export_model_sample

fast_gpt
fast_transformer
}
ernie-3.0(){
cd ${nlp_dir}/model_zoo/ernie-3.0/
#训练
python run_seq_cls.py  --model_name_or_path ernie-3.0-medium-zh  --dataset afqmc --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_seq_cls >>${log_path}/ernie-3.0_train_seq_cls 2>&1
print_info $? ernie-3.0_train_seq_cls
python run_token_cls.py --model_name_or_path ernie-3.0-medium-zh --dataset msra_ner --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_token_cls >>${log_path}/ernie-3.0_train_token_cls 2>&1
print_info $? ernie-3.0_train_token_cls
python run_qa.py --model_name_or_path ernie-3.0-medium-zh --dataset cmrc2018  --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=configs/default.yml --max_steps=2 --save_step=2 >${log_path}/ernie-3.0_train_qa >>${log_path}/ernie-3.0_train_qa 2>&1
print_info $? ernie-3.0_train_qa
# 预测
python run_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_seq_cls >>${log_path}/ernie-3.0_predict_seq_cls 2>&1
print_info $? ernie-3.0_predict_seq_cls
python run_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_token_cls >>${log_path}/ernie-3.0_predict_token_cls 2>&1
print_info $? ernie-3.0_predict_token_cls
python run_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models --do_predict --config=configs/default.yml >${log_path}/ernie-3.0_predict_qa >>${log_path}/ernie-3.0_predict_qa 2>&1
print_info $? ernie-3.0_predict_qa
#压缩
python compress_seq_cls.py  --model_name_or_path best_models/afqmc/  --dataset afqmc --output_dir ./best_models/afqmc --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5 --save_steps 5 --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_seq_cls >>${log_path}/ernie-3.0_compress_seq_cls 2>&1
print_info $? ernie-3.0_compress_seq_cls
python compress_token_cls.py  --model_name_or_path best_models/msra_ner/  --dataset msra_ner --output_dir ./best_models/msra_ner --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5  --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_token_cls >>${log_path}/ernie-3.0_compress_token_cls 2>&1
print_info $? ernie-3.0_compress_token_cls
python compress_qa.py --model_name_or_path best_models/cmrc2018/ --dataset cmrc2018  --output_dir ./best_models/cmrc2018 --config=configs/default.yml --max_steps 10 --eval_steps 5 --save_steps 5  --algo_list mse --batch_size_list 4 >${log_path}/ernie-3.0_compress_qa >>${log_path}/ernie-3.0_compress_qa 2>&1
print_info $? ernie-3.0_compress_qa
}
ernie-health(){
cd ${nlp_dir}/tests/model_zoo/
if [ ! -f 'test_ernie-health.py' ];then
    echo '模型测试文件不存在！'
else
    python -m pytest tests/model_zoo/test_ernie-health.py >${log_path}/ernie-health_unittest>>${log_path}/ernie-health_unittest 2>&1
    print_info $? tests ernie-health_unittest
fi
}
uie(){
cd ${nlp_dir}/model_zoo/uie/
mkdir data && cd data && wget https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json && cd ../
python doccano.py --doccano_file ./data/doccano_ext.json --task_type ext --save_dir ./data --splits 0.8 0.2 0 --schema_lang ch >${log_path}/uie_doccano>>${log_path}/uie_doccano 2>&1
print_info $? uie_doccano
python -u -m paddle.distributed.launch finetune.py --device gpu --logging_steps 2 --save_steps 2 --eval_steps 2 --seed 42 \
    --model_name_or_path uie-base --output_dir ./checkpoint/model_best --train_path data/train.txt --dev_path data/dev.txt \
    --max_seq_length 512 --per_device_eval_batch_size 16 --per_device_train_batch_size 16 --num_train_epochs 100 --learning_rate 1e-5 \
    --do_train --do_eval --do_export --export_model_dir ./checkpoint/model_best --label_names start_positions end_positions \
    --overwrite_output_dir --disable_tqdm True --metric_for_best_model eval_f1 --load_best_model_at_end True \
    --save_total_limit 1 --max_steps 2  >${log_path}/uie_train>>${log_path}/uie_train2>&1
print_info $? uie_train
python evaluate.py --model_path ./checkpoint/model_best --test_path ./data/dev.txt --batch_size 16 --max_seq_len 512 >${log_path}/uie_eval>>${log_path}/uie_eval 2>&1
print_info $? uie_eval
}
ernie-layout(){
cd ${nlp_dir}/model_zoo/ernie-layout/
# train ner
python -u run_ner.py --model_name_or_path ernie-layoutx-base-uncased --output_dir ./ernie-layoutx-base-uncased/models/funsd/ \
    --dataset_name funsd --do_train --do_eval --max_steps 2 --eval_steps 2 --save_steps 2 --save_total_limit 1 --seed 1000 --overwrite_output_dir \
    --load_best_model_at_end --pattern ner-bio --preprocessing_num_workers 4 --overwrite_cache false --doc_stride 128 --target_size 1000 \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --learning_rate 2e-5 --lr_scheduler_type constant --gradient_accumulation_steps 1 \
    --metric_for_best_model eval_f1 --greater_is_better true >${log_path}/ernie-layout_train>>${log_path}/ernie-layout_train 2>&1
print_info $? ernie-layout_train
# export ner
python export_model.py --task_type ner --model_path ./ernie-layoutx-base-uncased/models/funsd/ --output_path ./ner_export >${log_path}/ernie-layout_export>>${log_path}/ernie-layout_export2>&1
print_info $? ernie-layout_export
# deploy ner
cd ${nlp_dir}/model_zoo/ernie-layout/deploy/python
wget https://bj.bcebos.com/paddlenlp/datasets/document_intelligence/images.zip && unzip images.zip
python infer.py --model_path_prefix ../../ner_export/inference --task_type ner --lang "en" --batch_size 8 >${log_path}/ernie-layout_deploy>>${log_path}/ernie-layout_deploy 2>&1
print_info $? ernie-layout_deploy
}
ernie-1.0(){
    ernie
}

ernie-3.0(){
    ernie
}

ernie_m(){
    ernie-m
}

ernie_layout(){
ernie-layout
}

ernie_csc(){
    ernie-csc
}

ernie_ctm(){
    ernie-ctm
}

ernie_doc(){
    ernie-doc
}

ernie_health(){
    ernie-health
}

gpt-3() {
    bash ${nlp_dir}/scripts/regression/ci_gpt-3.sh
    print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'`
}
$1
