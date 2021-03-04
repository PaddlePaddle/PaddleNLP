#!/bin/bash

train_atis_slot(){ 
  if [ ! -d "./data/saved_models/atis_slot" ]; then
      mkdir "./data/saved_models/atis_slot"
  fi
  python -u train.py \
  --task_name=atis_slot \
  --use_cuda=true \
  --do_train=true \
  --in_tokens=false \
  --epoch=2 \
  --batch_size=32 \
  --data_dir=./data/input/data/atis/atis_slot \
  --bert_config_path=./data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json \
  --vocab_path=./data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt \
  --init_from_pretrain_model=./data/pretrain_model/uncased_L-12_H-768_A-12/params \
  --save_model_path=./data/saved_models/atis_slot \
  --save_param="params" \
  --save_steps=100 \
  --learning_rate=2e-5 \
  --weight_decay=0.01 \
  --max_seq_len=128 \
  --print_steps=10 \
  --use_fp16=false \
  --enable_ce=store_true 
}

train_mrda(){
  if [ ! -d "./data/saved_models/mrda" ]; then
      mkdir "./data/saved_models/mrda"
  fi
  python -u train.py \
  --task_name=mrda \
  --use_cuda=true \
  --do_train=true \
  --in_tokens=true \
  --epoch=2 \
  --batch_size=4096 \
  --data_dir=./data/input/data/mrda \
  --bert_config_path=./data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json \
  --vocab_path=./data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt \
  --init_from_pretrain_model=./data/pretrain_model/uncased_L-12_H-768_A-12/params \
  --save_model_path=./data/saved_models/mrda \
  --save_param="params" \
  --save_steps=500 \
  --learning_rate=2e-5 \
  --weight_decay=0.01 \
  --max_seq_len=128 \
  --print_steps=200 \
  --use_fp16=false \
  --enable_ce=store_true 
}

# FIXME(zjl): this model would fail when GC is enabled,
# but it seems that this error is from the model itself.
# See issue here: https://github.com/PaddlePaddle/Paddle/issues/18994#event-2532039900       
# To fix ce, disable gc in this model temporarily.  
export FLAGS_eager_delete_tensor_gb=1

cudaid=${multi:=0,1,2,3}
export CUDA_VISIBLE_DEVICES=$cudaid
train_atis_slot | python _ce.py
sleep 20

cudaid=${single:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
train_atis_slot | python _ce.py
