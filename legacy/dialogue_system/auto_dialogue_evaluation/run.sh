#!/bin/bash

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1.0

export CUDA_VISIBLE_DEVICES=0

if [ $# -ne 2 ]
then
  echo "please input parameters: TRAIN_TYPE and TASK_TYPE"
  echo "TRAIN_TYPE: [matching|seq2seq_naive|seq2seq_att|keywords|human]"
  echo "TASK_TYPE: [train|predict|evaluate|inference]"
  exit 255
fi

TRAIN_TYPE=$1
TASK_TYPE=$2

candi_train_type=("matching" "seq2seq_naive" "seq2seq_att" "keywords" "human")
candi_task_type=("train" "predict" "evaluate" "inference")

if [[ ! "${candi_train_type[@]}" =~ ${TRAIN_TYPE} ]] 
then
  echo "unknown parameter: ${TRAIN_TYPE}, just support [matching|seq2seq_naive|seq2seq_att|keywords|human]"
  exit 255
fi

if [[ ! "${candi_task_type[@]}" =~ ${TASK_TYPE} ]] 
then
  echo "unknown parameter: ${TRAIN_TYPE}, just support [train|predict|evaluate|inference]"
  exit 255
fi

INPUT_PATH="data/input/data"
OUTPUT_PATH="data/output"
SAVED_MODELS="data/saved_models"
INFERENCE_MODEL="data/inference_models"
PYTHON_PATH="python"

#train pretrain model
if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
  export CPU_NUM=1
  use_cuda=false
else
  use_cuda=true
fi

#training
function pretrain_train()
{

    pretrain_model_path="${SAVED_MODELS}/matching_pretrained"
    if [ -f ${pretrain_model_path} ]
    then
        rm ${pretrain_model_path}
    fi

    if [ ! -d ${pretrain_model_path} ]
    then
        mkdir ${pretrain_model_path}
    fi

    ${PYTHON_PATH} -u main.py \
      --do_train=true \
      --use_cuda=${1} \
      --loss_type="CLS" \
      --max_seq_len=50 \
      --save_model_path=${pretrain_model_path} \
      --training_file="${INPUT_PATH}/unlabel_data/train.ids" \
      --epoch=20 \
      --print_step=1 \
      --save_step=400 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016 \
      --learning_rate=0.001 \
      --sample_pro=0.1 
}

function finetuning_train()
{
    save_model_path="${SAVED_MODELS}/${2}_finetuned"

    if [ -f ${save_model_path} ]
    then
        rm ${save_model_path}
    fi

    if [ ! -d ${save_model_path} ]
    then
        mkdir ${save_model_path}
    fi

    ${PYTHON_PATH} -u main.py \
      --do_train=true \
      --use_cuda=${1} \
      --loss_type="L2" \
      --max_seq_len=50 \
      --init_from_pretrain_model="${SAVED_MODELS}/matching_pretrained/step_final" \
      --save_model_path=${save_model_path} \
      --training_file="${INPUT_PATH}/label_data/${2}/train.ids" \
      --epoch=50 \
      --print_step=1 \
      --save_step=400 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016 \
      --learning_rate=0.001 \
      --sample_pro=0.1
}

#predict
function pretrain_predict()
{
    ${PYTHON_PATH} -u main.py \
      --do_predict=true \
      --use_cuda=${1} \
      --predict_file="${INPUT_PATH}/unlabel_data/test.ids" \
      --init_from_params="${SAVED_MODELS}/trained_models/matching_pretrained/params/params" \
      --loss_type="CLS" \
      --output_prediction_file="${OUTPUT_PATH}/pretrain_matching_predict" \
      --max_seq_len=50 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016
}

function finetuning_predict()
{
    ${PYTHON_PATH} -u main.py \
      --do_predict=true \
      --use_cuda=${1} \
      --predict_file="${INPUT_PATH}/label_data/${2}/test.ids" \
      --init_from_params="${SAVED_MODELS}/trained_models/${2}_finetuned/params/params" \
      --loss_type="L2" \
      --output_prediction_file="${OUTPUT_PATH}/finetuning_${2}_predict" \
      --max_seq_len=50 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016
}

#evaluate
function pretrain_eval()
{
    ${PYTHON_PATH} -u main.py \
      --do_eval=true \
      --use_cuda=${1} \
      --evaluation_file="${INPUT_PATH}/unlabel_data/test.ids" \
      --output_prediction_file="${OUTPUT_PATH}/pretrain_matching_predict" \
      --loss_type="CLS" 
}

function finetuning_eval()
{
    ${PYTHON_PATH} -u main.py \
      --do_eval=true \
      --use_cuda=${1} \
      --evaluation_file="${INPUT_PATH}/label_data/${2}/test.ids" \
      --output_prediction_file="${OUTPUT_PATH}/finetuning_${2}_predict" \
      --loss_type="L2" 
}

#inference model
function pretrain_infer()
{
    ${PYTHON_PATH} -u main.py \
      --do_save_inference_model=true \
      --use_cuda=${1} \
      --init_from_params="${SAVED_MODELS}/trained_models/matching_pretrained/params" \
      --inference_model_dir="${INFERENCE_MODEL}/matching_inference_model"

}
function finetuning_infer()
{
    ${PYTHON_PATH} -u main.py \
      --do_save_inference_model=true \
      --use_cuda=${1} \
      --init_from_params="${SAVED_MODELS}/trained_models/${2}_finetuned/params" \
      --inference_model_dir="${INFERENCE_MODEL}/${2}_inference_model"
}

if [ "${TASK_TYPE}" = "train" ]
then
    echo "train ${TRAIN_TYPE} start.........."
    if [ "${TRAIN_TYPE}" = "matching" ]
    then
        pretrain_train ${use_cuda};
    else
        finetuning_train ${use_cuda} ${TRAIN_TYPE};
    fi
elif [ "${TASK_TYPE}" = "predict" ]
then
    echo "predict ${TRAIN_TYPE} start.........."
    if [ "${TRAIN_TYPE}" = "matching" ]
    then
        pretrain_predict ${use_cuda};
    else
        finetuning_predict ${use_cuda} ${TRAIN_TYPE};
    fi
elif [ "${TASK_TYPE}" = "evaluate" ]
then
    echo "evaluate ${TRAIN_TYPE} start.........."
    if [ "${TRAIN_TYPE}" = "matching" ]
    then
        pretrain_eval ${use_cuda};
    else
        finetuning_eval ${use_cuda} ${TRAIN_TYPE};
    fi
elif [ "${TASK_TYPE}" = "inference" ]
then
    echo "save ${TRAIN_TYPE} inference model start.........."
    if [ "${TRAIN_TYPE}" = "matching" ]
    then
        pretrain_infer ${use_cuda};
    else
        finetuning_infer ${use_cuda} ${TRAIN_TYPE};
    fi
else
    exit 255
fi

