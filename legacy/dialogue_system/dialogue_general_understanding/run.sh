#!/bin/bash

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1

export CUDA_VISIBLE_DEVICES=
if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    export CPU_NUM=1
    use_cuda=false
else
    use_cuda=true
fi

TASK_NAME=$1
TASK_TYPE=$2

BERT_BASE_PATH="./data/pretrain_model/uncased_L-12_H-768_A-12"
INPUT_PATH="./data/input/data/${TASK_NAME}"
SAVE_MODEL_PATH="./data/saved_models/${TASK_NAME}"
TRAIN_MODEL_PATH="./data/saved_models/trained_models"
OUTPUT_PATH="./data/output"
INFERENCE_MODEL="data/inference_models"
PYTHON_PATH="python"

if [ -f ${SAVE_MODEL_PATH} ]; then
    rm ${SAVE_MODEL_PATH}
fi

if [ ! -d ${SAVE_MODEL_PATH} ]; then
	mkdir ${SAVE_MODEL_PATH}
fi

#parameter configuration
if [ "${TASK_NAME}" = "udc" ]
then
  save_steps=1000
  max_seq_len=210
  print_steps=1000
  batch_size=32
  epoch=2
  learning_rate=2e-5
elif [ "${TASK_NAME}" = "swda" ]
then
  save_steps=500
  max_seq_len=128
  print_steps=200
  batch_size=32
  epoch=3
  learning_rate=2e-5
elif [ "${TASK_NAME}" = "mrda" ]
then
  save_steps=500
  max_seq_len=128
  print_steps=200
  batch_size=32
  epoch=7
  learning_rate=2e-5
elif [ "${TASK_NAME}" = "atis_intent" ]
then
  save_steps=100
  max_seq_len=128
  print_steps=10
  batch_size=32
  epoch=20
  learning_rate=2e-5
  INPUT_PATH="./data/input/data/atis/${TASK_NAME}"
elif [ "${TASK_NAME}" = "atis_slot" ]
then
  save_steps=100
  max_seq_len=128
  print_steps=10
  batch_size=32
  epoch=50
  learning_rate=2e-5
  INPUT_PATH="./data/input/data/atis/${TASK_NAME}"
elif [ "${TASK_NAME}" = "dstc2" ]
then
  save_steps=400
  print_steps=20
  epoch=40
  learning_rate=5e-5
  INPUT_PATH="./data/input/data/dstc2/${TASK_NAME}"
  if [ "${TASK_TYPE}" = "train" ]
  then
    max_seq_len=256
    batch_size=32
  else
    max_seq_len=512
    batch_size=16
  fi
else
  echo "not support ${TASK_NAME} dataset.."
  exit 255
fi

#training
function train()
{
    $PYTHON_PATH -u main.py \
       --task_name=${TASK_NAME} \
       --use_cuda=$1 \
       --do_train=true \
       --epoch=${epoch} \
       --batch_size=${batch_size} \
       --do_lower_case=true \
       --data_dir=${INPUT_PATH} \
       --bert_config_path=${BERT_BASE_PATH}/bert_config.json \
       --vocab_path=${BERT_BASE_PATH}/vocab.txt \
       --init_from_pretrain_model=${BERT_BASE_PATH}/params \
       --save_model_path=${SAVE_MODEL_PATH} \
       --save_steps=${save_steps} \
       --learning_rate=${learning_rate} \
       --weight_decay=0.01 \
       --max_seq_len=${max_seq_len} \
       --print_steps=${print_steps};
}

#predicting
function predict()
{
    $PYTHON_PATH -u main.py \
       --task_name=${TASK_NAME} \
       --use_cuda=$1 \
       --do_predict=true \
       --batch_size=${batch_size} \
       --data_dir=${INPUT_PATH} \
       --do_lower_case=true \
       --init_from_params=${TRAIN_MODEL_PATH}/${TASK_NAME}/params/params \
       --bert_config_path=${BERT_BASE_PATH}/bert_config.json \
       --vocab_path=${BERT_BASE_PATH}/vocab.txt \
       --output_prediction_file=${OUTPUT_PATH}/pred_${TASK_NAME} \
       --max_seq_len=${max_seq_len};
}

#evaluating
function evaluate()
{
    $PYTHON_PATH -u main.py \
       --task_name=${TASK_NAME} \
       --use_cuda=$1 \
       --do_eval=True \
       --evaluation_file=${INPUT_PATH}/test.txt \
       --output_prediction_file=${OUTPUT_PATH}/pred_${TASK_NAME};
}

#saving the inference model
function save_inference()
{
    $PYTHON_PATH -u main.py \
       --task_name=${TASK_NAME} \
       --use_cuda=$1 \
       --init_from_params=${TRAIN_MODEL_PATH}/${TASK_NAME}/params \
       --do_save_inference_model=True \
       --bert_config_path=${BERT_BASE_PATH}/bert_config.json \
       --inference_model_dir=${INFERENCE_MODEL}/${TASK_NAME};
}

if [ "${TASK_TYPE}" = "train" ]
then
    echo "train $TASK_NAME start..........";
    train $use_cuda;
    echo ""train $TASK_NAME finish..........
elif [ "${TASK_TYPE}" = "predict" ]
then 
    echo "predict $TASK_NAME start..........";
    predict $use_cuda;
    echo "predict $TASK_NAME finish..........";
elif [ "${TASK_TYPE}" = "evaluate" ]
then
    export CUDA_VISIBLE_DEVICES=
    echo "evaluate $TASK_NAME start.........."; 
    evaluate false;
    echo "evaluate $TASK_NAME finish..........";
elif [ "${TASK_TYPE}" = "inference" ]
then
    echo "save $TASK_NAME inference model start..........";
    save_inference $use_cuda;
    echo "save $TASK_NAME inference model finish..........";
elif [ "${TASK_TYPE}" = "all" ]
then
    echo "Execute train、predict、evaluate and save inference model in sequence...."
    train $use_cuda;
    predict $use_cuda;
    evaluate false;
    save_inference $use_cuda;
    echo "done";
else
    echo "Parameter $TASK_TYPE is not supported, you can input parameter in [train|predict|evaluate|inference|all]"
    exit 255;
fi
    

