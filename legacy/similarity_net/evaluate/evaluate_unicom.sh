#!/usr/bin/env bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=3
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='simnet'
VOCAB_PATH=./data/term2id.dict
CKPT_PATH=./model_files
INFER_RESULT_PATH=./evaluate/unicom_infer_result
TASK_MODE='pairwise'
CONFIG_PATH=./config/bow_pairwise.json
INIT_CHECKPOINT=./model_files/simnet_bow_pairwise_pretrained_model/

# use JiebaTokenizer to evaluate
TOKENIZER="JiebaTokenizer"
INFER_DATA_PATH=./data/unicom_infer_raw

# use tokenized data by WordSeg to evaluate
#TOKENIZER=""
#INFER_DATA_PATH=./evaluate/unicom_infer

python unicom_split.py
cd ..
python ./run_classifier.py \
    --task_name ${TASK_NAME} \
    --use_cuda false \
    --do_infer True \
    --batch_size 128 \
    --infer_data_dir ${INFER_DATA_PATH} \
    --infer_result_path ${INFER_RESULT_PATH} \
    --config_path ${CONFIG_PATH} \
    --vocab_path ${VOCAB_PATH} \
    --tokenizer ${TOKENIZER:-""} \
    --task_mode ${TASK_MODE} \
    --init_checkpoint ${INIT_CHECKPOINT}
cd evaluate
python unicom_compute_pos_neg.py
