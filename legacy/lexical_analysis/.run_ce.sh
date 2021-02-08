#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1


train()
{
    python run_sequence_labeling.py \
        --do_train True \
        --do_test True \
        --do_infer False \
        --train_data ./data/train.tsv \
        --test_data ./data/test.tsv \
        --model_save_dir ./models \
        --valid_model_per_batches 1000 \
        --save_model_per_batches 10000 \
        --batch_size 100 \
        --epoch 2 \
        --use_cuda True \
        --traindata_shuffle_buffer 200000 \
        --word_emb_dim 768 \
        --grnn_hidden_dim 768 \
        --bigru_num 2 \
        --base_learning_rate 1e-3 \
        --emb_learning_rate 5 \
        --crf_learning_rate 0.2 \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic \
        --enable_ce
}

export CUDA_VISIBLE_DEVICES=0
train | python _ce.py
sleep 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
