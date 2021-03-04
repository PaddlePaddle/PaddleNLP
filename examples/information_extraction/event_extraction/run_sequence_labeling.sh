export CUDA_VISIBLE_DEVICES=0

data_dir=$1
conf_path=$2
vocab_path=$3
ckpt_dir=$4
predict_data=$5
learning_rate=$6
is_train=$7
max_seq_len=$8
batch_size=$9
epoch=${10}
pred_save_path=${11}

python sequence_labeling.py --num_epoch ${epoch} \
    --learning_rate ${learning_rate} \
    --tag_path ${conf_path} \
    --vocab_path ${vocab_path} \
    --train_data ${data_dir}/train.tsv \
    --dev_data ${data_dir}/dev.tsv \
    --test_data ${data_dir}/test.tsv \
    --predict_data ${predict_data} \
    --do_train ${is_train} \
    --do_predict True \
    --max_seq_len ${max_seq_len} \
    --batch_size ${batch_size} \
    --skip_step 10 \
    --valid_step 50 \
    --checkpoints ${ckpt_dir} \
    --init_ckpt ${ckpt_dir}/best.pdparams \
    --predict_save_path ${pred_save_path} \
    --n_gpu 1
