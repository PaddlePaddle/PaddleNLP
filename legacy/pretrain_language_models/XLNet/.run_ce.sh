
train(){
python run_classifier.py \
    --data_dir data/STS-B \
    --verbose True \
    --shuffle false \
    --init_checkpoint xlnet_cased_L-12_H-768_A-12/params \
    --predict_dir exp/sts-b \
    --model_config_path xlnet_cased_L-12_H-768_A-12/xlnet_config.json \
    --uncased False \
    --save_steps 50 \
    --train_steps 50 \
    --epoch 1 \
    --skip_steps 10 \
    --validation_steps 30 \
    --task_name sts-b \
    --warmup_steps 5 \
    --random_seed 100 \
    --spiece_model_file xlnet_cased_L-12_H-768_A-12/spiece.model \
    --checkpoints checkpoints_sts-b \
    --is_regression True \
    --use_cuda True \
    --eval_batch_size 4 \
    --enable_ce
}

export CUDA_VISIBLE_DEVICES=0
train | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
