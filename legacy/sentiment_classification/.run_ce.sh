#! /bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CPU_NUM=1


# run_train on train.tsv and do_val on test.tsv
train() {
    python -u run_classifier.py \
        --task_name 'senta' \
        --use_cuda true \
        --do_train true \
        --do_val true \
        --do_infer false \
        --batch_size 16 \
        --data_dir ./senta_data/ \
        --vocab_path ./senta_data/word_dict.txt \
        --checkpoints ./save_models \
        --save_steps 500 \
        --validation_steps 50 \
        --epoch 2 \
        --senta_config_path ./senta_config.json \
        --skip_steps 10 \
        --random_seed 0 \
        --enable_ce
}


export CUDA_VISIBLE_DEVICES=0
train | grep "dev evaluation" | grep "ave loss" | tail -1 | awk '{print "kpis\ttrain_loss_senta_card1\t"$5"\nkpis\ttrain_acc_senta_card1\t"$8"\nkpis\teach_step_duration_senta_card1\t"$11}' | tr -d "," | python _ce.py
sleep 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | grep "dev evaluation" | grep "ave loss" | tail -1 | awk '{print "kpis\ttrain_loss_senta_card4\t"$5"\nkpis\ttrain_acc_senta_card4\t"$8"\nkpis\teach_step_duration_senta_card4\t"$11}' | tr -d "," | python _ce.py
