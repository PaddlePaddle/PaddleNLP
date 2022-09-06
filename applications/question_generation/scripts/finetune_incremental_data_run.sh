unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "2,3,5,6" --log_dir ./unimo/finetune/merge9/log run_gen.py \
    --dataset_name=dureader_qg \
    --train_file=/root/project/data/qa-dataset/qa-clean-qg-merge/merge9.json \
    --model_name_or_path='unimo-text-1.0' \
    --save_dir=./unimo/finetune/merge9/checkpoints \
    --output_path=./unimo/finetune/merge9/predict.txt \
    --logging_steps=100 \
    --save_steps=3000 \
    --epochs=10 \
    --batch_size=16 \
    --learning_rate=5e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_train \
    --do_predict \
    --max_dec_len=40 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --template=1 \
    --device=gpu



unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "2,3,5,6" --log_dir ./unimo/finetune/merge9_finetune/log run_gen.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/merge9/checkpoints/model_best \
    --save_dir=./unimo/finetune/merge9_finetune/checkpoints \
    --output_path=./unimo/finetune/merge9_finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=400 \
    --epochs=10 \
    --batch_size=16 \
    --learning_rate=5e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=50 \
    --do_train \
    --do_predict \
    --max_dec_len=40 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --adversarial_training=None \
    --template=1 \
    --device=gpu




# unset CUDA_VISIBLE_DEVICES
# python -m paddle.distributed.launch --gpus "1" --log_dir ./unimo/finetune/dureader_full/log run_gen.py \
#     --dataset_name=dureader_qg \
#     --train_file=/root/project/data/qa-dataset/qa-clean-qg/dureader_data.json \
#     --model_name_or_path='unimo-text-1.0' \
#     --save_dir=./unimo/finetune/dureader_full/checkpoints \
#     --output_path=./unimo/finetune/dureader_full/predict.txt \
#     --logging_steps=100 \
#     --save_steps=400 \
#     --epochs=10 \
#     --batch_size=16 \
#     --learning_rate=5e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=50 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=40 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --adversarial_training=None \
#     --template=1 \
#     --device=gpu



# unset CUDA_VISIBLE_DEVICES
# python -m paddle.distributed.launch --gpus "1" --log_dir ./unimo/finetune/dureader_full_finetune/log run_gen.py \
#     --dataset_name=dureader_qg \
#     --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/dureader_full/checkpoints/model_best \
#     --save_dir=./unimo/finetune/dureader_full_finetune/checkpoints \
#     --output_path=./unimo/finetune/dureader_full_finetune/predict.txt \
#     --logging_steps=100 \
#     --save_steps=400 \
#     --epochs=10 \
#     --batch_size=16 \
#     --learning_rate=5e-5 \
#     --warmup_propotion=0.02 \
#     --weight_decay=0.01 \
#     --max_seq_len=512 \
#     --max_target_len=50 \
#     --do_train \
#     --do_predict \
#     --max_dec_len=40 \
#     --min_dec_len=3 \
#     --num_return_sequences=1 \
#     --adversarial_training=None \
#     --template=1 \
#     --device=gpu
