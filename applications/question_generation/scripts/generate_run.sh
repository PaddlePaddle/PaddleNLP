export CUDA_VISIBLE_DEVICES=4
python -u run_gen.py \
    --dataset_name=dureader_qg \
    --predict_file=/root/project/data/dureader_qg/raw/DuReaderQG/minidev.json \
    --model_name_or_path=/root/project/paddle/PaddleNLP/applications/question_generation/unimo/finetune/drawer/model_2270 \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --template=1 \
    --device=gpu