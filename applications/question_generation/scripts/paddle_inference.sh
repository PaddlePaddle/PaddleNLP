export CUDA_VISIBLE_DEVICES=3
python deploy/paddle_inference/inference.py \
               --inference_model_dir unimo/static \
               --model_name_or_path "unimo-text-1.0" \
               --output_path unimo/inference/predict.txt \
               --device gpu \
               #    --predict_file /root/project/data/dureader_qg/raw/DuReaderQG/minidev.json \