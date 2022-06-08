
python -u data_tools/trans_to_json.py \
    --workers 120 \
    --input_path /home/gongenlei/Python/ \
    --output_path data_tools/data

python -u data_tools/create_pretraining_data.py \
    --data_format JSON \
    --model_name alphacode-small \
    --input_path data_tools/data.jsonl \
    --output_prefix data_tools/code_python  \
    --workers 120 \
    --log_interval 10000
