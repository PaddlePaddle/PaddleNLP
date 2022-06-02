
python -u data_tools/trans_to_json.py \
    --workers 60 \
    --input_path /home/gongenlei/python_test/ \
    --output_path data_tools/data

python -u data_tools/create_pretraining_data.py \
    --data_format JSON \
    --input_path data_tools/data.jsonl \
    --output_prefix data_tools/code_python  \
    --workers 60 \
    --log_interval 10000
