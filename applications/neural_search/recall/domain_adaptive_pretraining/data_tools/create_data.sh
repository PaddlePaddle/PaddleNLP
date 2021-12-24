# wanfangdata/
# python trans_to_json.py  --input_path ./data --output_path baike_sample
python trans_to_json.py  --input_path ./wanfangdata --output_path baike_sample

python -u  create_pretraining_data.py \
    --model_name ernie-1.0 \
    --tokenizer_name ErnieTokenizer \
    --input_path baike_sample.jsonl \
    --split_sentences\
    --chinese \
    --cn_whole_word_segment \
    --output_prefix baike_sample  \
    --workers 1 \
    --log_interval 5