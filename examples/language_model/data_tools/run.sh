#python  trans_to_json.py --input_path  ./part-00001 --output_path sg_sample
# python  trans_to_json.py \
#     --input_path  ~/ernie_1.0_dataset/wiki_21M \
#     --output_path wiki_21M \
#     --workers 32

# python  create_pretraining_data.py --model_name gpt-cpm-large-cn  --tokenizer_name GPTChineseTokenizer --data_format JSON --input_path sg_sample.jsonl --output_prefix sg_sample  --workers  32 --log-interval 1000
#    --chinese_words_segment \

# python -u  create_pretraining_data.py \
#     --model_name ernie-1.0 \
#     --tokenizer_name ErnieTokenizer \
#     --data_format JSON \
#     --input_path ernie_1.0_dataset_jsonl \
#     --chinese_words_segment \
#     --chinese_splited \
#     --output_prefix ernie_1.0_all_2   \
#     --workers 48 \
#     --log-interval 10000

python -u  create_pretraining_data.py \
    --model_name ernie-1.0 \
    --tokenizer_name ErnieTokenizer \
    --data_format JSON \
    --input_path sg_sample.jsonl \
    --chinese \
    --split_sentence \
    --cn_whole_word_segment \
    --output_prefix xxxsg   \
    --workers 32 \
    --log-interval 100
