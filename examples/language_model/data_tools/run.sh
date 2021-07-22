python  trans_to_json.py --input_path  ./part-00001 --output_path sg_sample
python  create_pretraining_data.py --model_name gpt-cpm-large-cn  --tokenizer_name GPTChineseTokenizer --data_format JSON --input_path sg_sample.jsonl  --workers  10 --log-interval 100
