
# python predict/flask_server.py --model_name_or_path $MODEL_NAME --port 8010 --flash_port 8011

if [ ! -f test.jsonl ]; then
    wget https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
fi

python bench_gsm8k.py --ip 127.0.0.1 --port 8011