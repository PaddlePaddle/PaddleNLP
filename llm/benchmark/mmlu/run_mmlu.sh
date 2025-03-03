
python predict/flask_server.py --model_name_or_path $MODEL_NAME --port 8010 --flash_port 8011

if [ ! -f data.tar ]; then
    wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
fi

tar xf data.tar

python bench_mmlu.py --ip 127.0.0.1 --port 8011