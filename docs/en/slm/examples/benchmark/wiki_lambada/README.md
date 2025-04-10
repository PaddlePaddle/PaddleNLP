## Data Preparation
```
wget https://paddlenlp.bj.bcebos.com/data/benchmark/lambada_test.jsonl
```
```
wget https://paddlenlp.bj.bcebos.com/data/benchmark/wikitext-103.tar.gz
```
```
wget https://paddlenlp.bj.bcebos.com/data/benchmark/wikitext-2.tar.gz
```

## Running the Prediction Script

To evaluate the Lambada dataset, run the following script:
```
python eval.py \
--model_name_or_path /path/to/your/model \
--batch_size 4 \
--eval_path /path/to/your/dataset/lambada_test.jsonl \
--tensor_parallel_degree 1 \
--cloze_eval
```

To evaluate the WikiText dataset, run the following script:
```
python eval.py \
--model_name_or_path /path/to/your/model \
--batch_size 4 \
--eval_path /path/to/your/dataset/wikitext-103/wiki.valid.tokens \
--tensor_parallel_degree 1
```
