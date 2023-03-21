# Bloom

## 模型介绍

Bloom 。

## 文本生成

* 单卡小模型生成

```bash
python run_generation.py --model_name_or_path "bigscience/bloom-560m"
```

参数说明：
- `model_name_or_path`: 模型名称, 例如：`bigscience/bloom-560m`, `bigscience/bloom-3b`, `bigscience/bloom-7b1`等。


## 模型评估

我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 并将数据放置在data 目录下， 使用如下命令启动评估：

1. WikiText数据集评估
```bash

CUDA_VISIBLE_DEVICES="1" python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_glue" \
    --batch_size 8 \
    --eval_path ./data/wikitext-103/wiki.valid.tokens
```

2. LAMBADA数据集评估
```bash
# 覆盖default.yaml中的eval_path配置字段
python run_eval.py \
    --model_type bloom \
    --model_name_or_path "bigscience/bloom-560m" \
    --tokenizer_name_or_path "bigscience/bloom-560m" \
    --input_dir "old" \
    --output_dir "output_glue" \
    --batch_size 8 \
    --eval_path ./data/./lambada_test.jsonl \
    --cloze_eval
```
