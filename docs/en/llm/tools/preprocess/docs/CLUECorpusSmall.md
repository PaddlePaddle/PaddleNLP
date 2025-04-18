# CLUECorpusSmall

| Name | Text Type | Raw Text Size |
|-|-|-|
| CLUECorpusSmall| Chinese | 14GB |

**Dataset Description**: Suitable for language modeling, pre-training, or generation tasks. Contains over 14GB of data, nearly 4000 well-defined txt files, and 5 billion characters. Main components originate from the nlp_chinese_corpus project.

Includes the following sub-corpora (total 14G):
- News corpus [news2016zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/6bac09db4e6d4857b6d680d34447457490cb2dbdd8b8462ea1780a407f38e12b?responseContentDisposition=attachment%3B%20filename%3Dnews2016zh_corpus.zip)
- Community interaction corpus [webText2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/83da03f7b4974871a52348b41c16c7e3b34a26d5ca644f558df8435be4de51c3?responseContentDisposition=attachment%3B%20filename%3DwebText2019zh_corpus.zip)
- Wikipedia corpus [wiki2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/d7a166408d8b4ffdaf4de9cfca09f6ee1e2340260f26440a92f78134d068b28f?responseContentDisposition=attachment%3B%20filename%3Dwiki2019zh_corpus.zip)
- Comment data corpus [comment2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/b66ddd445735408383c42322850ac4bb82faf9cc611447c2affb925443de7a6d?responseContentDisposition=attachment%3B%20filename%3Dcomment2019zh_corpus.zip)

## Data Access

Users can download via the official GitHub page: https://github.com/CLUEbenchmark/CLUECorpus2020. For convenience, we also provide AI Studio dataset download links: [part1](https://aistudio.baidu.com/aistudio/datasetdetail/60598), [part2](https://aistudio.baidu.com/aistudio/datasetdetail/124357). When using the AI Studio version, you can verify the MD5 values after download.
```shell
> md5sum ./*
8a8be341ebce39ccf5e9524fb0b46b08c5  ./comment2019zh_corpus.zip
4bdc2c941a7adb4a061caf273fea42b8  ./news2016zh_corpus.zip
fc582409f078b10d717caf233cc58ddd  ./webText2019zh_corpus.zip
157dacde91dcbd2e52a60af49f710fa5  ./wiki2019zh_corpus.zip
```

Unzip the files
```shell
unzip comment2019zh_corpus.zip -d  clue_corpus_small_14g/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  clue_corpus_small_14g/news2016zh_corpus
unzip webText2019zh_corpus.zip -d  clue_corpus_small_14g/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  clue_corpus_small_14g/wiki2019zh_corpus
```

Convert txt files to jsonl format
```
python trans_to_json.py  --input_path ./clue_corpus_small_14g --output_path clue_corpus_small_14g.jsonl
```

Now we obtain the dataset in jsonl format.

## Chinese Pre-training Data Preparation

Below are dataset applications for training tasks.

* Example for LLaMA
```shell
python -u  create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --input_path "clue_corpus_small_14g.jsonl" \
    --output_prefix "clue_corpus_small_14g" \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 10000 \
    --workers 48
```

* Example for ERNIE
```shell
python -u create_pretraining_data.py \
    --model_name "ernie-3.0-base-zh" \
    --tokenizer_name "ErnieTokenizer" \
    --input_path "clue_corpus_small_14g.jsonl" \
    --output_prefix "clue_corpus_small_14g" \
    --data_format "JSON" \
    --json_key "text" \
    --split_sentences \
    --data_impl "mmap" \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func "lac" \
    --log_interval 10000 \
    --workers 48
```

- The model_name can be replaced with [other models](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm).
- workers indicates the number of conversion threads

The data consists of approximately `15,702,702` documents. Due to the time-consuming segmentation process, it takes about one hour to complete. The training data will be generated in the current directory:
```
clue_corpus_small_14g.bin
clue_corpus_small_14g.idx
```
Users can utilize this data for pre-training tasks.
