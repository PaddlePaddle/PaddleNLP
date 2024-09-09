# CLUECorpusSmall

| 名称 | 文本类型 | 纯文本大小 |
|-|-|-|
| CLUECorpusSmall| 中文 | 14GB |

**数据集简介**：可用于语言建模、预训练或生成型任务等，数据量超过14G，近4000个定义良好的txt文件、50亿个字。主要部分来自于nlp_chinese_corpus项目
包含如下子语料库（总共14G语料）：新闻语料[news2016zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/6bac09db4e6d4857b6d680d34447457490cb2dbdd8b8462ea1780a407f38e12b?responseContentDisposition=attachment%3B%20filename%3Dnews2016zh_corpus.zip)， 社区互动语料[webText2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/83da03f7b4974871a52348b41c16c7e3b34a26d5ca644f558df8435be4de51c3?responseContentDisposition=attachment%3B%20filename%3DwebText2019zh_corpus.zip)，维基百科语料[wiki2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/d7a166408d8b4ffdaf4de9cfca09f6ee1e2340260f26440a92f78134d068b28f?responseContentDisposition=attachment%3B%20filename%3Dwiki2019zh_corpus.zip)，评论数据语料[comment2019zh_corpus.zip](https://bj.bcebos.com/v1/ai-studio-online/b66ddd445735408383c42322850ac4bb82faf9cc611447c2affb925443de7a6d?responseContentDisposition=attachment%3B%20filename%3Dcomment2019zh_corpus.zip)。

## 数据获取

用户可以通过官方github网页下载，https://github.com/CLUEbenchmark/CLUECorpus2020 。同时，为方便用户，我们也提供了aistudio数据集下载地址。[part1](https://aistudio.baidu.com/aistudio/datasetdetail/60598)，[part2](https://aistudio.baidu.com/aistudio/datasetdetail/124357)。使用aistudio版本的数据，下载好后，可以核对md5值：
```shell
> md5sum ./*
 8a8be341ebce39cfe9524fb0b46b08c5  ./comment2019zh_corpus.zip
 4bdc2c941a7adb4a061caf273fea42b8  ./news2016zh_corpus.zip
 fc582409f078b10d717caf233cc58ddd  ./webText2019zh_corpus.zip
 157dacde91dcbd2e52a60af49f710fa5  ./wiki2019zh_corpus.zip
```
解压文件
```shell
unzip comment2019zh_corpus.zip -d  clue_corpus_small_14g/comment2019zh_corpus
unzip news2016zh_corpus.zip    -d  clue_corpus_small_14g/news2016zh_corpus
unzip webText2019zh_corpus.zip -d  clue_corpus_small_14g/webText2019zh_corpus
unzip wiki2019zh_corpus.zip    -d  clue_corpus_small_14g/wiki2019zh_corpus
```
将txt文件转换为jsonl格式
```
python trans_to_json.py  --input_path ./clue_corpus_small_14g --output_path clue_corpus_small_14g.jsonl
```
现在我们得到了jsonl格式的数据集。

## 中文预训练数据制作

下面是针对训练任务的数据集应用。

* llama为例
```shell
python -u  create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --tokenizer_name "LlamaTokenizer" \
    --input_path "clue_corpus_small_14g.jsonl" \
    --output_prefix "clue_corpus_small_14g" \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 10000 \
    --workers 48
```

* ernie为例
```shell
python -u  create_pretraining_data.py \
    --model_name "ernie-3.0-base-zh" \
    --tokenizer_name "ErnieTokenizer" \
    --input_path "clue_corpus_small_14g.jsonl" \
    --output_prefix "clue_corpus_small_14g"  \
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

- model_name 可以更换为[其他模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm)。
- workers 表示转化的线程数目

数据共有文档`15702702`条左右，由于分词比较耗时，大概一小时左右可以完成。在当前目录下产出训练所需数据。
```
clue_corpus_small_14g.bin
clue_corpus_small_14g.idx
```
用户可以使用此数据进行预训练任务。
