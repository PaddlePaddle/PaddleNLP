# WuDaoCorpus2.0 Base 语料


| 名称 | 文本类型 | 纯文本大小 |
|-|-|-|
| WuDaoCorpus2.0 Base| 中文 | 200GB |


WuDaoCorpora是悟道爬取的中文大规模语料。整体数量为3TB，目前开源的部分为WuDaoCorpus2.0 bases数据集，大小为200GB。
用户微信登录[官网](https://resource.wudaoai.cn/home)，即可直接下载数据。下载好的压缩数据约 64GB
```
64GB WuDaoCorpus2.0_base_200G.rar
```


```shell
python wudao_process.py \
    --input_path WuDaoCorpus2.0_base_200G \
    --workers 40  \
    --ouput_path ./wudao_lac_cut \
```
注：预训练需要实现 SOP( Sentence Order Predict) 任务，在分词的同时，我们使用 简单规则 进行了文本断句。如果语料只有一句话，建议去除SOP loss，训练时设置 `binary_head=False`。

文本转化完成后。我们使用 `../data_tools/trans_to_json.py`重新转换为jsonl格式（分词完毕）。
```shell
python ../data_tools/trans_to_json.py  \
    --input_path ./wudao_lac_cut \
    --output_path wudao_corpus_200g_0623.jsonl \
    --workers 40 \
    --no-shuffle
```
