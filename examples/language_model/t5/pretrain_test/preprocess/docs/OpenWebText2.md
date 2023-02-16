# OpenWebText2

| 名称 | 文本类型 | 纯文本大小 |
|-|-|-|
| OpenWebText2 | 英文 | 70GB |

## 数据获取

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

## GPT训练数据制作

然后使用[proprecess]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0/proprecess) 工具下的`create_pretraining_data.py`脚本进行数据集制作：
```
python -u  create_pretraining_data.py \
    --model_name gpt2-en \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix gpt_openwebtext  \
    --workers 40 \
    --log_interval 10000
```
处理时间约一个小时左右，就可以得到我们需要的`gpt_openwebtext_ids.npy`, `gpt_openwebtext_idx.npz`数据集文件。

为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv gpt_en_dataset_300m_ids.npy ./data
mv gpt_en_dataset_300m_idx.npz ./data
```
