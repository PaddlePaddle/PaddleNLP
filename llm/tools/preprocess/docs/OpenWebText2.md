# OpenWebText2

| 名称 | 文本类型 | 纯文本大小 |
|-|-|-|
| OpenWebText2 | 英文 | 70GB |

## 数据获取

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
# wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

## Llama训练数据制作

然后使用`create_pretraining_data.py`脚本进行数据集制作：
```
python -u  create_pretraining_data.py \
    --model_name meta-llama/Llama-2-7b \
    --tokenizer_name LlamaTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix llama_openwebtext  \
    --workers 40 \
    --log_interval 10000 \
    --data_impl "mmap"
```
处理时间约一个小时左右，就可以得到我们需要的`llama_openwebtext.bin`, `llama_openwebtext.idx`数据集文件。

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv llama_openwebtext.bin ./data
mv llama_openwebtext.idx ./data
```
