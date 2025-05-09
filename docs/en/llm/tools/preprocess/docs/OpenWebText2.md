# OpenWebText2

| Name | Text Type | Plain Text Size |
|-|-|-|
| OpenWebText2 | English | 70GB |

## Data Acquisition

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/) is an open-source English web text dataset sourced from Reddit. After deduplication, cleaning, and extraction, it contains over 8 million documents.
This example uses the cleaned [OpenWebText2 data](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version) by EleutherAI.

After downloading, decompress with the following command:

```shell
wget https://paddlenlp.bj.bcebos.com/models/transformers/gpt/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C /path/to/openwebtext
```

## Llama Training Data Preparation

Then use the `create_pretraining_data.py` script for dataset preparation:
```
python -u create_pretraining_data.py \
    --model_name meta-llama/Llama-2-7b \
    --tokenizer_name LlamaTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix llama_openwebtext \
    --workers 40 \
    --log_interval 10000 \
    --data_impl "mmap"
```
Processing takes approximately one hour, yielding the required dataset files `llama_openwebtext.bin` and `llama_openwebtext.idx`.

Organize all preprocessed files into a unified directory for training:

```
mkdir data
mv llama_openwebtext.bin ./data
mv llama_openwebtext.idx ./data
```
