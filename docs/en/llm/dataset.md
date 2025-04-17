# 1. Pretrain Datasets

## 1. Built-in Datasets

Name|Text Type|Raw Text Size|Compatible Models|Time to Process|Source|Download Link bin|Download Link idx|
|-|-|-|-|-|-|-|-|
OpenWebText2|English|70GB|`meta-llama/Llama-2-7b`<br> `meta-llama/Llama-2-7b-chat`<br> `meta-llama/Llama-2-13b`<br> `meta-llama/Llama-2-13b-chat` <br>`facebook/llama-7b`<br> `facebook/llama-13b`<br>| 42min |  [Link](https://skylion007.github.io/OpenWebTextCorpus/) |[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.bin) | [*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.idx) |
|OpenWebText2|English|70GB|`gpt2-en`|37min|[Link](https://skylion007.github.io/OpenWebTextCorpus/)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/gpt/mmap/gpt2-en-mmap.idx)|
CLUECorpusSmall|Chinese|14GB|`idea-ccnl/ziya-llama-13b-v1`|15min|[Link](https://github.com/CLUEbenchmark/CLUECorpus2020)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/ziya/mmap/ziya_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/ziya/mmap/ziya_mmap.idx)|
-|Chinese|14GB|`baichuan-inc/Baichuan-7B`|12min||[* bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan/mmap/baichuan_mmap.idx)|
-|Chinese|14GB|
`linly-ai/chinese-llama-2-7b` <br>`linly-ai/chinese-llama-2-13b`|19min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/linly/mmap/linly_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/linly/mmap/linly_mmap.idx)|
-|Chinese|14GB|`baichuan-inc/Baichuan-13B-Base` <br>`baichuan-inc/Baichuan-13B-Chat`|14min || [*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan13b/mmap/baichuan13b_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan13b/mmap/baichuan13b_mmap.idx)|
-|Chinese|14GB|`baichuan-inc/Baichuan2-7B-Base`<br> `baichuan-inc/Baichuan2-7B-Chat`<br> `baichuan-inc/Baichuan2-13B-Base`<br> `baichuan-inc/Baichuan2-13B-Chat` |13min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan2/mmap/baichuan2_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/baichuan2/mmap/baichuan2_mmap.idx)|
-|Chinese|14GB|`meta-llama/Llama-2-7b`<br> `meta-llama/Llama-2-7b-chat`<br> `meta-llama/Llama-2-13b`<br> `meta-llama/Llama-2-13b-chat`<br> `facebook/llama-7b` <br> `facebook/llama-13b`<br> `FlagAlpha/Llama2-Chinese-7b-Chat`<br> `FlagAlpha/Llama2-Chinese-13b-Chat` |20min|| [*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/llama/mmap/llama_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/clue/llama/mmap/llama_mmap.idx)|
WuDaoCorpus2.0 Base|Chinese|200GB|
`idea-ccnl/ziya-llama-13b-v1`|3h 35min| [Link](https://data.baai.ac.cn/details/WuDaoCorporaText)|[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/ziya/mmap/ziya_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/ziya/mmap/ziya_mmap.idx)|
WuDaoCorpus2.0 Base|Chinese|200GB|`baichuan-inc/Baichuan-7B`|2h 52min||[*bin](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/baichuan/mmap/baichuan_mmap.bin)|[*idx](https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/wudao/baichuan/mmap/baichuan_mmap.idx)|

Download the bin and idx files and place them in the same directory. Specify input_dir in the pretraining script.

If you need to create a custom dataset, the overall process is as described in section 2.1, with detailed steps provided in section 2.2 below.

## 2. Custom Dataset

### 2.1 Data Creation Workflow

|Step|Phase|Data Format|Example|
|-|-|-|-|
|0️⃣Initial State|-|Raw Data: <br/> **Separate docs with blank lines** <br/> - Chinese: Default line breaks as sentence endings<br/> - English: Use nltk for sentence segmentation|```飞桨是功能完备、开源开放的产业级深度学习平台。``` <br/> ```飞桨拥有核心训练和推理框架、基础模型库。``` <br/><br/> ```PaddleNLP是自然语言处理领域的优秀工具。```|
|1️⃣Raw Data Conversion<br/>`trans_to_json.py`|Preprocessing <br>Input: 0️⃣Initial State <br>Output: jsonl|jsonl format: Each doc as a json string per line|```{"text": "飞桨是功能完备、开源开放的产业级深度学习平台。飞桨拥有..."}```<br/>```{"text": "PaddleNLP是自然语言..."}```|
|2️⃣Data Tokenization<br/>`create_pretrain_data.py`|Preprocessing|bin format: Tokenized data as token IDs <br/>idx format: Sentence and document position indices| - |

### 2.2 Detailed Preparation
Here we use the ziya-llama-13b-v1 model as an example to illustrate the complete data preparation process.

**2.2.1 Raw Data**

First download sample data:

```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/baike.txt
cd ..
```

**2.2.2 Convert Raw Data to jsonl Format**

Use trans_to_json.py to convert data into json string format. Below is the script usage instructions:
```bash
optional arguments:
  -h, --help
  --input_path INPUT_PATH
                        "Required. Can be a directory or single file. By default searches up to two subdirectory levels in folders."
  --output_path OUTPUT_PATH
                        "Required. Output filename."
  --json_key JSON_KEY
                        "Recommended to keep default. Default key is 'text'"
  --doc_spliter DOC_SPLITER
                        "Article delimiter. Can be modified according to actual needs. Default uses blank lines as article delimiter."
  --min_doc_length MIN_DOC_LENGTH
                        "Optional. Filter articles shorter than this length. Default 10"
  --workers WORKERS
                        "Optional. Multi-process conversion for scenarios with large number of files in input_path."
                        "Each file will be processed by different workers"
  --log_interval LOG_INTERVAL
                        "Optional. Here interval refers to the number of processed files between logging."
  --no-merge
                        "Optional. By default not enabled. Default behavior merges converted jsonl texts from all files into single output file."
  --no-shuffle
                        "Optional. By default not enabled. Default shuffles processed documents."
```

Using the instructions, we execute the following command to obtain baike_sample.jsonl file. Here we shuffle all documents:
```
python trans_to_json.py --input_path ./data --output_path baike_sample
```

Check data:
```json
{"text": "China's process of emulating Western industrial development began to progress smoothly from the establishment of the Nationalist Government of the Republic of China until the eve of the Sino-Japanese War, despite multiple interruptions from internal and external factors. It was not until the conclusion of the Sino-Japanese War and the Chinese Civil War that China entered a relatively prolonged period of peaceful development.\nSince the 1980s, the Deng Xiaoping administration announced the reform and opening-up policy, initiating the implementation of a socialist market economy and promoting economic system reforms. By 2010, mainland China's GDP had exceeded $7.2 trillion, becoming the world's second-largest economy after the United States. While China is widely recognized as the world's fastest-growing economy, its per capita gross national product remains at a medium level globally (ranked 89th), and it faces increasing constraints from resource limitations and widening wealth disparity. Among China's provinces, Guangdong ranks first in GDP, while Zhejiang is the wealthiest province in terms of per capita income. Economic ties between mainland China, Hong Kong, Macau, and Taiwan have grown increasingly close in the process of globalization.\n"}
```

**2.2.3 Data ID Processing**

In this section, we use the `create_pretraining_data.py` script to perform tokenize and ID conversion on the previously obtained `baike_sample.jsonl`. Models can reference existing vocabulary lists.
```bash
optional arguments:
  --model_name MODEL_NAME
                        "Must be specified, e.g.: idea-ccnl/ziya-llama-13b-v1"
  --tokenizer_name {LlamaTokenizer}
                        "Tokenizer corresponding to the model. Llama models require LlamaTokenizer"

data input/output:
  --input_path INPUT_PATH
                        "Must be specified. Directory of input jsonl files"
  --output_prefix OUTPUT_PREFIX
                        "Must be specified. Prefix for output files."
                        "If name is XXX, outputs will be XXX.bin and XXX.idx."
                        "bin file: tokenized data; idx file: sentence/document position indices."
  --data_format {JSON}
                        "No need to set. Currently only processes jsonl format by default"
  --json_key JSON_KEY
                        "Key for text strings in json. Same as json_key in trans_to_json.py, default 'text'"
  --split_sentences
                        "Whether to split documents into sentences. Generally not needed for GPT."
  --data_impl
                        "Processed data format. Options: 'mmap' or 'lazy',"
                        "where 'mmap' creates memory mapping during training, 'lazy' reads directly from file."

chinese words:
  --chinese
                        "Required if split_sentences is set and processing Chinese text."
  --cn_whole_word_segment
                        "Optional. Whether to apply Whole Word Masking (WWM). Generally not needed for GPT models."
  --cn_seg_func {lac,seg,jieba}
                        "Default: jieba (faster). lac model is more accurate but computationally intensive."
  --cn_splited
                        "Optional. For pre-segmented text. When set, cn_seg_func is disabled."
                        "E.g. pre-segmented text: '中国 效仿 西方 发展 工业 的过 程'"
  --cn_split_dimer CN_SPLIT_DIMER
                        "Used with cn_splited. Default delimiter is whitespace."

common config:
  --append_eos
                        "GPT-specific. Adds EOS token at document end for GPT models."
  --log_interval LOG_INTERVAL
                        "Logging interval (number of processed lines/docs between logs)"
  --workers WORKERS
                        "Number of processes for text tokenization."
```
We can obtain the processed pre-training data through the following training script:
```bash
python -u  create_pretraining_data.py \
    --model_name "idea-ccnl/ziya-llama-13b-v1" \
    --data_format "JSON" \
    --input_path "/home/data/baike_sample.jsonl" \
    --append_eos \
    --output_prefix "/home/data/baike_sample"  \
    --workers 1 \
    --log_interval 5 \
    --data_impl "mmap"
```

1. If using pre-tokenized corpus, set `--cn_split` to True and specify `--cn_split_dimer` (e.g., space).
2. For custom vocabulary, set model_name to the directory containing the vocabulary.

After processing, we obtain preprocessed training data baike_sample.bin and document index file `baike_sample.idx` in the "/home/data/" directory.

**2.2.4 (Optional) Merge Datasets**

For large input files requiring preprocessing, the processing time might be excessively long. In such cases, consider splitting the jsonl file into multiple smaller files, process them in parallel using create_pretraining_data.py, and then merge the resulting .bin & .idx files using the following merge script. Merging two 500GB files into 1TB typically takes about 1 hour.
```bash
python merge.py \
    --input "/home/data/" \
    --output-prefix "/home/data/merged" \
    --data_impl mmap
```

Usage:
```bash
arguments:
  --input INPUT_PATH
                        "Directory containing files to be merged. Files should be named in the order required for merging"
                        "E.g., 1.bin/1.idx, 2.bin/2.idx..."
  --output_prefix OUTPUT_PREFIX
                        "Prefix for merged output files. Given prefix XXX, will generate XXX.bin and XXX.idx".
  --data_impl {mmap,lazy}
                        "Data format before/after merging. Options: 'mmap' or 'lazy'. All input files must have consistent format."
```
After processing with the above merge script, the "/home/data" directory will contain merged.bin and merged.idx files, which are combined from the small files in "/home/data/".

**Note: A single dataset should not be too large to avoid int32 overflow. It is recommended that the number of docs in a single file does not exceed 500 million.**

## Common Dataset Construction

[CLUECorpus2020 Corpus Construction](./tools/preprocess/docs/CLUECorpus2020.md)

[CLUECorpusSmall Corpus Construction](./tools/preprocess/docs/CLUECorpusSmall.md)

[OpenWebText2 Corpus Construction](./tools/preprocess/docs/OpenWebText2.md)

[WuDaoCorpus2.0 Base Corpus](./tools/preprocess/docs/WuDaoCorpusBase.md)
