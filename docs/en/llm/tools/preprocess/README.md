# PaddleNLP Pretraining Data Process

This example aims to establish best practices for PaddleNLP pretrained models.

We divide the pretraining data process into the following stages:

- Raw Data Conversion: Convert raw text into jsonl (JSON Lines) format.
- Data Tokenization: Sentence splitting, word segmentation, and tokenization into token IDs.
- Training Index File Generation: Create sample indices for train/valid/test sets.
- Dynamic Token Masking (Optional): Real-time text masking at Python level.

This directory mainly contains the following files:
```
├── create_pretraining_data.py
├── merge.py
├── trans_to_json.py
├── words_segmentation.py
└── README.md
```

### Environment Dependencies

- tqdm
- numpy
- pybind11
- fast_dataindex
- lac (optional)
- zstandard (optional)

Installation command: `pip install tqdm numpy pybind11 fast_dataindex lac zstandard`. Additionally, some features require `g++>=4.8`.
## Training Full Pipeline Data

PaddlePaddle is a self-developed, fully-featured, open-source industrial-grade deep learning platform. It integrates core deep learning training and inference frameworks, foundational model libraries, end-to-end development kits, and abundant tool components.

| Step                                               | Phase&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Data Format                                                                                                                   | Example                                                                                                                                                                           |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0️⃣Initial State                                    | -                                                                                                                                 | Raw data: <br/> **Separate docs with blank lines** <br/> - Chinese: Default line breaks as sentence boundaries <br/> - English: Use nltk for sentence segmentation | ```PaddlePaddle is a fully-featured, open-source industrial-grade deep learning platform.``` <br/> ```It contains core training and inference frameworks, foundational model libraries.``` <br/><br/> ```PaddleNLP is an excellent tool in NLP field.``` |
| 1️⃣Raw Data Conversion<br/>`trans_to_json.py`         | Preprocessing <br>Input: 0️⃣Initial State <br>Output: jsonl                                                                                         | jsonl format: Each doc corresponds to a json string per line                                                                                            | ```{"text": "Paddle is a feature-complete, open-source, industry-grade deep learning platform. Paddle has..."}```<br/>```{"text": "PaddleNLP is the natural language..."}```                                                           |
| ❇️(**Optional**)Chinese Word Segmentation<br/>`words_segmentation.py` | Corpus Segmentation: Chinese WWM <br>Input: jsonl <br>Output: 0️⃣Initial State                                                                            | Convert jsonl format data back to original segmented format <br>                                                                              | ```Paddle is a feature-complete, open-source, industry-grade deep learning platform.``` <br/> ```Paddle has core training and inference frameworks, and a base model library.``` <br/><br/> ```PaddleNLP is an excellent tool in the field of natural language processing.``` |
| 2️⃣Token to ID Conversion<br/>`create_pretrain_data.py` | Convert text to token IDs <br>Input: jsonl <br>Output: bin & meta | Generate binary files required for training, convert text to token IDs and save in binary format. The meta file contains metadata information. | ```# Example of processed data structure: [CLS] token1 [SEP] token2 [SEP] ...``` |
| Process                                                                                                                         | Preprocessing                                                                                                                      | Format: <br/>- `bin`: Token IDs after data ID-ification <br/>- `idx`: Sentence/article position indices                          | -                                                                                                                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3️⃣Training index file generation                                                                                             | Training initialization                                                                                                           | `npy` format: <br/> Generates sample index files (train/valid/test) based on max training steps                                 | -                                                                                                                                                                                  |
| 4️⃣Dynamic token masking (optional)                                                                                          | Dataset sampling                                                                                                                   | None                                                                                                                           | -                                                                                                                                                                                  |


Notes:
- **❇️(Optional) Chinese word segmentation** is an optional step for WWM in Chinese pretraining
  - When dataset is small, word segmentation time is negligible. Direct use without segmentation is acceptable.
`create_pretrain_data.py` Step requires tokenization:
- Purpose: To perform tokenization in advance, accelerating subsequent data ID conversion steps.
- If input files are in jsonl format, it's preferable to use multiple files and enable the `no-merge` option during `trans_to_json.py`.
- When working with large datasets or requiring multiple data conversion attempts, pre-tokenization avoids rerunning tokenization each time in `create_pretrain_data.py`.
- After conversion, you need to redo Step 1️⃣`Raw Data Conversion trans_to_json.py`, and finally set the `--cn_splited=True` parameter in Step 2️⃣`Data ID Conversion`.
- Step 2️⃣`Data ID Conversion` can also perform tokenization simultaneously, eliminating the need for ❇️`Chinese Tokenization` step.

## Data Tutorial Summary

For currently available open-source datasets, PaddleNLP provides detailed data tutorials. Click the corresponding dataset links to start data preparation:

| Name                                             | Text Type | Plain Text Size | Compatible Models |
|--------------------------------------------------|-----------|-----------------|--------------------|
| [CLUECorpusSmall](./docs/CLUECorpusSmall.md)     | Chinese   | 14GB            | Llama             |
| [OpenWebText2](./docs/OpenWebText2.md)           | English   | 70GB            | Llama             |
| [WuDaoCorpus2.0 Base](./docs/WuDaoCorpusBase.md) | Chinese   | 200GB           | Llama             |
| [CLUECorpus2020](./docs/CLUECorpus2020.md)       | Chinese   | 200GB           | Llama             |

## Pretraining Detailed Preparation

Below we take ziya-llama-13b-v1 pretraining as an example to briefly illustrate the complete pretraining workflow.

### Raw Data
First download sample data:
```
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/data_tools/baike.txt
cd ..
```

### Raw Data Conversion to jsonl Format
Use `trans_to_json.py` to convert into json string format. Below is the script usage instructions:
```markdown
optional arguments:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        Path to your raw files. Folder or file path.
                        Must be specified. Can be a folder or a single file. By default, search up to two levels of subdirectories.
  --output_path OUTPUT_PATH
                        Path to save the output json files.
                        Must be specified. The filename for output.
  --json_key JSON_KEY   The content key of json file.
                        Recommended to keep default (default key is "text")
  --doc_spliter DOC_SPLITER
                        Spliter between documents. We will strip the line. If using blank line to split documents, leave it empty.
                        Modify according to actual needs. Default uses blank line as document separator.
  --min_doc_length MIN_DOC_LENGTH
                        Minimum character length of a document.
                        Optional. Filter documents shorter than this value (default 10)
  --workers WORKERS     Number of worker processes to launch
                        Optional. Use multiple processes for conversion, suitable when input_path contains large number of files. Each file is processed by separate worker.
  --log_interval LOG_INTERVAL
                        Interval between progress updates.
                        Optional. This interval refers to the number of files processed between updates.
  --no-merge            Don't merge the files.
                        Optional. By default not enabled. Default behavior is to concatenate converted jsonl texts from all files into single output file.
  --no-shuffle          Don't shuffle the files.
                        Optional. By default not enabled. Default behavior shuffles processed documents.
```

With the instructions, we can use the following simple command to obtain the `baike_sample.jsonl` file. Here, we shuffled all documents:
```shell
python trans_to_json.py --input_path ./data --output_path baike_sample
```
```shell
# View data
head -1 baike_sample.jsonl
{"text": "China's process of emulating Western industrial development progressed smoothly from the establishment of the Nationalist Government of the Republic of China until the eve of the Sino-Japanese War, despite multiple interferences from internal and external factors. It was not until after the Sino-Japanese War and the Chinese Civil War that China entered a relatively long-term period of peaceful development.\nSince the 1980s, the Deng Xiaoping administration announced the reform and opening-up policy, began implementing a socialist market economy, and promoted economic system reforms. By 2010, mainland China's GDP had exceeded $7.2 trillion, becoming the world's second-largest economy after the United States. While China is widely recognized as the world's fastest-growing economy, its per capita gross national product remains at a medium global level (ranking 89th), and it faces increasing constraints from resource limitations and widening wealth gaps. Among China's provinces, Guangdong ranks first in GDP, while Zhejiang has the highest per capita income. Economic ties between mainland China, Hong Kong, Macau, and Taiwan have grown increasingly close in the process of globalization.\n"}
```

### Data IDization
In this section, we use the `create_pretraining_data.py` script to perform tokenize id conversion on the previously obtained `baike_sample.jsonl`.
```
optional arguments:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        The model to use.
                        Required. Example: idea-ccnl/ziya-llama-13b-v1. Refer to existing model names at https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm

data input/output:
  --input_path INPUT_PATH
                        Path to input JSON files.
                        Required. Input directory containing jsonl files.
  --output_prefix OUTPUT_PREFIX
                        Output prefix for storing result files.
                        Required. Output file name (e.g. XXX generates XXX.bin and XXX.idx).
                        .bin file contains tokenized IDs; .idx file contains sentence/article indices.
  --data_format {JSON}  Currently only supports JSON format with one document per line.
                        No need to set. Default processes jsonl format.
  --json_key JSON_KEY   For JSON format. Space-separated list of keys to extract from json.
                        The json key for text content. Same as json_key in trans_to_json.py. Default key is "text".

  --split_sentences     Split documents into sentences.
                        Whether to split documents into sentences. Typically required for BERT/ERNIE models, not for GPT.
  --data_impl {mmap,lazy}
                        Convert json to mmap/lazy format.
                        Processed data format. "mmap" uses memory mapping, "lazy" reads directly from file.

chinese words:
  --chinese             Whether Chinese corpus requires word segmentation.
                        Required when split_sentences is enabled for Chinese text.
  --cn_whole_word_segment
                        Whether to apply Whole Word Masking (WWM) strategy for Chinese words.
                        Optional. Typically required for BERT/ERNIE models, not for GPT.
  --cn_seg_func {lac,seg,jieba}
                        Word segmentation function for Chinese text.
                        Default: jieba (faster). lac provides more accurate segmentation with higher computational cost.
  --cn_splited          Whether Chinese text is pre-segmented.
                        Optional. When enabled, cn_seg_func is ignored. Example: pre-segmented text "中国 效仿 西方 发展 工业 的过 程"
  --cn_split_dimer CN_SPLIT_DIMER
                        Delimiter between Chinese words for pre-segmented text.
                        Used with cn_splited. Default delimiter is whitespace.

common config:
  --append_eos          Append <eos> token at document end.
                        For GPT-style models. Adds <eos> if not present in tokenizer. Warnings shown if eos_token not in tokenizer.
  --log_interval LOG_INTERVAL
                        Progress logging interval.
                        Logs progress every N processed lines/documents.
  --workers WORKERS     Number of worker processes.
                        Number of parallel processes for text tokenization.
  --max_repeated_len    Maximum allowed length of repeated characters.
                        Maximum retained repetitions length for consecutive characters.
```
By running the script below, we can obtain the processed pretraining data, token ids: `baike_sample.bin`, and article index information `baike_sample.idx`.

* For llama model:
```shell
python -u create_pretraining_data.py \
    --model_name_or_path "idea-ccnl/ziya-llama-13b-v1" \
    --input_path "baike_sample.jsonl" \
    --output_prefix "baike_sample" \
    --data_format "JSON" \
    --json_key "text" \
    --data_impl "mmap" \
    --append_eos \
    --log_interval 5 \
    --workers 40
```

* For ernie model:
```shell
python -u create_pretraining_data.py \
    --model_name_or_path "ernie-3.0-base-zh" \
    --input_path "baike_sample.jsonl" \
    --output_prefix "baike_sample" \
    --data_format "JSON" \
    --json_key "text" \
    --split_sentences \
    --data_impl "mmap" \
    --chinese \
    --cn_whole_word_segment \
    --cn_seg_func "jieba" \
    --log_interval 5 \
    --workers 40
```

Notes:
1. If using pre-segmented corpus, set `--cn_splited` to True and specify `--cn_split_dimer` such as space.
2. When using a custom vocabulary, specify `model_name` as the directory path containing the vocabulary.

If the preprocessing file is too large, the script may take very long time. In this case, consider splitting the jsonl file into multiple smaller files, processing them in parallel with create_pretraining_data.py to obtain multiple .bin & .idx files. Then use the following merge script to combine multiple small .bin & .idx files:
```
python merge.py \
    --input /root/data \
    --output-prefix /root/data/merged \
    --data_impl mmap
```

Usage instructions:
```
arguments:
  --input INPUT_PATH
                        Path to the folder where the files to be merged.
                        Folder containing files to merge. The small files inside should be ordered by merge sequence, e.g. 1.bin/1.idx, 2.bin/2.idx...
  --output_prefix OUTPUT_PREFIX
                        Output prefix to store output file.
                        Name prefix for merged files. Given name XXX, will generate XXX.bin and XXX.idx.
  --data_impl {mmap,lazy}
                        Convert the json into mmap/lazy format.
                        Data format before/after merging. Options: "mmap" or "lazy". All input files must have consistent format.
```

### Pre-training Preparation
After obtaining the processed training data, model pre-training can begin. Simply copy the preprocessed data to the data directory to start pre-training.
```shell
mkdir data
mv ./preprocess/baike_sample* ./data
```

* For LLaMA pre-training, refer to [Pre-training](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm).
* For ERNIE pre-training, refer to [Pre-training](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/slm/model_zoo/ernie-1.0/pretraining_introduction.md).


Code Notes:
- Dynamic masking related code is implemented in `./data_tools/dataset_utils.py`
  Users can flexibly modify masking strategies according to their needs. Refer to the `create_masked_lm_predictions` function in `dataset_utils.py`.
  Customizable options include: do_whole_word_mask, favor_longer_ngram, do_permutation, geometric_dist, etc.
  For reference, see [Megatron](https://github.com/NVIDIA/Megatron-LM) for these lm_mask strategies.

## References

Note: Most data processing workflows reference [Megatron](https://github.com/NVIDIA/Megatron-LM). Special thanks to the contributors.
