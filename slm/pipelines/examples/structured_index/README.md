# 文档层次化索引

## 方法

1. 加载数据(load)：把需要处理的 pdf 或者 html 文档加载到流程中。
2. 文档语篇结构解析(parse)：使用大语言模型对文档进行语篇结构解析，根据语义重新切分文章，并解析出文档的语篇结构树。
3. 层次化摘要生成(summary)：根据语篇结构树，自底向上对文档解析结果进行层次化摘要生成，生成不同层次信息的摘要。
4. 层次化索引构建(index)：通过文本编码器，将这些不同层次的文本摘要片段嵌入到稠密检索的向量空间中，从而构建一个层次化文本索引。这种索引不仅包含了局部信息，还包含了较高层次的全局信息，能够支持对多种粒度信息的召回，以适应用户查询中的不同信息需求。

## 安装

### 环境依赖

推荐安装 gpu 版本的[PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html)，以 cuda11.7的 paddle 为例，安装命令如下：

```bash
conda install paddlepaddle-gpu==2.6.2 cudatoolkit=11.7 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```
安装其他依赖：
```bash
pip install -r requirements.txt
```

### 数据准备

- 源文档：需要构建层次化索引的文档语料，如路径`data/source`下的文档示例。每篇文档为单个文件，目前支持 PDF 或 HTML 格式。
脚本`data/source/download.sh`可用于下载示例文档：
```bash
apt install jq -y # 安装 jq 工具, 需要系统权限，若已安装可跳过
cd data/source
bash download.sh
```
- 查询文件：用户查询文本，目前支持 json 格式，单条查询为`query_id: query_text`，如查询文件示例`data/query.json`。


## 运行

### 索引构建

为单个文档文件构建层次化索引：
```bash
python construct_index.py \
--source data/source/2308.12950.pdf \
--parse_model_name_or_path Qwen/Qwen2-72B-Instruct \
--summarize_model_name_or_path Qwen/Qwen2-72B-Instruct \
--encode_model_name_or_path BAAI/bge-large-en-v1.5 \
--log_dir .logs
```

为整个路径下的所有文档文件构建层次化索引：
```bash
python construct_index.py \
--source data/source \
--parse_model_name_or_path Qwen/Qwen2-72B-Instruct \
--summarize_model_name_or_path Qwen/Qwen2-72B-Instruct \
--encode_model_name_or_path BAAI/bge-large-en-v1.5 \
--log_dir .logs
```

可调整参数包括：
- `source`: 需要构建层次化索引的所有源文件的目录路径，或需要构建层次化索引的单个源文件

- `parse_model_name_or_path`: 用于文档语篇结构解析(parse)的模型的名称或路径

- `parse_model_url`: 用于文档语篇结构解析(parse)的模型的 URL。如果不需要则不要写这个参数

- `summarize_model_name_or_path`: 用于文档层次化摘要(summarize)的模型的名称或路径

- `summarize_model_url`: 用于文档层次化摘要(summarize)的模型的 URL。如果不需要则不要写这个参数

- `encode_model_name_or_path`: 用于文本编码的模型的名称或路径

- `log_dir`: 保存日志文件的路径

层次化索引的结果会保存在 `data/index/{encode_model_name_or_path}`, 每个源文档在此路径下有两个对应的缓存文件用于检索：`.pkl`文件包含源文档的层次化摘要文本，`.npy`文件包含对应的摘要文本编码向量。
例如，对 `data/source/CodeLlama.pdf` 构建的层次化索引缓存文件包括 `index/BAAI/bge-large-en-v1.5/CodeLlama.npy` 和 `index/BAAI/bge-large-en-v1.5/CodeLlama.pkl`。

### 检索输出

在层次化索引中检索查询相关摘要片段，并输出检索结果。

以文件形式查询多条文本：
```bash
python query.py \
--search_result_dir data/search_result \
--encode_model_name_or_path BAAI/bge-large-en-v1.5 \
--log_dir .logs \
--query_filepath data/query.json \
--top_k 5 \
--embedding_batch_size 128
```

以文本形式查询单条文本：
```bash
python query.py \
--search_result_dir data/search_result \
--encode_model_name_or_path BAAI/bge-large-en-v1.5 \
--log_dir .logs \
--query_text "What is the relationship between CodeLlama and Llama?" \
--top_k 5 \
--embedding_batch_size 1
```

可调整参数为：
- `search_result_dir`: 保存查询的检索结果的路径

- `encode_model_name_or_path`: 用于文本编码的模型的名称或路径

- `query_filepath`: query 的文件路径。如果有，它必须是一个查询字典的 JSON 文件

- `query_text`: 单条 query 的文本。如果有，它必须是一个字符串

- `top_k`: 设置为每条查询返回前 top_k 个结果

- `embedding_batch_size`: 编码 query 时的批处理大小

- `log_dir`: 保存日志文件的路径

检索结果保存在`{search_result_dir}/{encode_model_name_or_path}`路径下。此路径下的每个结果文件对应一次查询调用，包含若干条查询，即每次会在`{search_result_dir}/{encode_model_name_or_path}`路径下产生一个`query_{时间戳}.json`的文件记录查询结果，由查询 ID 唯一标识单次查询中的每条查询。若通过`query_text`传入查询文本，则查询 ID 设置为`"0"`。

例如上述单条查询的检索结果如下：
```json
{
    "0": {
        "query": "What is the relationship between CodeLlama and Llama?",
        "hits": [
            {
                "corpus_id": 122,
                "score": 0.7032119035720825,
                "content": "CoDE LLAMA is a family of large language models for code, based on LLAMA 2, designed for state-of-the-art performance in programming tasks, including infilling, large context handling, and zero-shot instruction-following, with a focus on safety and alignment.",
                "source": "data/source/2308.12950.pdf",
                "level": 0
            },
            {
                "corpus_id": 127,
                "score": 0.6490256786346436,
                "content": "CoDE LLAMA models are general-purpose code generation tools, with specialized versions like CoDE LLAMA -PyTHON for Python code and CoDE LLAMA -INsTRUCT for understanding and executing instructions.",
                "source": "data/source/2308.12950.pdf",
                "level": 3
            },
            {
                "corpus_id": 128,
                "score": 0.6398724317550659,
                "content": "CoDE LLAMA -PyTHON is specialized for Python code generation, while CoDE LLAMA -INsTRUCT models are designed to understand and execute instructions.",
                "source": "data/source/2308.12950.pdf",
                "level": 4
            },
            {
                "corpus_id": 161,
                "score": 0.6116989254951477,
                "content": "CoDE LLAMA models are designed for real-world applications, excelling in infilling and large context handling, and they achieve state-of-the-art performance on code generation benchmarks while ensuring safety and alignment.",
                "source": "data/source/2308.12950.pdf",
                "level": 2
            },
            {
                "corpus_id": 129,
                "score": 0.6056838631629944,
                "content": "CoDE LLAMA -INsTRUCT are instruction-following models designed to understand and execute instructions.",
                "source": "data/source/2308.12950.pdf",
                "level": 5
            }
        ]
    }
}
```
其中，每条 query 检索结果的格式如下:
```
查询ID: {
        "query": 查询文本,
        "hits": [
            {
                "corpus_id": 本条语料在所有语料中的编号,
                "score": 相似度分数,
                "content": 语料摘要内容,
                "source": 语料来源文档的路径,
                "level": 本条语料在源文档中的信息粒度层级, 0代表最高级, 数字越大，信息粒度越细
            },
            ...
        ]
    }
```