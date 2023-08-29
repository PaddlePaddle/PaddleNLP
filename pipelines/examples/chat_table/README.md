# ChatTable

## 1. 功能概述

ChatTable 是一个表格问答助手，他可以根据您给出的查询问题，快速定位表格，并返回查询结果。目前单表查询的性能较优。

## 2. 安装依赖

```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/examples/chat_table
```

## 3. 快速开始

根据表格数据，建立索引
```
python create_index.py \
--dirname ...
```
开始ChatTable
```
python chat_table_web.py \
--api_key ... \
--secret_key ...
```

## 4. 效果展示

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/137043369/794270c1-3c3e-4e4a-aa1d-8a48206c70b4" width="1000px">
</div>
