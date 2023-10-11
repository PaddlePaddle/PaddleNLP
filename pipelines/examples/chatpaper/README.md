# ChatPaper

## 1. 功能概述
ChatPaper 是一个文献阅读助手，可以随时随地不厌其烦地回答您的疑问，帮助您理解文献。具体功能包含文章检索摘要和单篇论文翻译、精读、多轮问答。

## 2. 安装依赖
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/examples/chatpaper
pip install -r requirements.txt
```
## 3. 快速开始
```
python chat_paper.py \
--api_key  \
--secret_key \
--bos_ak \
--bos_sk \
--txt_file \
--retriever_api_key \
--retriever_secret_key \
--es_host \
--es_port \
--es_username \
--es_password \
--es_index_abstract \
--es_index_full_text
```
## 4. 效果展示
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/137043369/8d9cd087-5bc5-4de5-b897-9f3f4e514241" width="1000px">
</div>
