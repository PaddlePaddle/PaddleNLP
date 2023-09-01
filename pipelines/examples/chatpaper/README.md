# ChatPaper

## 1. 功能概述
<<<<<<< HEAD

ChatPaper 是一个文献阅读助手，可以随时随地不厌其烦地回答您的疑问，帮助您理解文献。具体功能包含论文【单篇/多篇】翻译、精读、多轮问答以及基于本地知识库的多轮问答。目前对英文论文仅提供翻译和精读功能，中文论文【单篇/多篇】提供精读、多轮问答以及学术检索功能。

## 2. 安装依赖

=======
ChatPaper 是一个文献阅读助手，可以随时随地不厌其烦地回答您的疑问，帮助您理解文献。具体功能包含论文【单篇/多篇】翻译、精读、多轮问答以及基于本地知识库的多轮问答。目前对英文论文仅提供翻译和精读功能，中文论文【单篇/多篇】提供精读、多轮问答以及学术检索功能。
## 2. 安装依赖
>>>>>>> 886218498232006ef738a937d5fe963355c29d35
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/examples/chatpaper
pip install -r requirements.txt
```
<<<<<<< HEAD

## 3. 快速开始

=======
## 2. 快速开始
>>>>>>> 886218498232006ef738a937d5fe963355c29d35
```
python chat_paper.py \
--api_key  \
--secret_key \
--bos_ak \
--bos_sk \
--json_dir \
--retriever_api_key \
--retriever_secret_key \
--es_host \
--es_port \
--es_username \
--es_password \
--es_index_abstract \
--es_index_full_text
```
<<<<<<< HEAD

## 4. 效果展示

=======
## 3. 效果展示
>>>>>>> 886218498232006ef738a937d5fe963355c29d35
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/137043369/fcce60b8-993c-45f8-8892-19cd8bd9b906" width="1000px">
</div>
