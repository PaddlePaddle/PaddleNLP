# ChatPaper
## 1. 功能概述
ChatPaper 是一个文献阅读助手，可以随时随地不厌其烦地回答您的疑问，帮助您理解文献。具体功能包含论文【单篇/多篇】翻译、精读、多轮问答以及基于本地知识库的多轮问答。目前对英文论文仅提供翻译和精读功能，中文论文【单篇/多篇】提供精读、多轮问答以及学术检索功能。
## 2. 安装依赖
```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines/examples/chatpaper
pip install -r requirements.txt
```
## 2. 快速开始
如果要实现学术检索功能，首先需要建立本地知识库
建立本地知识库（中文pdf）
```
python create_index.py \
--api_key ... \
--secret_key ... \
--dirname ...
```
开始chatpaper
```
python chat_paper.py \
--api_key ... \
--secret_key ...
```
【注意】如果要进行英文论文的翻译和精读，请在前端页面 切换输入论文类别为English
## 3. 效果展示
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleNLP/assets/137043369/fcce60b8-993c-45f8-8892-19cd8bd9b906" width="1000px">
</div>
