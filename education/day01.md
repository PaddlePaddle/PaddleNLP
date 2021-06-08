# Day01 词向量作业辅导

本教程旨在辅导同学如何完成 AI Studio课程——[『NLP打卡营』实践课1：词向量应用展示
](https://aistudio.baidu.com/aistudio/projectdetail/1535355)课后作业。

## 1. 选择词向量预训练模型

在[PaddleNLP 中文Embedding模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/embeddings.md#%E4%B8%AD%E6%96%87%E8%AF%8D%E5%90%91%E9%87%8F)查询PaddleNLP所支持的中文预训练模型。选择其中一个模型，如**中文维基百科**语料中的w2v.wiki.target.word-word.dim300。

## 2. 更换TokenEmbedding预训练模型

![image](https://user-images.githubusercontent.com/10826371/121013730-cad24b00-c7cb-11eb-8d9a-4a8b644b684f.png)
使用新模型（如w2v.wiki.target.word-word.dim300）替换红色框中的模型名字，并运行该cell。


## 3. 查看新模型下的可视化结果

查看词向量可视化结果

![image](https://user-images.githubusercontent.com/10826371/121014101-2b618800-c7cc-11eb-8a5b-e8a8ac8d473c.png)

执行图中所示的代码cell。通过查看**启动VisualDL查看词向量降维效果**所在cell，观察新模型下词向量可视化结果。

## 4. 计算句对语义相似度

按顺序依次执行**基于TokenEmbedding的词袋模型**，**构造Tokenizer**，**相似句对数据读取**，**查看相似语句相关度** 所示代码cell。
