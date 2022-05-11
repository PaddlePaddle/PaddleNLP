## PaddleNLP智能文本产线范例

Pipelines是基于PaddleNLP搭建的NLP重点应用场景产线范例库，依托百度NLP领先语义理解技术与飞桨PaddleNLP模型库，提供易用的端到端的场景示例，为开发者提无门槛搭建NLP重点应用场景系统方案的能力。

## 特点

- **开源最佳的中文效果**：基于最新文心大模型ERNIE 3.0，为NLP下游任务效果保驾护航；结合PaddleNLP训练、压缩、推理的全流程文本加速技术，提供高效经济的预训练模型落部署方案。

- **丰富的产业应用范例**：围绕主流NLP应用场景，提供多套产业应用范例、如语义检索系统、文档视觉问答、语音指令信息抽取、评论观点抽取等，为开发者提供文心大模型的最佳落地实践。

- **灵活可扩展的NLP流水线**：基于Pipeline的设计理念，针对NLP业务落地痛点，提供流水线的模块化的抽象，可快速完成模块的复用与拓展，覆盖从数据标注、数据清晰、模型优化、推理部署、可视化调优的NLP业务全生命周期覆盖。

## 安装

```script
# 本地编译生成 pipelines package
python setup.py bdist_wheel
# 本地安装 pipelines package
pip install ./dist/pipelines-0.1.0a0-py3-none-any.whl
```


## 快速开始

### 一键体验智能问答应用效果
```python
python ./examples/question-answering/dense_qa_example.py
```
我们针对中国重点城市的基本信息搭建了百科知识问答小系统，运行上述代码体验效果如下:
![](https://tianxin1860.github.io/img/question_answering.png)

### 讨论组
- [文心大模型官方主页](https://wenxin.baidu.com/)
- [Github Issues](https://github.com/PaddlePaddle/PaddleNLP/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ 群: 758287592 (飞桨NLP技术交流群).
- 微信群: 758287592 (飞桨NLP技术交流群).
