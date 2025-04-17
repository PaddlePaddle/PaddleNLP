# 问答系统

问答系统(Question Answering System, QA)是信息检索系统的一种高级形式，它能用准确、简洁的自然语言回答用户用自然语言提出的问题。问答系统的应用空间十分包括，包括搜索引擎，小度音响等智能硬件，聊天机器人，以及政府、金融、银行、电信、电商领域的智能客服等。

在问答系统中，检索式问答系统是最容易落地的一种，它具有速度快、可控性好、容易拓展等特点。
检索式问答系统是一种基于问题答案对进行检索匹配的系统，根据是否需要 FAQ（Frequently asked questions）可以进一步分为有监督检索式问答系统和无监督检索式问答系统，前者需要用户提供 FAQ 语料，后者不需要预备问答语料，可通过问题答案对生成的方式自动生成语料。

PaddleNLP 提供了[无监督检索式问答系统](./unsupervised_qa)，开发者可根据实际情况进行选择。

关于问答场景应用案例请查阅飞桨新产品[RocketQA](https://github.com/PaddlePaddle/RocketQA)。


**无监督检索式问答系统效果展示**：
<div align="center">
    <img src="https://user-images.githubusercontent.com/20476674/199488926-c64d3f4e-8117-475f-afe6-b02088105d09.gif">
</div>
