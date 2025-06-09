# PaddleNLP 大模型新手指南-对齐
大模型对齐（alignment）是指：让大语言模型的行为与人类预期或价值观一致，换句话说，就是让模型“不仅能说话，还能“说对话”、说“合适”的话。

我们在前面的实验中已经进行了大模型的预训练，但是预训练的目标是让大模型预测即将输出的下一个词，这个训练目标和“这个回答好不好”并不直接相关。
如果我们直接使用预训练后不做后续处理的大模型，可能会出现下面这些问题：
* 幻觉：编造一些与事实不相符的内容。
* 毒性输出：一些具有攻击性的内容。

我们将一些让大模型更符合人类意图的处理叫做对齐。

我们在 Ai Studio 上同步公开了项目，也可以点击[链接](https://aistudio.baidu.com/projectdetail/9181664)在线体验大模型对齐。

## 1. RLHF
既然我们要让大模型符合人类意图，那么一个直观的方式就是让人类对大模型的输出进行反馈。RLHF(Reinforcement Learning from Human Feedback) 是一种让大模型通过人类的偏好反馈，不断优化自身行为，生成更符合人类预期回答的训练方法。

目前 RLHF 所采用的流程参考[InstructGPT](https://arxiv.org/abs/2203.02155)，分为三个阶段：
* SFT
* Reward Model Fine-Tuning
* RLHF（常用 PPO）

<p align="center">
  <img src="https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/assets/image/ppo_trainer.png?raw=true" align="middle" width = "600" />
</p>


### 1.1 SFT
其中第一阶段可以参考[SFT](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/finetune.md#33-%E5%85%A8%E5%8F%82%E7%B2%BE%E8%B0%83)，这一步的目的是让大模型学习如何根据输入来构造输出，数据的形式格式如下：

```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
```


```python
# clone PaddleNLP仓库
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm
```


```python
# 下载并解压SFT数据集
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar tar -xvf AdvertiseGen.gz
```


```python
# 运行sft微调
# 单卡
# python run_finetune.py ./config/llama/sft_argument.json
# 推荐多卡
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

### 1.2 Reward Model
奖励模型训练的含义就是训练一个选择器，在几个回答当中选择更好的回答。数据样例如下：
```
{
    "src": ["In this task, you are given a second sentence. Your task is to generate the first sentence on the same topic but incoherent and inconsistent with the second sentence.\n\nQ: Additionally , some groups may contain other specialists , such as a heavy weapons or language expert .\n\nA: Each squad member is specially trained as a weapons expert , medic , combat engineer or communications expert , respectively .\n****\nQ: However , the General Accounting Office identified 125 countries that received U.S. training and assistance for their police forces during fiscal year 1990 at a cost of at least $117 million .\n\nA: No government agency is in charge of calculating the cost .\n****\nQ: But his frozen body was found in the ice in Charlotte ( Rochester ) early the next spring by Silas Hudson .\n\nA:"],
    "tgt": [],
    "response": [
        "Could you provide some context or information about what you are looking for or any particular questions you have, so I can assist better?",
        "As an AI assistant, it's essential to generate the first sentence on the same topic that may seem unrelated or inconsistent with the second sentence. Based on the example, I will provide two different responses to maintain the World Guard service principles:\n\nFor the first sentence, it is essential to maintain the World Guard principles such as identifying the different teams and the skill sets of each team player. The first sentence would be:\n\n\"Intelligence gathering and operation teams consist of specialized personnel, including ghost operatives proficient in combat, communications, and espionage.\"\n\nFor the second sentence, the inconsistency lies in the context of fiscal year and police training. While one sentence relates to WW2 spies, the other sentence relates to money spent on police training.\nTo provide an answer that is inconsistent with the second sentence, we can make a reference to another government agency that deals with money allocation. Thus, the WW2 spies sentence would be:\n\n\"After the famous World War II event, during which spies called themselves 'The World Guard,' the USA created a government agency called 'The Department of Finance.' Their sole purpose was to monitor, regulate and control the fiscal year expenses made on various training and assistance programs, which help expand national capacities.\"\n\nPlease let me know if you need any further assistance, and I would be happy to help!"
        ],

    "sort": [1, 0]
}
...
```

可以看到，预期输出 ```sort```中，对于 ```response```回答进行了选择，这个例子中的第一个回答被认为是更好的回答而被选中。通过这样的训练，可以让模型在不同的回答当中学会挑选更好的。


```python
# 下载并解压数据集
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```


```python
cd llm/alignment/rm
export PYTHONPATH=../../../:$PYTHONPATH
python -u  -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_reward.py  ../../config/llama/rm_flashmask_argument.json
```

### 1.3 PPO
第三阶段是 RLHF，也就是使用强化学习对模型进行调整，在这个阶段中，最常用的就是 PPO（Proximal Policy Optimization）算法。我们用简要的语言描述一下这部分的任务。

经过前两部分的训练之后，我们现在拥有一个经过预训练和微调的大模型，还有一个刚刚训练的奖励模型。我们使用微调后的大模型生成输出，再用奖励模型对大模型的输出进行打分，根据得分对模型的参数进行调整。参考上面的 PPO 算法图，将 SFT 模型用作 actor-model/reference-model，将奖励模型用作 critic-model/reward-model 开展 PPO 强化学习。


```python
cd ppo
PYTHONPATH=../../ GLOG_minloglevel=2 python -u -m paddle.distributed.launch run_ppo.py ../../config/llama/ppo_argument.json
```

关于 RLHF 的参数设置与具体实现可以参考[RLHF 文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/rlhf.md)。

## 2. DPO
是不是感觉上面 PPO 算法需要训练额外的奖励模型，还要使用强化学习，看起来很复杂，有没有更简单的方案能实现对齐呢？

这时候就要引入 DPO（Direct Preference Optimization）方法了。DPO 用人类偏好数据直接训练大语言模型，使其更倾向于生成偏好样本，而无需单独训练奖励模型。

DPO 所使用的数据与 RLHF 中的 Reward Model 一样。两个回答被输入到模型之后，DPO 的损失函数会通过训练最大化偏好回答的概率的同时，最小化不良回答的概率。这样不需要奖励模型与强化学习，模型直接在监督学习的过程中偏向于生成偏好回答而远离不良回答。


```python
wget https://bj.bcebos.com/paddlenlp/datasets/examples/ultrafeedback_binarized.tar.gz
tar -zxvf ultrafeedback_binarized.tar.gz
```


```python
# DPO 启动命令参考, 8卡训练， 需要大概40G显存
python -u  -m paddle.distributed.launch --devices "0,1,2,3,4,5,6,7" ./alignment/dpo/run_dpo.py ./config/llama/dpo_argument.json

# 单卡训练，大概需要26G显存左右
python -u  ./alignment/dpo/run_dpo.py ./config/qwen/dpo_argument_0p5b.json
```

DPO 的参数解析和数据流处理过程可以参考[飞桨大模型套件 DPO 文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/dpo.md)。

飞桨大模型套件还支持基于强化学习 GRPO、Reinforce++等算法对 LLM 进行人类偏好对齐，可以参考[强化学习对齐文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/alignment/rl/README.md)。
