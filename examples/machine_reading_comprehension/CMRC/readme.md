# 阅读理解 CMRC

## 简介

### 任务说明
本文主要介绍基于xlnet预训练模型的CMRC（Chinese Machine Reading Comprehension）数据集的阅读理解任务，给定一篇文章和一个问题，计算答案在文章中的起始位置和结束位置。

### 数据集

此任务的数据集包括以下数据集：

CMRC
- [train.json](https://github.com/ymcui/cmrc2018/blob/master/squad-style-data/cmrc2018_train.json)
- [dev.json](https://github.com/ymcui/cmrc2018/blob/master/squad-style-data/cmrc2018_dev.json)

## 快速开始

### 数据准备

为了方便开发者进行测试，我们使用了HuggingFace的数据集，需要安装HuggingFace的datasets包加载数据集

### 安装依赖

除了要安装paddlenlp及其依赖外，本示例还应安装sentencepeice，具体命令如下
```shell
pip install sentencepiece
```
另外nltk需要下载punkt资源，具体方法为：

![图片](https://user-images.githubusercontent.com/50627048/162560106-e3e1469b-5e68-4d8d-a1d7-e977e1ca242f.png)

### Fine-tune

对于 CMRC任务,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_cmrc.py \
    --model_type xlnet \
    --model_name_or_path chinese-xlnet-mid \
    --max_seq_length 1024 \
    --batch_size 6 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/cmrc/ \
    --device gpu \
    --do_train \
    --do_predict
 ```

* `model_type`: 预训练模型的种类。如xlnet。
* `model_name_or_path`: 预训练模型的具体名称。如chinese-xlnet-base,chinese-xlnet-mid等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。
* `do_train`: 是否进行训练。
* `do_predict`: 是否进行预测。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 61.26126126126126,
  "f1": 84.18154653513831,
  "total": 3219,
  "HasAns_exact": 61.26126126126126,
  "HasAns_f1": 84.18154653513831,
  "HasAns_total": 3219
}
```

**NOTE:** 如需恢复模型训练，则model_name_or_path只需指定到文件夹名即可。如`--model_name_or_path=./tmp/cmrc/model_3000/`，程序会自动加载模型参数`/model_state.pdparams`，也会自动加载词表，模型config和tokenizer的config。

### 预测

如需使用训练好的模型预测并输出结果，需将自己的数据集改成CMRC格式。

```text
{
  "data": [
    {
      "paragraphs": [
        {
          "id": "DEV_0", 
          "context": "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品", 
          "qas": [
            {
              "question": "《战国无双3》是由哪两个公司合作开发的？", 
              "id": "DEV_0_QUERY_0", 
              "answers": [
                {
                  "text": "光荣和ω-force", 
                  "answer_start": 11
                }
              ]
            }
          ]
        }
      ], 
      "id": "DEV_0", 
      "title": "战国无双3"
    }
   ]
  }
```

并参考[以内置数据集格式读取本地数据集](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html#id4)中的方法创建自己的数据集并修改`run_cmrc.py`中对应的数据集读取代码。再运行以下脚本：

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" run_cmrc.py \
    --model_type xlnet \
    --model_name_or_path chinese-xlnet-mid \
    --max_seq_length 1024 \
    --batch_size 6 \
    --learning_rate 3e-5 \
    --num_train_epochs 3 \
    --logging_steps 500 \
    --save_steps 500 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/cmrc/ \
    --device gpu \
    --do_predict
 ```

即可完成预测，预测的答案保存在`prediction.json`中。数据格式如下所示，左边的id与输入中的id对应。

```text
{
    "DEV_0_QUERY_0": "光荣和ω-force",
    ...
}
```
