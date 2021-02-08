# 对话自动评估模块ADE

## 目录
- [**模型简介**](#模型简介)

- [**快速开始**](#快速开始)

- [**进阶使用**](#进阶使用)

- [**参考论文**](#参考论文)

- [**版本更新**](#版本更新)

## 模型简介

&ensp;&ensp;&ensp;&ensp;对话自动评估（Auto Dialogue Evaluation）评估开放领域对话系统的回复质量，能够帮助企业或个人快速评估对话系统的回复质量，减少人工评估成本。

&ensp;&ensp;&ensp;&ensp;1. 在无标注数据的情况下，利用负采样训练匹配模型作为评估工具，实现对多个对话系统回复质量排序；

&ensp;&ensp;&ensp;&ensp;2. 利用少量标注数据（特定对话系统或场景的人工打分），在匹配模型基础上进行微调，可以显著提高该对话系统或场景的评估效果。

同时推荐用户参考[ IPython Notebook demo](https://aistudio.baidu.com/aistudio/projectDetail/122301)

## 快速开始

### 安装说明

#### &ensp;&ensp;a、环境依赖
- Python >= 2.7
- cuda >= 9.0
- cudnn >= 7.0
- pandas >= 0.20.1
- PaddlePaddle >= 1.8.0，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装, 本模块使用bert作为pretrain model进行模型的finetuning训练，训练速度较慢，建议安装GPU版本的PaddlePaddle

#### &ensp;&ensp;b、下载代码

&ensp;&ensp;&ensp;&ensp;克隆数据集代码库到本地

```
git clone https://github.com/PaddlePaddle/models.git
cd models/PaddleNLP/dialogue_system/auto_dialogue_evaluation
```

### 任务简介

&ensp;&ensp;&ensp;&ensp;本模块内模型训练主要包括两个阶段：

&ensp;&ensp;&ensp;&ensp;1）第一阶段：训练一个匹配模型作为评估工具，可用于待评估对话系统内的回复内容进行排序；（matching任务)

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;模型结构: 输入为context和response, 对两个输入学习embedding表示, 学习到的表示经过lstm学习高阶表示, context和response的高阶表示计算双线性张量积logits, logits和label计算sigmoid_cross_entropy_with_logits loss;

&ensp;&ensp;&ensp;&ensp;2）第二阶段：利用少量的对话系统的标记数据，对第一阶段训练的匹配模型进行finetuning, 可以提高评估效果（包含human，keywords，seq2seq_att，seq2seq_naive，4个finetuning任务）;

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;模型结构: finetuning阶段学习表示到计算logits部分和第一阶段模型结构相同，区别在于finetuning阶段计算square_error_cost loss；

&ensp;&ensp;&ensp;&ensp;用于第二阶段fine-tuning的对话系统包括下面四部分：

```
human: 人工模拟的对话系统；
keywords：seq2seq keywords对话系统；
seq2seq_att：seq2seq attention model 对话系统；
seq2seq_naive：naive seq2seq model对话系统；
```

注意: 目前ade模块内提供的训练好的官方模型及效果, 均是在GPU单卡上面训练和预测得到的, 用户如需复线效果, 可使用单卡相同的配置.

### 数据准备
&ensp;&ensp;&ensp;&ensp;数据集说明：本模块内只提供训练方法，真实涉及的匹配数据及4个对话系统的数据只开源测试集数据，仅提供样例，用户如有自动化评估对话系统的需求，可自行准备业务数据集按照文档提供流程进行训练；

```
unlabel_data（第一阶段训练匹配数据集）

label_data（第二阶段finetuning数据集）
1、human: 人工对话系统产出的标注数据；
2、keywords：关键词对话系统产出的标注数据；
3、seq2seq_att：seq2seq attention model产出的标注对话数据；
4、seq2seq_naive：传统seq2seq model产出的标注对话数据；
```

&ensp;&ensp;&ensp;&ensp;数据集、相关模型下载
&ensp;&ensp;&ensp;&ensp;linux环境下：

```
python ade/prepare_data_and_model.py
```
&ensp;&ensp;&ensp;&ensp;数据路径：data/input/data

&ensp;&ensp;&ensp;&ensp;模型路径：data/saved_models/trained_models

&ensp;&ensp;&ensp;&ensp;windows环境下：
```
python ade\prepare_data_and_model.py
```
&ensp;&ensp;&ensp;&ensp;数据路径：data\input\data

&ensp;&ensp;&ensp;&ensp;模型路径：data\saved_models\trained_models


&ensp;&ensp;&ensp;&ensp;下载经过预处理的数据，运行该脚本之后，data目录下会存在unlabel_data(train.ids/val.ids/test.ids)，lable_data: human、keywords、seq2seq_att、seq2seq_naive(四个任务数据train.ids/val.ids/test.ids)，以及word2ids.

### 模型配置

&ensp;&ensp;&ensp;&ensp;配置文件路径: data/config/ade.yaml

```
loss_type: loss类型, 可选CLS或者L2
training_file: 训练数据路径
val_file: 验证集路径
predict_file: 预测文件路径
print_steps: 每隔print_steps个步数打印一次日志
save_steps: 每隔save_steps个步数来保存一次模型
num_scan_data:
word_emb_init: 用于初始化embedding的词表路径
init_model: 初始化模型路径
use_cuda: 是否使用cuda, 如果是gpu训练时，设置成true
batch_size: 一个batch内输入的样本个数
hidden_size: 隐层大小
emb_size: embedding层大小
vocab_size: 词表大小
sample_pro: 采样比率
output_prediction_file: 输出的预测文件
init_from_params: 训练好的模型参数文件，一般用于预测
init_from_pretrain_model: 预训练模型路径，如bert的模型参数
inference_model_dir: inference model的保存路径
save_model_path: 训练产出模型的输出路径
evaluation_file: 参与评估的inference 文件
vocab_path: 词表路径
max_seq_len: 输入最大序列长度
random_seed: 随机种子设置
do_save_inference_model: 是否保存inference model
encable_ce: 是否开启ce
```

### 单机训练

#### 1、第一阶段matching模型的训练：
#### linux环境下：

#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本训练

```
bash run.sh matching train
```

&ensp;&ensp;&ensp;&ensp;如果为CPU训练:

```
请将run.sh内参数设置为:
1、export CUDA_VISIBLE_DEVICES=
```

&ensp;&ensp;&ensp;&ensp;如果为GPU训练:

```
请将run.sh内参数设置为:
1、如果为单卡训练（用户指定空闲的单卡）：
export CUDA_VISIBLE_DEVICES=0
2、如果为多卡训练（用户指定空闲的多张卡）：
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行训练相关的代码:

```
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1  #开启显存优化

export CUDA_VISIBLE_DEVICES=0  #GPU单卡训练
#export CUDA_VISIBLE_DEVICES=0,1,2,3  #GPU多卡训练

#export CUDA_VISIBLE_DEVICES=  #CPU训练
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

pretrain_model_path="data/saved_models/matching_pretrained"

if [ -f ${pretrain_model_path} ]
then
    rm ${pretrain_model_path}
fi

if [ ! -d ${pretrain_model_path} ]
then
     mkdir ${pretrain_model_path}
fi

python -u main.py \
      --do_train=true \
      --use_cuda=${use_cuda} \
      --loss_type="CLS" \
      --max_seq_len=50 \
      --save_model_path="data/saved_models/matching_pretrained" \
      --training_file="data/input/data/unlabel_data/train.ids" \
      --epoch=20 \
      --print_step=1 \
      --save_step=400 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016 \
      --learning_rate=0.001 \
      --sample_pro=0.1
```

注意: 用户进行模型训练、预测、评估等, 可通过修改data/config/ade.yaml配置文件或者从命令行传入来进行参数配置, 优先推荐命令行参数传入;

#### windows环境下：
训练：
```
python -u main.py --do_train=true --use_cuda=false --loss_type=CLS --max_seq_len=50 --save_model_path=data\saved_models\matching_pretrained --training_file=data\input\data\unlabel_data\train.ids --epoch=20 --print_step=1 --save_step=400 --batch_size=256 --hidden_size=256 --emb_size=256 --vocab_size=484016 --learning_rate=0.001 --sample_pro=0.1
```

#### 2、第二阶段finetuning模型的训练：
#### linux环境下：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本训练

```
bash run.sh task_name task_type
参数说明：
task_name: seq2seq_naive、seq2seq_att、keywords、human，选择4个任务中任意一项；
task_type: train、predict、evaluate、inference, 选择4个参数选项中任意一项(train: 只执行训练，predict: 只执行预测，evaluate：只执行评估过程，依赖预测的结果，inference: 保存inference model;

训练示例： bash run.sh human train
```

&ensp;&ensp;&ensp;&ensp;CPU和GPU使用方式如单机训练1中所示；

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行训练相关的代码:

```
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1  #开启显存优化

export CUDA_VISIBLE_DEVICES=0  #GPU单卡训练
#export CUDA_VISIBLE_DEVICES=0,1,2,3  #GPU多卡训练

#export CUDA_VISIBLE_DEVICES=  #CPU训练
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

save_model_path="data/saved_models/human_finetuned"

if [ -f ${save_model_path} ]
then
    rm ${save_model_path}
fi

if [ ! -d ${save_model_path} ]
then
    mkdir ${save_model_path}
fi

python -u main.py \
      --do_train=true \
      --use_cuda=${use_cuda} \
      --loss_type="L2" \
      --max_seq_len=50 \
      --init_from_pretrain_model="data/saved_models/trained_models/matching_pretrained/params/params" \
      --save_model_path="data/saved_models/human_finetuned" \
      --training_file="data/input/data/label_data/human/train.ids" \
      --epoch=50 \
      --print_step=1 \
      --save_step=400 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016 \
      --learning_rate=0.001 \
      --sample_pro=0.1
```

#### windows环境下：
```
python -u main.py --do_train=true --use_cuda=false --loss_type=L2 --max_seq_len=50 --save_model_path=data\saved_models\human_finetuned --training_file=data\input\data\label_data\human\train.ids --epoch=50 --print_step=1 --save_step=400 --batch_size=256 --hidden_size=256 --emb_size=256 --vocab_size=484016 --learning_rate=0.001 --sample_pro=0.1
```

### 模型预测
#### 1、第一阶段matching模型的预测：
#### linux环境下：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本预测

```
bash run.sh matching predict
```

&ensp;&ensp;&ensp;&ensp;如果为CPU预测:

```
请将run.sh内参数设置为:
export CUDA_VISIBLE_DEVICES=
```

&ensp;&ensp;&ensp;&ensp;如果为GPU预测:

```
请将run.sh内参数设置为:
单卡预测：
export CUDA_VISIBLE_DEVICES=0 #用户可自行指定空闲的卡
```
注：预测时，如采用方式一，用户可通过修改run.sh中init_from_params参数来指定自己需要预测的模型，目前代码中默认预测本模块提供的训练好的模型；

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行预测相关的代码:

```
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1  #开启显存优化

export CUDA_VISIBLE_DEVICES=0  #单卡预测
#export CUDA_VISIBLE_DEVICES=  #CPU预测
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

python -u main.py \
      --do_predict=true \
      --use_cuda=${use_cuda} \
      --predict_file="data/input/data/unlabel_data/test.ids" \
      --init_from_params="data/saved_models/trained_models/matching_pretrained/params" \
      --loss_type="CLS" \
      --output_prediction_file="data/output/pretrain_matching_predict" \
      --max_seq_len=50 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016
```

注：采用方式二时，模型预测过程可参考run.sh内具体任务的参数设置

#### windows环境下：
预测：
```
python -u main.py --do_predict=true --use_cuda=false --predict_file=data\input\data\unlabel_data\test.ids --init_from_params=data\saved_models\trained_models\matching_pretrained\params --loss_type=CLS --output_prediction_file=data\output\pretrain_matching_predict --max_seq_len=50 --batch_size=256 --hidden_size=256 --emb_size=256 --vocab_size=484016
```

#### 2、第二阶段finetuning模型的预测：

#### linux环境下：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本预测

```
bash run.sh task_name task_type
参数说明：
task_name: seq2seq_naive、seq2seq_att、keywords、human，选择4个任务中任意一项；
task_type: train、predict、evaluate、inference, 选择4个参数选项中任意一项(train: 只执行训练，predict: 只执行预测，evaluate：只执行评估过程，依赖预测的结果，inference: 保存inference model;

预测示例： bash run.sh human predict
```

&ensp;&ensp;&ensp;&ensp;指定CPU或者GPU方法同上模型预测1中所示；

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行预测相关的代码:

```
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1  #开启显存优化

export CUDA_VISIBLE_DEVICES=0  #单卡预测
#export CUDA_VISIBLE_DEVICES=  #CPU预测
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

python -u main.py \
      --do_predict=true \
      --use_cuda=${use_cuda} \
      --predict_file="data/input/data/label_data/human/test.ids" \
      --init_from_params="data/saved_models/trained_models/human_finetuned/params" \
      --loss_type="L2" \
      --output_prediction_file="data/output/finetuning_human_predict" \
      --max_seq_len=50 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016
```

#### windows环境下：
```
python -u main.py --do_predict=true --use_cuda=false --predict_file=data\input\data\label_data\human\test.ids --init_from_params=data\saved_models\trained_models\human_finetuned\params --loss_type=L2 --output_prediction_file=data\output\finetuning_human_predict --max_seq_len=50 --batch_size=256 --hidden_size=256 --emb_size=256 --vocab_size=484016
```

### 模型评估

&ensp;&ensp;&ensp;&ensp;模块中5个任务，各任务支持计算的评估指标内容如下：

```
第一阶段：
matching: 使用R1@2, R1@10, R2@10, R5@10四个指标进行评估排序模型的效果；

第二阶段：
human: 使用spearman相关系数来衡量评估模型对系统的打分与实际对话系统打分之间的关系；
keywords：使用spearman相关系数来衡量评估模型对系统的打分与实际对话系统打分之间的关系；
seq2seq_att：使用spearman相关系数来衡量评估模型对系统的打分与实际对话系统打分之间的关系；
seq2seq_naive：使用spearman相关系数来衡量评估模型对系统的打分与实际对话系统打分之间的关系；
```

&ensp;&ensp;&ensp;&ensp;效果上，以四个不同的对话系统（seq2seq\_naive／seq2seq\_att／keywords／human）为例，使用对话自动评估工具进行自动评估。

&ensp;&ensp;&ensp;&ensp;1. 无标注数据情况下，直接使用预训练好的评估工具进行评估；

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;在四个对话系统上，自动评估打分和人工评估打分spearman相关系数，如下：

   ||seq2seq\_naive|seq2seq\_att|keywords|human|
   |--|:--:|--:|:--:|--:|
   |cor|0.361|0.343|0.324|0.288|

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;对四个系统平均得分排序：

   |人工评估|k(0.591)<n(0.847)<a(1.116)<h(1.240)|
   |--|--:|
   |自动评估|k(0.625)<n(0.909)<a(1.399)<h(1.683)|

&ensp;&ensp;&ensp;&ensp;2. 利用少量标注数据微调后，自动评估打分和人工打分spearman相关系数，如下：

   ||seq2seq\_naive|seq2seq\_att|keywords|human|
   |--|:--:|--:|:--:|--:|
   |cor|0.474|0.477|0.443|0.378|

#### linux环境下：
#### 1、第一阶段matching模型的评估：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本评估

```
bash run.sh matching evaluate
```

注：评估计算ground_truth和predict_label之间的打分，默认CPU计算即可；

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行评估相关的代码:

```
export CUDA_VISIBLE_DEVICES=  #指默认CPU评估
export CPU_NUM=1 #CPU训练时指定CPU number

python -u main.py \
      --do_eval=true \
      --use_cuda=false \
      --evaluation_file="data/input/data/unlabel_data/test.ids" \
      --output_prediction_file="data/output/pretrain_matching_predict" \
      --loss_type="CLS"
```
#### windows环境下：
评估：
```
python -u main.py --do_eval=true --use_cuda=false --evaluation_file=data\input\data\unlabel_data\test.ids --output_prediction_file=data\output\pretrain_matching_predict --loss_type=CLS
```

#### 2、第二阶段finetuning模型的评估：
#### linux环境下：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本评估

```
bash run.sh task_name task_type
参数说明：
task_name: seq2seq_naive、seq2seq_att、keywords、human，选择4个任务中任意一项；
task_type: train、predict、evaluate、inference, 选择4个参数选项中任意一项(train: 只执行训练，predict: 只执行预测，evaluate：只执行评估过程，依赖预测的结果，inference: 保存inference model;

评估示例： bash run.sh human evaluate
```

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行评估相关的代码:

```
export CUDA_VISIBLE_DEVICES=  #指默认CPU评估
export CPU_NUM=1 #CPU训练时指定CPU number

python -u main.py \
      --do_eval=true \
      --use_cuda=false \
      --evaluation_file="data/input/data/label_data/human/test.ids" \
      --output_prediction_file="data/output/finetuning_human_predict" \
      --loss_type="L2"
```

#### windows环境下：
```
python -u main.py --do_eval=true --use_cuda=false --evaluation_file=data\input\data\label_data\human\test.ids --output_prediction_file=data\output\finetuning_human_predict --loss_type=L2
```

### 模型推断
#### 1、第一阶段matching模型的推断：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本保存inference model

```
bash run.sh matching inference
```

&ensp;&ensp;&ensp;&ensp;如果为CPU执行inference model过程:

```
请将run.sh内参数设置为:
export CUDA_VISIBLE_DEVICES=
```

&ensp;&ensp;&ensp;&ensp;如果为GPU执行inference model过程:

```
请将run.sh内参数设置为:
单卡推断（用户指定空闲的单卡）：
export CUDA_VISIBLE_DEVICES=0
```

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行inference model相关的代码:

```
export CUDA_VISIBLE_DEVICES=0  # 指GPU单卡推断
#export CUDA_VISIBLE_DEVICES=  #CPU推断
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

python -u main.py \
      --do_save_inference_model=true \
      --use_cuda=${use_cuda} \
      --init_from_params="data/saved_models/trained_models/matching_pretrained/params" \
      --inference_model_dir="data/inference_models/matching_inference_model"
```

#### 2、第二阶段finetuning模型的推断：
#### &ensp;&ensp;&ensp;&ensp;方式一: 推荐直接使用模块内脚本保存inference model

```
bash run.sh task_name task_type
参数说明：
task_name: seq2seq_naive、seq2seq_att、keywords、human，选择4个任务中任意一项；
task_type: train、predict、evaluate、inference, 选择4个参数选项中任意一项(train: 只执行训练，predict: 只执行预测，evaluate：只执行评估过程，依赖预测的结果，inference: 保存inference model;

评估示例： bash run.sh human inference
```

&ensp;&ensp;&ensp;&ensp;CPU和GPU指定方式同模型推断1中所示；

#### &ensp;&ensp;&ensp;&ensp;方式二: 执行inference model相关的代码:

```
export CUDA_VISIBLE_DEVICES=0  # 指GPU单卡推断
#export CUDA_VISIBLE_DEVICES=  #CPU推断
#export CPU_NUM=1 #CPU训练时指定CPU number

if  [ ! "$CUDA_VISIBLE_DEVICES" ]
then
    use_cuda=false
else
    use_cuda=true
fi

python -u main.py \
      --do_save_inference_model=true \
      --use_cuda=${use_cuda} \
      --init_from_params="data/saved_models/trained_models/human_finetuned/params" \
      --inference_model_dir="data/inference_models/human_inference_model"
```

### 服务部署
&ensp;&ensp;&ensp;&ensp;模块内提供已训练好的5个inference_model模型，用户可根据自身业务情况进行下载使用。

#### 服务器部署
&ensp;&ensp;&ensp;&ensp;请参考PaddlePaddle官方提供的[服务器端部署](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/deploy/inference/index_cn.html)文档进行部署上线。

## 进阶使用

### 背景介绍
&ensp;&ensp;&ensp;&ensp;对话自动评估任务输入是文本对（上文，回复），输出是回复质量得分，匹配任务（预测上下文是否匹配）和自动评估任务有天然的联系，该项目利用匹配任务作为自动评估的预训练，利用少量标注数据，在匹配模型基础上微调。

### 模型概览

&ensp;&ensp;&ensp;&ensp;本模块内提供的模型为：

&ensp;&ensp;&ensp;&ensp;1）匹配模型：context和response作为输入，使用lstm学习两个句子的表示，在计算两个线性张量的积作为logits，然后sigmoid_cross_entropy_with_logits作为loss, 最终用来评估相似程度;

&ensp;&ensp;&ensp;&ensp;2）finetuing模型：在匹配模型的基础上，将sigmoid_cross_entropy_with_logits loss优化成平方损失loss，来进行训练；

&ensp;&ensp;&ensp;&ensp;模型中所需数据格式如下：

&ensp;&ensp;&ensp;&ensp;训练、预测、评估使用的数据示例如下，数据由三列组成，以制表符（'\t'）分隔，第一列是以空格分开的上文id（即context），第二列是以空格分开的回复id（即response），第三列是标签（标签含义：2-完全匹配，1-部分匹配，0-不匹配）。

```
723 236 7823 12 8     887 13 77 4       2
8474 13 44 34         2 87 91 23       0
```

## 参考论文

- Anjuli Kannan and Oriol Vinyals. 2017. Adversarial evaluation of dialogue models. arXiv preprint arXiv:1701.08198.
- Ryan Lowe, Michael Noseworthy, Iulian V Serban, Nicolas Angelard-Gontier, Yoshua Bengio, and Joelle Pineau. 2017. Towards an automatic turing test: Learning to evaluate dialogue responses. arXiv preprint arXiv:1708.07149.
- Sebastian M¨oller, Roman Englert, Klaus Engelbrecht, Verena Hafner, Anthony Jameson, Antti Oulasvirta, Alexander Raake, and Norbert Reithinger. 2006. Memo: towards automatic usability evaluation of spoken dialogue services by user error simulations. In Ninth International Conference on Spoken Language Processing.
- Kishore Papineni, Salim Roukos, ToddWard, andWei-Jing Zhu. 2002. Bleu: a method for automatic evaluation
of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics, pages 311–318. Association for Computational Linguistics.
- Chongyang Tao, Lili Mou, Dongyan Zhao, and Rui Yan. 2017. Ruber: An unsupervised method for automatic evaluation of open-domain dialog systems. arXiv preprint arXiv:1701.03079.
- Marilyn AWalker, Diane J Litman, Candace A Kamm, and Alicia Abella. 1997. Paradise: A framework for evaluating spoken dialogue agents. In Proceedings of the eighth conference on European chapter of the Association for Computational Linguistics, pages 271–280. Association for Computational Linguistics.
- Zhao Yan, Nan Duan, Junwei Bao, Peng Chen, Ming Zhou, Zhoujun Li, and Jianshe Zhou. 2016. Docchat: An information retrieval approach for chatbot engines using unstructured documents. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), volume 1, pages 516–525.
- Chia-Wei Liu, Ryan Lowe, Iulian V Serban, Michael Noseworthy, Laurent Charlin, and Joelle Pineau. 2016. How not to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation. arXiv preprint arXiv:1603.08023.
- Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. Text Summarization Branches Out.

## 版本更新

第一版：PaddlePaddle 1.4.0版本
主要功能：支持4个不同对话系统数据上训练、预测和系统性能评估

第二版：PaddlePaddle 1.6.0版本
更新功能：在第一版的基础上，根据PaddlePaddle的模型规范化标准，对模块内训练、预测、评估等代码进行了重构，提高易用性；

## 如何贡献代码

&ensp;&ensp;&ensp;&ensp;如果你可以修复某个issue或者增加一个新功能，欢迎给我们提交PR。如果对应的PR被接受了，我们将根据贡献的质量和难度进行打分（0-5分，越高越好）。如果你累计获得了10分，可以联系我们获得面试机会或者为你写推荐信。
