# squeezebert-paddle
paddle2.0+复现论文：SqueezeBERT: What can computer vision teach NLP about efficient neural networks?

# 提交标准
## 1. 完成模型权重从pytorch到paddle的转换代码，转换3个预训练权重
- 从https://huggingface.co/squeezebert 下载权重到models对应的目录下
- python convert_torch_to_paddle.py
- 转好的模型链接: https://pan.baidu.com/s/1Jis7In0veo4ODae5OR_FqA 提取码: p5bk

## 2. "squeezebert/squeezebert-mnli-headless"模型指标：QQP验证集accuracy=89.4
### 训练模型
```
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME="QQP"
nohup python -u ./run_glue.py \ 
--model_type squeezebert \
--model_name_or_path ./models/squeezebert-mnli-headless \
--task_name QQP --batch_size 16 \
--learning_rate 4e-5 \
--num_train_epochs 5 \
--logging_steps 10 --save_steps 2000 \
--output_dir ./tmp/QQP/ \
--device gpu \
--lr_scheduler 1 \
--seed 5
```
### **训练结果**
```
acc and f1: 0.8936136479314183, eval done total : 196.82215237617493 s
```
### 训练日志
见 **train_log.txt**
## 3. SqueezeBERT模型加速比对比BERT-Base达到4.3x
- 通过对比在pytoch和paddle框架下的加速比
- 测试数据集为qqp-dev
- 设备为1060显卡
### 不用dataloader的情况
```
# paddle跑sequeezebert，用时137s
python run_qqp_paddle.py \
 --model_path ./models/squeezebert-mnli-headless \
 --device gpu

# pytorch跑sequeezebert，用时112s
python run_qqp_torch.py \
 --model_path ./models/squeezebert-mnli-headless \
 --device gpu
 
# paddle跑bert，用时186s
python run_qqp_paddle.py \
 --model_path bert-base-uncased \
 --device gpu \
 --model_type bert

# pytorch跑bert，用时172s
python run_qqp_torch.py \
 --model_path bert-base-uncased \
 --device gpu \
 --model_type bert
```
| - |squeeze|bert|加速比|
| :----:| :----:| :----:| :----:|
|paddle|137s|186s|1.36|
|pytorch|112s|172s|1.54|

### 使用dataloader的情况
#### paddle跑squeezebert
```
export CUDA_VISIBLE_DEVICES=0
python -u ./run_glue.py \
--model_type squeezebert \
--model_name_or_path ./models/squeezebert-mnli-headless \
--task_name QQP \
--batch_size 16 \
--learning_rate 4e-5 \
--num_train_epochs 5 \
--logging_steps 10 \
--save_steps 2000 \
--output_dir ./tmp/QQP/ \
--device gpu \
--lr_scheduler 1 \
--max_steps 1
```

#### paddle跑bert
```
export CUDA_VISIBLE_DEVICES=0
python -u ./run_glue.py \
--model_type bert \
--model_name_or_path bert-base-uncased \
--task_name QQP \
--batch_size 16 \
--learning_rate 4e-5 \
--num_train_epochs 5 \
--logging_steps 10 \
--save_steps 2000 \
--output_dir ./tmp/QQP/ \
--device gpu \
--lr_scheduler 1 \
--max_steps 1
```
#### pytorch跑squeezebert
```
python run_glue_torch.py \
  --model_name_or_path squeezebert/squeezebert-mnli-headless \
  --task_name qqp \
  --do_eval \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 5 \
  --output_dir /tmp/QQP2/
```
#### pytorch跑bert
```
python run_glue_torch.py \
  --model_name_or_path bert-base-uncased \
  --task_name qqp \
  --do_eval \
  --max_seq_length 128 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 5 \
  --output_dir /tmp/QQP2/
```
**由于网络问题，无法跑pytorch版本。pytorch版本的是在colab上K80跑的，推理时间仅供参考**
| - |squeeze|bert|加速比|
| :----:| :----:| :----:| :----:|
|paddle|163s|209s|1.28|
|pytorch|463s|598s|1.29|
