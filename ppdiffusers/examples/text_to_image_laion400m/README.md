## Latent Diffusion Model 从零训练代码

本教程带领大家如何开启32层的**Latent Diffusion Model**的训练。

## 1 本地运行
### 1.1 安装依赖

在运行这个训练代码前，我们需要安装下面的训练依赖。

```bash
# 安装develop版本的paddle
pip install -U paddlenlp ppdiffusers visualdl fastcore Pillow
```

### 1.2 准备数据

#### laion400m_en.filelist文件内部格式如下所示
自己准备好处理后的数据，并且将文件放置于`/data/laion400m/`目录，其中里面的每个part的前三列为`caption文本描述, 占位符空, base64编码的图片`，`caption, _, img_b64 = vec[:3]`。

注意，当前`laion400m_en.filelist`只存放了10条数据路径，如果想要更多数据的话，请运行`python write_filelist.py`代码，运行后会生成6万条数据路径。
```
/data/laion400m/part-00000.gz
/data/laion400m/part-00001.gz
/data/laion400m/part-00002.gz
/data/laion400m/part-00003.gz
/data/laion400m/part-00004.gz
/data/laion400m/part-00005.gz
/data/laion400m/part-00006.gz
/data/laion400m/part-00007.gz
/data/laion400m/part-00008.gz
/data/laion400m/part-00009.gz
```
#### train.filelist.list训练文件内部格式如下所示
我们提供了`laion400m_en.filelist`，当然也可以存放其他`filelist`
```
./data/filelist/laion400m_en.filelist
```

### 1.3 使用trainner开启训练
#### 1.3.1 硬件要求
下面的代码需要具有32GB的显卡才可以预训练成功。

#### 1.3.2 单机单卡训练
```bash
python -u train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 10 \
    --learning_rate 4.6875e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --model_name_or_path CompVis/ldm_laion400M_pretrain \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased
```
或
```bash
bash run_single_trainer.sh
```

`train_txt2img_laion400m_trainer.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--model_name_or_path`: 本教程只能加载已经使用pytorch初始化好后的模型，可选**32层**的`CompVis/ldm_laion400M_pretrain`或者**12层**的`CompVis/ldm_12_laion400M_pretrain`模型，Tips: 这两个模型的`embedding`词表大小与`bert-base-uncased`的词表大小相一致。
> * `--per_device_train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--weight_decay`: AdamW优化器的`weight_decay`。
> * `--max_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--save_total_limit`: 最多保存多少个模型。
> * `--lr_scheduler_type`: 要使用的学习率调度策略。默认为 `constant`。
> * `--warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--logging_steps`: logging日志的步数。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--dataloader_num_workers`: Dataloader所使用的`num_workers`参数。
> * `--file_list`: file_list文件地址。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`。

#### 1.3.3 单机多卡训练
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ./laion400m_pretrain_output_trainer \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 10 \
    --learning_rate 4.6875e-5 \
    --weight_decay 0.02 \
    --max_steps 1000000000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 23 \
    --dataloader_num_workers 6 \
    --model_name_or_path CompVis/ldm_laion400M_pretrain \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased
```
或
```bash
bash run_multi_trainer.sh
```

### 1.4 自定义训练逻辑开启训练
#### 1.4.1 单机单卡训练
```bash
python -u train_txt2img_laion400m_no_trainer.py \
    --output_dir ./laion400m_pretrain_output_no_trainer \
    --train_batch_size 6 \
    --gradient_accumulation_steps 10 \
    --learning_rate 4.6875e-5 \
    --adam_weight_decay 0.02 \
    --max_train_steps 1000000000 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --seed 23 \
    --num_workers 6 \
    --pretrained_model_name_or_path CompVis/ldm_laion400M_pretrain \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased
```
或
```bash
bash run_single_no_trainer.sh
```

`train_txt2img_laion400m_no_trainer.py`代码可传入的参数解释如下：
> 主要修改的参数
> * `--pretrained_model_name_or_path`: 本教程只能加载已经使用pytorch初始化好后的模型，可选**32层**的`CompVis/ldm_laion400M_pretrain`或者**12层**的`CompVis/ldm_12_laion400M_pretrain`模型。
> * `--train_batch_size`: 训练时每张显卡所使用的`batch_size批量`，当我们的显存较小的时候，需要将这个值设置的小一点。
> * `--gradient_accumulation_steps`: 梯度累积的步数，用户可以指定梯度累积的步数，在梯度累积的step中。减少多卡之间梯度的通信，减少更新的次数，扩大训练的batch_size。
> * `--learning_rate`: 学习率。
> * `--adam_weight_decay`: AdamW优化器的`weight_decay`。
> * `--max_train_steps`: 最大的训练步数。
> * `--save_steps`: 每间隔多少步`（global step步数）`，保存模型。
> * `--lr_scheduler`: 要使用的学习率调度策略。默认为 `constant`。
> * `--lr_warmup_steps`: 用于从 0 到 `learning_rate` 的线性 warmup 的步数。
> * `--logging_steps`: logging日志的步数。
> * `--output_dir`: 模型保存路径。
> * `--seed`: 随机种子，为了可以复现训练结果，Tips：当前paddle设置该随机种子后仍无法完美复现。
> * `--num_workers`: Dataloader所使用的`num_workers`参数。
> * `--file_list`: file_list文件地址。
> * `--num_inference_steps`: 推理预测时候使用的步数。
> * `--model_max_length`: `tokenizer`中的`model_max_length`参数，超过该长度将会被截断。
> * `--tokenizer_name`: 我们需要使用的`tokenizer_name`。

#### 1.4.2 单机多卡训练
```bash
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_no_trainer.py \
    --output_dir ./laion400m_pretrain_output_no_trainer \
    --train_batch_size 6 \
    --gradient_accumulation_steps 10 \
    --learning_rate 4.6875e-5 \
    --adam_weight_decay 0.02 \
    --max_train_steps 1000000000 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --logging_steps 50 \
    --save_steps 5000 \
    --seed 23 \
    --num_workers 6 \
    --pretrained_model_name_or_path CompVis/ldm_laion400M_pretrain \
    --file_list ./data/filelist/train.filelist.list \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --tokenizer_name bert-base-uncased
```
或
```bash
bash run_multi_no_trainer.sh
```

## 2 TODO
- 同步pytorch初始化方法至paddle。
- 更新paddle加载模型的方式。
