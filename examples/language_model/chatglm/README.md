# ChatGLM

ChatGLM-6B 是一个开源的、支持中英双语问答的对话语言模型，基于 [General Language Model (GLM)](https://arxiv.org/abs/2103.10360) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGLM 相同的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。


本示例提供了 ChatGLM 模型的生成任务微调流程，适用于 THUDM/chatglm-6b 模型。

## 环境依赖
目前版本支持的功能较多，建议使用 paddlepaddle develop 版本以获得较好体验。下面给出了 cuda 11.2 的 paddle 安装方法。更多其他版本，请参考[官网首页](https://www.paddlepaddle.org.cn/)下载。
```
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

## AdvertiseGen 广告生成任务

本示例基于广告生成数据集 AdvertiseGen，输入为服装描述关键词，输出为相应的广告语，可从[这里](https://paddlenlp.bj.bcebos.com/datasets/examples/AdvertiseGen.tar.gz)下载。

### 多卡训练脚本（模型并行策略）

```
python -m paddle.distributed.launch --gpus "0,1,2,3" finetune_generation.py \
--model_name_or_path THUDM/chatglm-6b \
--task_path AdvertiseGen/ \
--max_steps 3000 \
--learning_rate 3e-5 \
--warmup_steps 20 \
--eval_steps 100 \
--logging_steps 1 \
--save_steps 1000 \
--save_total_limit 1 \
--output_dir ./checkpoints/chatglm-6b \
--src_length 64 \
--tgt_length 64 \
--per_device_eval_batch_size 4 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 32 \
--fp16 \
--fp16_opt_level O2 \
--recompute True \
--do_train \
--do_eval \
--tensor_parallel_degree 4
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`THUDM/chatglm-6b`。
- `task_path`: 数据集存储目录。
- `max_steps`: 模型训练步数。
- `learning_rate`: 参数更新的学习率。
- `warmup_steps`: 学习率热启的步数。
- `eval_steps`: 模型评估的间隔步数。
- `logging_steps`: 训练日志打印的间隔步数。
- `save_steps`: 模型参数保存的间隔步数。
- `save_total_limit`: 模型 checkpoint 保存的份数。
- `output_dir`: 模型参数保存目录。
- `src_length`: 上下文的最大输入长度，默认为128.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `gradient_accumulation_steps`: 模型参数梯度累积的步数，可用于扩大 batch size。实际的 batch_size = per_device_train_batch_size * gradient_accumulation_steps。
- `fp16`: 使用 float16 精度进行模型训练和推理。
- `fp16_opt_level`: float16 精度训练模式，`O2`表示纯 float16 训练。
- `recompute`: 使用重计算策略，开启后可节省训练显存。
- `do_train`: 是否训练模型。
- `do_eval`: 是否评估模型。
- `tensor_parallel_degree`: 模型并行数量。


## 模型预测

可以将模型python前向与推理结果比较：

```
python predict_generation.py \
    --model_name_or_path  ./checkpoints/chatglm-6b
```

当 checkpoint 使用`tensor parallel`存储为多分片格式时，也可以使用此脚本预测，或者将其合并为一个单分片权重。例如，下面模型保存为了四分片，

```
(base) root@localhost glm $ ll ./checkpoints/chatglm-6b/checkpoint-100/
total 82G
drwxr-xr-x 2 root root 4.0K Apr 16 22:41 ./
drwxr-xr-x 4 root root 4.0K Apr 16 22:41 ../
-rw-r--r-- 1 root root  811 Apr 16 22:40 config.json
-rw-r--r-- 1 root root 2.6M Apr 16 22:40 ice_text.model
-rw-r--r-- 1 root root 3.2G Apr 16 22:40 model_state.tp00.pdparams
-rw-r--r-- 1 root root 3.2G Apr 16 22:40 model_state.tp01.pdparams
-rw-r--r-- 1 root root 3.2G Apr 16 22:40 model_state.tp02.pdparams
-rw-r--r-- 1 root root 3.2G Apr 16 22:40 model_state.tp03.pdparams
```

可以运行以下命令将模型合并为单分片并保存。

```
python -m paddle.distributed.launch --gpus 0,1,2,3 predict_generation.py \
    --model_name_or_path  ./checkpoints/chatglm-6b/checkpoint-100/ \
    --merge_tensor_parallel_path  ./checkpoints/chatglm-merged
```

其中参数 `merge_tensor_parallel_path` 指定了合并后模型参数的存储位置。如果不设置这一参数，将只跑前向。

## 模型导出

在模型训练完毕后，可使用如下脚本将模型参数导出为静态图，用于模型推理。

```
python export_generation_model.py \
   --model_name_or_path ./checkpoints/chatglm-6b \
   --output_path ./checkpoints/infer/chatglm \
   --dtype "float32"
```

其中参数定义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录。
- `output_path`: 导出模型存储地址和文件前缀。示例中导出地址为 `./checkpoints/infer`，模型前缀为 `chatglm`。
- `dtype`: 模型参数类型，默认为`float32`，可选参数`float16`和`float32`。

## 模型推理（c++推理）

**环境依赖**

模型推理依赖于最新版本的 FastDeploy，可使用以下命令安装：

```
# GPU 安装
pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html`
```

运行以下命令，使用静态图进行模型推理。

```
python infer_generation.py \
    --model_path  ./checkpoints/infer \
    --model_prefix chatglm
```
