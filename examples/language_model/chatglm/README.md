# ChatGLM

ChatGLM-6B 是一个开源的、支持中英双语问答的对话语言模型，基于 [General Language Model (GLM)](https://arxiv.org/abs/2103.10360) 架构，具有 62 亿参数。ChatGLM-6B 使用了和 ChatGLM 相同的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。


本示例提供了 ChatGLM 模型的生成任务微调流程，适用于 THUDM/chatglm-6b 模型。

## 环境依赖
目前版本支持的功能较多，建议使用paddlepaddle develop版本以获得较好体验。下面给出了cuda 11.2的paddle安装方法。更多其他版本，请参考[官网首页](https://www.paddlepaddle.org.cn/)下载。
```
python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```

## AdvertiseGen 广告生成任务

本示例基于广告生成数据集 AdvertiseGen，输入为服装描述关键词，输出为相应的广告语，可从[这里](https://paddlenlp.bj.bcebos.com/datasets/examples/AdvertiseGen.tar.gz)下载。

### 多卡训练脚本（模型并行策略

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
--per_device_eval_batch_size 16 \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 8 \
--fp16 \
--fp16_opt_level O2 \
--recompute \
--do_train \
--do_eval
--tensor_parallel_degree 2
```

其中参数释义如下：

- `model_name_or_path`: 预训练模型内置名称或者模型所在目录，默认为`THUDM/chatglm-6b`。
- `task_path`: 数据集存储目录。
- `src_length`: 上下文的最大输入长度，默认为128.
- `tgt_length`: 生成文本的最大长度，默认为160.
- `tensor_parallel_degree`: 模型并行参数。


## 模型导出
使用`export_generation_model.py`脚本，传入我们需要的模型地址，和输出地址即可。如果需要导出`float16`参数的模型，请指定`dtype`参数为`float16`。
```
python export_generation_model.py \
   --model_name_or_path ./checkpoints/chatglm-6b \
   --output_path ./checkpoints/infer/chatglm \
   --dtype "float32"
```

## 模型推理 (c++推理)
需要依赖` pip install fastdeploy-gpu-python==0.0.0 -f https://www.paddlepaddle.org.cn/whl/fastdeploy_nightly_build.html` (cpu请安装`fastdeploy-python`)
```
python infer_generation.py \
    --model_path  ./checkpoints/infer \
    --model_prefix chatglm
```

## 模型预测（python）
可以将模型python前向与推理结果比较：
```
python predict_generation.py \
    --model_name_or_path  ./checkpoints/chatglm-6b
```
当ckpt为使用的`tensor parallel`存储为多分片格式时，也可使用此脚本预测，或者合并为一个单分片权重
例如下面4分片的例子（此模型为`glm-10b-chinese`）
```
(base) root@localhost glm $ ll ./checkpoints/chatglm-6b/checkpoint-100/
total 130G
drwxr-xr-x 2 root root 4.0K Apr  7 18:21 ./
drwxr-xr-x 4 root root 4.0K Apr  7 20:02 ../
-rw-r--r-- 1 root root  201 Apr  7 18:20 added_tokens.json
-rw-r--r-- 1 root root 998K Apr  7 18:20 cog-pretrain.model
-rw-r--r-- 1 root root  892 Apr  7 18:20 config.json
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp00.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp01.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp02.pdparams
-rw-r--r-- 1 root root 4.7G Apr  7 18:20 model_state.tp03.pdparams
```
设置 merge_tensor_parallel_path，可以将merge好的参数存储到对应位置。不过不设置此参数，将只跑前向预测。
```
python -m paddle.distributed.launch --gpus 0,1,2,3 predict_generation.py \
    --model_name_or_path  ./checkpoints/chatglm-6b/checkpoint-100/ \
    --merge_tensor_parallel_path  ./checkpoints/chatglm-merged
```
