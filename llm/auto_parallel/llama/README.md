# LLaMA 自动并行训练

## 1. 模型组网介绍

- 动静统一自动并行组网[modeling_auto.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/llama/modeling_auto.py)，当前主要支持预训练，包括动态图和动转静训练，未来会扩展支持 SFT 等流程。

## 2. 预训练准备

安装最新的 Paddle，建议使用 nightly 版本，请前往 [Paddle 官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html) 进行安装。

下载预先处理好的数据，并解压到 `./data` 目录下：
```shell
# llama 模型数据下载
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx

mkdir data
mv llama_openwebtext_100k.bin ./data
mv llama_openwebtext_100k.idx ./data
```

安装自定义算子:
```shell
# 编译自定义算子，可选
cd ../../../slm/model_zoo/gpt-3/external_ops/ && python3 setup.py install && cd -

```
## 3. 预训练
- 动态图训练
参考训练脚本 **run_pretrain_auto.sh**，运行8卡 dp2mp2pp2的并行策略。
- 动转静训练
参考训练脚本 **run_pretrain_auto.sh**，并开启 `to_static=1`，运行8卡 dp2mp2pp2的并行策略。

您可以参考 **run_pretrain_auto.sh**，按需求修改相关参数进行训练。

## 4.推理
推理流程包括：动态图推理 -> 动转静导出模型 -> 静态图推理。当前自动并行预训练保存的模型参数已支持用于动态图推理；动转静导出模型、静态图推理步骤请参考 [LLaMA 系列大模型运行文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/llama.md)。

以动态图自动并行训练（dp2mp2pp2）为例。
- 分布式 ckpt 合并为单卡模型参数：

```python
import paddle
import paddle.distributed as dist

ckpt_path='/path/for/dist_ckpt'
# offload=1, 参数 offload 到 CPU，减少显存占用
merged_state_dict = dist.checkpoint.load_state_dict.load_merged_state_dict(ckpt_path, offload=1)
paddle.save(merged_state_dict, 'model_state.pdparams')

# 上述合并的模型参数格式为Paddle原生格式，如需转换为unified_param格式(safetensors)，可继续执行如下代码：
python PaddleNLP/llm/auto_parallel/utils/convert_to_safetensors.py --input_path input_path  [--output_path output_path] [--split_num split_num] [--offload offload]

# 参数介绍
--input_path: 输入的单卡模型参数路径
--output_path: 可选，输出模型参数路径，默认为'./temp'
--split_num: 可选，输出的模型参数分片数，默认为 1
--offload: 可选，是否将参数 offload 到 CPU，默认为 false
```

- 动态图推理

    [大模型推理教程](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/predict/inference.md)
