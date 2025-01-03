# 安装

git clone 代码到本地：

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git
export PYTHONPATH=/path/to/PaddleNLP:$PYTHONPATH
```

PaddleNLP 针对于Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能，使用之前需要预先安装自定义算子库：

```shell
#GPU设备安装自定义算子
cd PaddleNLP/csrc && python setup_cuda.py install
#XPU设备安装自定义算子
cd PaddleNLP/csrc/xpu/src && sh cmake_build.sh
#DCU设备安装自定义算子
cd PaddleNLP/csrc && python setup_hip.py install
#SDAA设备安装自定义算子
cd PaddleNLP/csrc/sdaa && python setup_sdaa.py install
```

到达运行目录，即可开始：

```shell
cd PaddleNLP/llm
```

大模型推理教程：

-  [llama](./llama.md)
-  [qwen](./qwen.md)
-  [mixtral](./mixtral.md)

获取最佳推理性能：

- [最佳实践](./best_practices.md)
