## 超大模型部署

目录：
- [简介](#简介)
- [环境安装](#环境安装)
- [模型导出](#模型导出)
- [自动切分](#自动切分)
- [推理部署](#推理部署)



超大模型由于参数容量大、显存/内容占用较多，对如何高效推理提出挑战。飞桨推出了针对分布式推理、大模型的压缩、服务化全流程部署方案。 其中分布式推理采用[张量模型并行、流水线并行技术](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/distributed_introduction.html)，这些技术通用用于超大模型训练。推理场景与训练场景的不同点包括: 硬件特性不同、硬件数量不同、通信环境不同。为了充分利用推理硬件效率，将飞桨自适应并行训练技术应用于推理场景，针对推理硬件拓扑、环境进行自适应的切分，进行自适应并行推理。

模型压缩旨在提升推理效率、节约部署资源，飞桨模型压缩工具PaddleSlim包含丰富的压缩方法，诸如量化、稀疏化技术，这些技术不仅可以使得模型容量大大减少，从而节约部署硬件数量，还可以降低推理时延，提升吞吐。在大模型压缩上，依然存在挑战。从算法上，超大模型通常层数较深，如量化误差会累积越大，稀疏化要求的大稀疏度下，很容易出现精度损失；从压缩工具上，由于超大模型显存占用较多，模型压缩工具也需要适配训练并行技术，在张量模型并行、Sharding并行、流水线并行的基础上支持量化缩放系数的统计，支持稀疏掩码训练等；从推理效率与精度平衡上，量化依据量化对象，分为了仅权重量化、权重/激活均量化，依据量化的Bit数，包括8Bit、4Bit等，依据量化的位置，分为部分量化、全量化，稀疏也分为非结构化稀疏、半结构化稀疏等，需依据精度和推理速度、显存/内存权衡选取策略，并且需要在分布式推理的基础上支持量化、稀疏推理。



超大模型云端部署，飞桨还提供了PaddleServing服务化支持，可以使得用户比较容易部署到多机多卡上，对于服务请求自动进行Batch处理、容错调度等。



本教程以GPT-3为例介绍如何进行超大模型部署，下面重点介绍模型导出、自动切分、推理部署，模型压缩内容后续会提供。服务化部署教程，采用其他预训练模型进行介绍。



### 环境安装

版本依赖如下:

Paddle: >= 2.3.0

PaddleNLP: develop分支



由于此前飞桨发布的Python安装包，不包含分布式推理功能，需要源码编译，后续会优化此步骤。参考[源码编译教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html) ，需要安装NCCL，编译命令设置参考如下，注意设置 `-DWITH_DISTRIBUTE=ON`

```
cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON
```



### 模型导出

在PaddleNLP，GTP-3提供了静态图、动态图训练，本次教程基于静态图训练代码进行，后续提供动态图支持。

首先，下载PaddleNLP:

```
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/examples/language_model/gpt-3/static/
```

推理模型导出，下面脚本默认是张量模型并行度为1，依据想要部署的GPU卡数，设置

```
run_gen.sh
```

关键参数介绍如下，也可以通过 `python run_generation.py --help` 查看参数列表和设置帮助信息。

- gpus:  设置GPU个数，也就是并行数
- model_type: 设置模型类型
- mp_degree: 张量模型并行度
- max_seq_len: 输入字长
- max_dec_len: 输出字长

注意: 自动切分时，不需要提供设置mp_degree，后续会补充自动切分内容。

运行`bash run_gen.sh`，模型会导出到当前目录。mp_degree为1时导出模型为`inference_model_pp1mp1`，mp_degree为2时导出模型为`inference_model_pp1mp1`


### 推理部署
推理部署前，参考前面的模型导出步骤，确保已导出好模型。
```
cd PaddleNLP/examples/language_model/gpt-3/static/
bash run_gen.sh  # 导出模型
```
导出好模型后，即可使用高性能推理脚本进行推理。以两卡张量模型并行为例，通过`model_path`指定导出的模型目录，
使用如下命令便可以基于Paddle Inference进行高性能预测：
```
cd ../deploy/python
python -m paddle.distributed.launch \
        --gpus 0,1 \
        inference.py --model_type gpt \
        --model_path ../../static/inference_model_pp1mp2/
```


#### 服务化部署
TBD




### Benchmark
TBD
