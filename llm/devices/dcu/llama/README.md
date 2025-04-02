# 使用 PaddleNLP 在海光 DCU 上运行 llama2-13b 模型
Paddle 框架与 PaddleNLP 套件在海光的 DCU 产品上进行了深度的适配和优化，实现了大模型在训练和推理上与 GPU 高度统一，在精度和性能上拥有先进水平。

海光 DCU 产品在 PaddleNLP 组合套件上拥有多种技术优点：

- **完全支持4D 混合并行分布式训练，灵活适应各种训练策略。**
- **各类高性能的融合算子，提升训推性能。**
- **优化的通讯库，掩盖分布式训推延迟。**

##  🚀 快速开始 🚀

## 环境准备：

### 1.硬件平台


 | 芯片类型 | DTK 版本 |
 | --- | --- |
 | K100_AI | 24.04.1 |

**本示例使用8卡机器，并通过微调训练+推理的流程演示运行方法，使用 hy-smi 命令查看运行环境中的 DCU 信息，如下所示：**
```
$ hy-smi

============================ System Management Interface =============================
======================================================================================
DCU     Temp     AvgPwr     Perf     PwrCap     VRAM%      DCU%      Mode
0       49.0C    118.0W     auto     800.0W     0%         0%        Normal
1       48.0C    120.0W     auto     800.0W     0%         0%        Normal
2       53.0C    116.0W     auto     800.0W     0%         0%        Normal
3       49.0C    138.0W     auto     800.0W     0%         0%        Normal
======================================================================================
=================================== End of SMI Log ===================================
```

### 2.环境准备：
推荐使用 docker 方式运行，提供拉取的 docker 镜像，关于本项目所需新版本 DTK 等均可从[光合](https://developer.hpccube.com/tool/)开发者社区下载安装，docker 中默认使用 dtk-24.04.1。

(1). 拉取镜像
```
# 注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
docker pull registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc82
```
(2). 参考如下命令启动容器：
```
docker run -it \
    --network=host \
    --name=paddle_llama \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size=128G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -u root \
    --ulimit stack=-1:-1 \
    --ulimit memlock=-1:-1 \
    -v $(pwd):/workspace \
    -v /opt/hyhal:/opt/hyhal \
    registry.baidubce.com/device/paddle-dcu:dtk24.04.1-kylinv10-gcc82 \
    /bin/bash
```
(3). 安装 paddle
```
# paddlepaddle『飞桨』深度学习框架，提供运算基础能力
python -m pip install paddlepaddle-dcu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
```

(4). 克隆 PaddleNLP 仓库代码，并安装依赖
```
# 用paddlenlp develop分支
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/llm # 到达运行目录
pip install -r ../requirements.txt
```
(5). 安装 paddlenlp_ops
```
# PaddleNLP仓库内置了rms相关的专用算子
cd slm/model_zoo/gpt-3/external_ops
python setup.py install
```

## 3.微调：
- **注：** 进入 llm 路径进行以下操作。
### 数据集准备
我们提供了数据集 demo 便于您调试使用
```
wget https://bj.bcebos.com/paddlenlp/datasets/examples/alpaca_demo.gz
tar -xvf alpaca_demo.gz
```
我们支持的精调数据格式是每行包含一个字典的 json 文件，每个字典包含以下字段：
- `src`: `str, List(str)`，指模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt`: `str, List(str)`，指模型的输出。
样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
#您可以根据此格式自行制作精调数据。
```
### Lora 微调

可参考以下脚本启动 Lora 微调训练：
```
PYTHONPATH=.. python run_finetune.py dcu/llama/lora_argument.json
```
### sft 微调
可参考以下超参启动 Lsft 微调训练：
```
PYTHONPATH=.. python run_finetune.py dcu/llama/sft_argument.json
```
## 3.预训练：
### 数据准备
数据详细制作流程可参考[此处](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/tools/preprocess/README.md)，例：OpenWebText2预训练数据制作参考[此处](https://paddlenlp.readthedocs.io/zh/latest/llm/pretraining/data/OpenWebText2.html)

为了方便用户运行测试本模型，本项目提供了处理好的100k 条 doc 的训练样本：

```
cd PaddleNLP/llm/
mkdir data && cd data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. && tree data
data
├── llama_openwebtext_100k.bin
└── llama_openwebtext_100k.idx
```
- **注：** 与微调数据集区分路径
### 运行脚本

该训练脚本可以单节点也可多节点运行，每节点8张 DCU-K100AI-64G。

并行配置采用 TP 1，PP 8，使用 fp16精度预训练。

可参考以下脚本启动预训练：

```
python -m paddle.distributed.launch \
    --gpus '0,1,2,3,4,5,6,7' \
    run_pretrain.py dcu/llama/pretrain_pp8.json
```

## 4.高性能推理
高性能推理内置动态插入和全环节算子融合策略，隐藏了底层实现的细节，实现了开箱即用高性能并行推理能力。在保持高性能推理和动态插入的基础上可以动态地为 cachekv 分配存储空间，极大地节省显存，从而在同一时刻处理更多的 query 以获得吞吐的提升。

(1). 环境准备

PaddleNLP 针对于 Transformer 系列编写了高性能自定义算子，提升模型在推理和解码过程中的性能，使用之前需要预先安装自定义算子库：
```
# DCU设备安装自定义算子
cd PaddleNLP/csrc && python3 setup_hip.py install
```
(2). 高性能推理

下面分别给出关闭 BlockAttention 和打开 BlockAttention 进行高性能推理的命令参考：

a.关闭 BlockAttention 的高性能推理

**动态图：**

```
# fp16
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
# a8w8
python3 ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
```
**静态图：**

```
# step1: 静态图导出
# fp16
python3 ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16
# a8w8
python3 ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16

# step2: 静态图推理
python3 ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype float16 --mode static （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
```

b. 打开 BlockAttebtion 的高性能推理

**动态图：**

```
# fp16
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 --block_attn （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
# a8w8
python3 ./predict/predictor.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --dtype float16 --block_attn （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
# cachekv
python3 ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --dtype float16 --block_attn ---cachekv_int8 （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
```

**静态图：**

```
# step1: 静态图导出
# fp16
python3 ./predict/export_model.py --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16 --block_attn
# a8w8
python3 ./predict/export_model.py --model_name_or_path checkpoints/llama_ptq_ckpts --inference_model --output_path ./inference --dtype float16 --block_attn
# cachekv
python3 ./predict/export_model.py  --model_name_or_path meta-llama/Llama-2-13b-chat --inference_model --output_path ./inference --dtype float16 --block_attn --cachekv_int8

# step2: 静态图推理
# fp16
python3 ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype float16 --mode static --block_attn  （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
# a8w8
python3 ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype float16 --mode static --block_attn  （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
# cachekv
python3 ./predict/predictor.py  --model_name_or_path ./inference --inference_model --dtype float16 --mode static --cachekv_int8 --block_attn  （测性能可选：--batch_size 1 --src_length 3072 --max_length 1024 --benchmark）
```

## 5.应用场景

(1).算法类别

`自然语言处理`

(2).热点应用行业

`医疗,教育,科研,金融`

## 6.源码仓库及问题反馈

- [https://developer.hpccube.com/codes/modelzoo/llama_paddle](https://developer.hpccube.com/codes/modelzoo/llama_paddle)

## 7.参考

* [https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)
