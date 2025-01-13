## 🚣‍♂️ 使用 PaddleNLP 在 MLU 下跑通 llama-13b 模型 🚣
PaddleNLP 在寒武纪 MLU（[了解寒武纪](https://www.cambricon.com/)）上对 llama-13B 模型进行了深度适配和优化，该套件实现了寒武纪 MLU 和 GPU 的训推入口基本统一，达到了『无缝切换』的效果。

##  🚀 快速开始 🚀

### （0）在开始之前，您需要有一台寒武纪 MLU 机器，对此机器的系统要求如下：

 | 芯片类型 | 驱动版本 | CNtoolKit 版本 |
 | --- | --- | --- |
 | MLU |  5.10.31  |  3.10.2 |

**注：本示例使用8卡机器，并通过微调训练+推理的流程演示运行方法**
**注：如果要验证您的机器是否为寒武纪芯片，只需系统环境下输入命令，看是否有输出：**
```
cnmon

#例如：$ cnmon , 输出如下
Thu Dec 19 22:05:42 2024
+------------------------------------------------------------------------------+
| CNMON v5.10.31                                               Driver v5.10.31 |
+-------------------------------+----------------------+-----------------------+
| Card  VF  Name       Firmware |               Bus-Id | Util        Ecc-Error |
| Fan   Temp      Pwr:Usage/Cap |         Memory-Usage | Mode     Compute-Mode |
|===============================+======================+=======================|
| 0     /   MLUXXX-XX    v1.5.0 |         0000:4F:00.0 | 0%                  0 |
|  0%   35C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 1     /   MLUXXX-XX    v1.5.0 |         0000:53:00.0 | 0%                  0 |
|  0%   34C        100 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 2     /   MLUXXX-XX    v1.5.0 |         0000:6F:00.0 | 0%                  0 |
|  0%   35C        100 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 3     /   MLUXXX-XX    v1.5.0 |         0000:73:00.0 | 0%                  0 |
|  0%   34C        109 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 4     /   MLUXXX-XX    v1.5.0 |         0000:AF:00.0 | 0%                  0 |
|  0%   34C        107 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 5     /   MLUXXX-XX    v1.5.0 |         0000:B3:00.0 | 0%                  0 |
|  0%   33C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 6     /   MLUXXX-XX    v1.5.0 |         0000:CF:00.0 | 0%                  0 |
|  0%   36C        102 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+
| 7     /   MLUXXX-XX    v1.5.0 |         0000:D3:00.0 | 0%                  0 |
|  0%   33C        105 W/ 550 W |     0 MiB/ xxxxx MiB | FULL          Default |
+-------------------------------+----------------------+-----------------------+

+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  Card  MI  PID     Command Line                             MLU Memory Usage |
|==============================================================================|
|  No running processes found                                                  |
+------------------------------------------------------------------------------+
```

### （1）环境准备：(这将花费您5～15min 时间)
1. 拉取镜像
```
# 注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
docker pull registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310
```
2. 参考如下命令启动容器
```
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    --device /dev/cambricon_dev0 \
    --pid=host --ipc=host -it --privileged \
    -v -v /usr/bin/cnmon/:/usr/bin/cnmon/ \
    -v /usr/local/dcmi:/usr/local/dcmi \
    registry.baidubce.com/device/paddle-mlu:ctr2.15.0-ubuntu20-x86_64-gcc84-py310 /bin/bash
```
3. 安装 paddle
```
# paddlepaddle『飞桨』深度学习框架，提供运算基础能力
pip install paddlepaddle==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```
4. 安装 paddleCustomDevice
```
# paddleCustomDevice是paddlepaddle『飞桨』深度学习框架的自定义硬件接入实现，提供MLU的算子实现。
pip install https://paddle-device.bj.bcebos.com/2.6.1/mlu/paddle_custom_mlu-2.6.1-cp310-cp310-linux_x86_64.whl
# 如想源码编译安装，请参考https://github.com/PaddlePaddle/PaddleCustomDevice/blob/release/2.6/backends/mlu/README_cn.md
```
5. 克隆 PaddleNLP 仓库代码，并安装依赖
```
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama2-13B模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
git checkout 1fc942924df46c8e149ac7ce8cbc42d884fbb823
python -m pip install -r requirements.txt
python -m pip install -e .
```


### （2）Pretrain 阶段数据准备：(这将花费您8～9min 时间)
```
# 下载 OpenWebtext2 数据集
mkdir openwebtext2 && cd openwebtext2
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.bin
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/PRETRAIN/openwebtext2/llama/mmap/llama_mmap.idx
```
### （3）模型下载：（这将花费您6～7min 时间）
```
# 随机初始化模型，使用此模型（__internal_testing__/sci-benchmark-llama-13b-init0501）初始化训练

python download_init0501_model.py
```

### （4）模型预训练：(这将花费您约5天时间)
当前为四机配置，需要用户根据机器自行调整，机器 ip，batch size。
```
# 机器1
bash run_train.sh

# 机器2
ssh notebook-devenviron-1104-202919-b065xu-worker-0
bash run_train.sh

# 机器3
ssh notebook-devenviron-1104-202919-b065xu-worker-1
bash run_train.sh

# 机器4
ssh notebook-devenviron-1104-202919-b065xu-worker-2
bash run_train.sh
```
### （5）分布式训练参数合并：（这将花费您1~2min 时间）
```
#分布式训练参数合并, 执行完后在./checkpoints/llama_pretrain_ckpts/checkpoint-5000/ 目录下，生成 25G model_state.pdparams
bash run_merge.sh
```

### （6）预训练后期模型精度验证：（这将花费您14~15min 时间）
使用提供的基准测试脚本，在给定的验证集上测试。
```
bash run_eval.sh
```

### （7）预训练模型效果测试： （这将花费您15~16min 时间）
使用提供的基准测试脚本，在给定的测试数据集 LAMBADA 上测试。
```
# 数据集准备
mkdir wiki_lambada && cd wiki_lambada
wget https://paddlenlp.bj.bcebos.com/data/benchmark/lambada_test.jsonl
cd -

bash run_acc.sh
```

### （8）精调模型效果测试（SFT+LORA）: （这将花费您约5天时间）
下载 meta-math/MetaMathQA 、sahil2801/CodeAlpaca-20k 、Open-Orca/SlimOrca 数据集，并且将这3个数据集放到指定的目录 ./data_math 、./data_code 、./data_slim。
数据集下载链接: https://pan.baidu.com/s/1tbGYBqdmlrBq3vP_-WAIQA  密码: a5eu
```
#1.meta-math/MetaMathQA 任务
bash run_math_lora.sh
bash run_math_sft.sh

#2.sahil2801/CodeAlpaca-20k 任务
bash run_code_lora.sh
bash run_code_sft.sh

#3.Open-Orca/SlimOrca 任务
bash run_slim_lora.sh
bash run_slim_sft.sh
```
