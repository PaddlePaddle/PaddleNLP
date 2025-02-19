## 🚣‍♂️ 使用 PaddleNLP 在 MX C550下跑通 llama2-13b 模型 🚣

PaddleNLP 在曦云®C550（[了解沐曦](https://www.metax-tech.com/)）上对 llama2-13B 模型进行了深度适配和优化，该套件实现了曦云 C550和 GPU 的训推入口完全统一，达到了『无缝切换』的效果。
曦云 C500 系列 GPU 是沐曦基于自主知识产权 GPU IP 打造的旗舰系列产品，具有强大的多精度混合算力，64GB 大容量高带宽内存，以及先进的多卡互联 MetaLink 技术。它搭载 MXMACA®软件栈，全面兼容主流 GPU 生态，应用迁移零成本，
可方便快捷地支撑智算、通用计算和数据处理等应用场景。

## 🚀 快速开始 🚀

### （0）在开始之前，您需要有一台插有曦云 C550机器，对此机器的系统要求如下：

| 芯片类型 | vbios 版本 | MXMACA 版本      |
| -------- | --------- | --------------- |
| 曦云 C550 | ≥ 1.13  | ≥ 2.23.0.1018 |

**注：如果需要验证您的机器是否插有曦云 C550 GPU，只需系统环境下输入以下命令，看是否有输出：**

```
mx-smi

#输出如下
mx-smi  version: 2.1.6

=================== MetaX System Management Interface Log ===================
Timestamp                                         : Mon Sep 23 06:24:52 2024

Attached GPUs                                     : 8
+---------------------------------------------------------------------------------+
| MX-SMI 2.1.6                        Kernel Mode Driver Version: 2.5.014         |
| MACA Version: 2.23.0.1018           BIOS Version: 1.13.4.0                      |
|------------------------------------+---------------------+----------------------+
| GPU         NAME                   | Bus-id              | GPU-Util             |
| Temp        Power                  | Memory-Usage        |                      |
|====================================+=====================+======================|
| 0           MXC550                 | 0000:2a:00.0        | 0%                   |
| 31C         44W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 1           MXC550                 | 0000:3a:00.0        | 0%                   |
| 31C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 2           MXC550                 | 0000:4c:00.0        | 0%                   |
| 31C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 3           MXC550                 | 0000:5c:00.0        | 0%                   |
| 31C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 4           MXC550                 | 0000:aa:00.0        | 0%                   |
| 30C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 5           MXC550                 | 0000:ba:00.0        | 0%                   |
| 31C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 6           MXC550                 | 0000:ca:00.0        | 0%                   |
| 30C         46W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+
| 7           MXC550                 | 0000:da:00.0        | 0%                   |
| 30C         47W                    | 810/65536 MiB       |                      |
+------------------------------------+---------------------+----------------------+

+---------------------------------------------------------------------------------+
| Process:                                                                        |
|  GPU                    PID         Process Name                 GPU Memory     |
|                                                                  Usage(MiB)     |
|=================================================================================|
|  no process found                                                               |
+---------------------------------------------------------------------------------+
```

### （1）环境准备：(这将花费您5~55min 时间)

1. 使用容器构建运行环境（可选）

```
 # 您可以使用 --device=/dev/dri/card0 指定仅GPU 0在容器内可见（其它卡同理），--device=/dev/dri 表示所有GPU可见
docker run -it --rm --device=/dev/dri
    --device=/dev/mxcd --group-add video -network=host --uts=host --ipc=host --privileged=true --shm-size 128g registry.baidubce.com/paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4
```

2. 安装 MXMACA 软件栈

   > 您可以联系 fae_support@metax-tech.com 以获取 MXMACA 安装包及技术支持， 已授权用户可以访问[沐曦软件中心](https://sw-download.metax-tech.com/login)获取相关安装包。
   >

```
# 假设您已下载并解压好MXMACA驱动
sudo bash /path/to/maca_package/mxmaca-sdk-install.sh
```

3. 安装 PaddlePaddle

①如果您已经通过 Metax 获取了 PaddlePaddle 安装包，您可以直接进行安装：

`pip install paddlepaddle_gpu-2.6.0+mc*.whl`

②您也可以通过源码自行编译 PaddlePaddle 安装包，请确保您已经正确安装 MXMACA 软件栈。编译过程使用了基于 MXMACA 的 cu-bridge 编译工具，您可以访问[文档](https://gitee.com/p4ul/cu-bridge/tree/master/docs/02_User_Manual)获取更多信息。

```

# 1. 访问 PaddlePaddle github仓库clone代码并切换至mxmaca分支.
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout release-mxmaca/2.6
# 2. 拉取第三方依赖
git submodule update --init
# 3. 配置环境变量
export MACA_PATH=/real/maca/install/path
export CUDA_PATH=/real/cuda/install/path
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${CUCC_PATH}/bin:${CUCC_PATH}/tools:${MACA_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
# 4. 检查配置是否正确
cucc --version
# 5. 执行编译
makdir -p build && cd build
cmake_maca .. -DPY_VERSION=3.8 -DWITH_GPU=ON -DWITH_DISTRIBUTE=ON -DWITH_NCCL=ON
make_maca -j64
# 6. 等待编译完成后安装whl包
pip install python/dist/paddlepaddle_gpu*.whl
```

4. 克隆 PaddleNLP 仓库代码，并安装依赖

```
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama2-13B模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
git checkout origin/release/3.0-beta1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### （2）推理：(这将花费您5~10min 时间)

1. 尝试运行推理 demo

```
cd llm/predict
python predictor.py --model_name_or_path meta-llama/Llama-2-13b-chat --dtype bfloat16 --output_file "infer.json" --batch_size 1 --decode_strategy "greedy_search"
```

成功运行后，可以查看到推理结果的生成，样例如下：

```
***********Source**********
解释一下温故而知新
***********Target**********

***********Output**********
 "温故而知新" (wēn gù er zhī xīn) is a Chinese idiom that means "to know the old and appreciate the new." It is often used to describe the idea that one can gain a deeper understanding and appreciation of something by studying its history and traditions, and then applying that knowledge to new situations and challenges.

The word "温" (wēn) in this idiom means "old" or "ancient," and "故" (gù) means "former" or "past." The word "知" (zhī) means "to know" or "to understand," and "新" (xīn) means "new."

This idiom is often used in the context of education, where it is believed that students should be taught the traditional methods and theories of a subject before being introduced to new and innovative ideas. By understanding the history and foundations of a subject, students can better appreciate and apply the new ideas and techniques that they are learning.

In addition to education, "温故而知新" can also be applied to other areas of life, such as business, where it is important to understand the traditions and practices of the industry before introducing new products or services. By understanding the past and the foundations of a particular field, one can gain a deeper appreciation of the present and make more informed decisions about the future.
```

2. 您也可以尝试参考 [文档](../../../../slm/examples/benchmark/wiki_lambada/README.md) 中的说明使用 wikitext 数据集验证推理精度。
