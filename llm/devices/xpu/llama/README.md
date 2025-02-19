## 🚣‍♂️ 使用 PaddleNLP 在 XPU 下跑通 llama2-7b 模型  🚣
PaddleNLP 在昆仑 XPU（[了解昆仑](https://www.kunlunxin.com/)）上对 llama2-7B 模型进行了深度适配和优化,下面给出详细安装步骤。

##  🚀 快速开始 🚀

### （0）在开始之前，您需要有一台昆仑 XPU 机器，对此机器的系统要求如下：

 | 芯片类型 | 卡型号 | 驱动版本 |
 | --- | --- | --- |
 | 昆仑 R480 | R300 | 4.31.0 |

#### 依赖环境说明
- **机器：** 昆仑 R480 32G，大概需要 17.5G（bs=1）
- **镜像：** registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
- **GCC 路径：** /usr/bin/gcc (8.4)
- **python 版本：**3.10
**注：本示例使用8卡机器：如果要验证您的机器是否为昆仑芯片，只需系统环境下输入命令，看是否有输出：**
```
lspci | grep 1d22
#例如：$ lspci | grep 1d22 , 输出如下
53:00.0 Communication controller: Device 1d22:3684
56:00.0 Communication controller: Device 1d22:3684
6d:00.0 Communication controller: Device 1d22:3684
70:00.0 Communication controller: Device 1d22:3684
b9:00.0 Communication controller: Device 1d22:3684
bc:00.0 Communication controller: Device 1d22:3684
d2:00.0 Communication controller: Device 1d22:3684
d5:00.0 Communication controller: Device 1d22:3684
```

### (1）环境准备：(这将花费您5～15min 时间)

1. 拉取镜像
```
# 注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包docker pull registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310
```

2. 参考如下命令启动容器
```
docker run -it --privileged=true  --net host --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 --device=/dev/xpuctrl:/dev/xpuctrl --name paddle-xpu-dev -v $(pwd):/work -w=/work -v xxx registry.baidubce.com/device/paddle-xpu:ubuntu20-x86_64-gcc84-py310 /bin/bash
```

3. 安装 paddlepaddle-xpu
```
# paddlepaddle『飞桨』深度学习框架，提供运算基础能力
wget https://paddle-whl.bj.bcebos.com/nightly/xpu/paddlepaddle-xpu/paddlepaddle_xpu-3.0.0.dev20240612-cp310-cp310-linux_x86_64.whl
python -m pip install paddlepaddle_xpu-3.0.0.dev20240612-cp310-cp310-linux_x86_64.whl

nightly版本链接：
https://www.paddlepaddle.org.cn/packages/nightly/xpu/paddlepaddle-xpu/
```

4. 克隆 PaddleNLP 仓库代码，并安装依赖
```
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama2-7B模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
# Clone PaddleNLP
git clone https://github.com/PaddlePaddle/PaddleNLP
cd PaddleNLP
# 切换到对应指定依赖的提交
git checkout 0844a5b730c636ad77975fd30a485ad5dc217eac
# 安装依赖
pip install -r requirements.txt
python -m pip install -e .

# 下载XPU自定义算子
cd csrc/xpu/src
# 设置 XDNN, XRE and XTDK 的路径后一键执行。
wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/release_paddle/20240429/xdnn-ubuntu_x86_64.tar.gz
wget https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev/release_paddle/20240429/xre-ubuntu_x86_64.tar.gz
wget https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/release_paddle/2.7.98.2/xtdk-llvm15-ubuntu1604_x86_64.tar.gz

# 解压到当前目录
tar -xf xdnn-ubuntu_x86_64.tar.gz
tar -xf xre-ubuntu_x86_64.tar.gz
tar -xf xtdk-llvm15-ubuntu1604_x86_64.tar.gz

# 设置环境变量
export PWD=$(pwd)
export XDNN_PATH=${PWD}/xdnn-ubuntu_x86_64/
export XRE_PATH=${PWD}/xre-ubuntu_x86_64/
export CLANG_PATH=${PWD}/xtdk-llvm15-ubuntu1604_x86_64/

#XPU设备安装自定义算子
bash ./cmake_build.sh
cd -
```

### （2）数据准备：(这将花费您2～5min 时间)
精调：为了方便测试，我们也提供了数据集可以直接使用：
```
# 进入llm目录
cd llm
# 下载数据集
wget https://baidu-kunlun-customer.su.bcebos.com/paddle-llm/infernce.tar.gz
# 解压
tar -zxvf infernce.tar.gz
```

### (3）推理：(这将花费您10~15min 时间)
```
#可以通过设置 FLAGS_selected_xpus 指定容器可见的昆仑芯片卡号
export FLAGS_selected_xpus=0
#设置环境变量
export PYTHONPATH=$PYTHONPATH:../../../PaddleNLP/
```

高性能动态图推理命令参考
```
python predictor.py --model_name_or_path ./inference --dtype float16 --src_length 2048 --max_length 2048 --mode "static" --batch_size 1 --inference_model --block_attn --device xpu
```

最终，预期结果：
```
[[2024-08-22 13:23:34,969] [    INFO] - preprocess spend 0.012732744216918945
[2024-08-22 13:23:34,994] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load './inference'.
[2024-08-22 13:23:35,014] [    INFO] - Start read result message
[2024-08-22 13:23:35,014] [    INFO] - Current path is /home/workspace/wangy_test/PaddleNLP/llm
[2024-08-22 13:23:53,313] [    INFO] - running spend 18.322898864746094
[2024-08-22 13:23:53,326] [    INFO] - Finish read result message
[2024-08-22 13:23:53,327] [    INFO] - End predict
***********Source**********
解释一下“温故而知新”
***********Target**********

***********Output**********
"温故而知新" (wēn gǔ èr zhī xīn) is a Chinese idiom that means "to understand the old in order to appreciate the new."
The word "温故" (wēn gǔ) means "old" or "ancient," while "知新" (zhī xīn) means "to know or understand something new." The idiom as a whole suggests that in order to fully appreciate something new, one must first have a deep understanding of the past or the traditional ways of doing things.
In other words, "温故而知新" means that one should have a foundation of knowledge and understanding before being open to new ideas or experiences. This can help prevent one from being too quick to dismiss the old in favor of the new, and instead allow for a more nuanced and informed appreciation of both.
For example, if someone is learning a new language, they may find it helpful to study the grammar and syntax of the language's ancestor languages in order to better understand the nuances of the new language. Similarly, if someone is learning a new skill or craft, they may find it helpful to study the traditional techniques and methods of the craft in order to better understand the new approaches and technologies that are being introduced.
Overall, "温故而知新" is a reminder to approach new things with a sense of respect and appreciation for the past, and to be open to learning and growing in a way that is informed by a deep understanding of both the old and the new.
[2024-08-22 13:23:53,328] [    INFO] - Start predict
[2024-08-22 13:23:53,335] [    INFO] - preprocess spend 0.007447242736816406
[2024-08-22 13:23:53,357] [    INFO] - We are using <class 'paddlenlp.transformers.llama.tokenizer.LlamaTokenizer'> to load './inference'.
[2024-08-22 13:23:53,386] [    INFO] - Start read result message
[2024-08-22 13:23:53,386] [    INFO] - Current path is /home/workspace/wangy_test/PaddleNLP/llm
[2024-08-22 13:23:57,859] [    INFO] - running spend 4.506801605224609
[2024-08-22 13:23:57,863] [    INFO] - Finish read result message
[2024-08-22 13:23:57,864] [    INFO] - End predict
***********Source**********
你好，请问你是谁?
***********Target**********

***********Output**********
Hello! I'm just an AI assistant, I don't have a personal identity or ego, but I'm here to help you with any questions or tasks you may have. I'm a machine learning model trained to provide helpful and informative responses, and I'm here to assist you in a safe and respectful manner. Is there anything else I can help you with?
```
