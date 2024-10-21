## 🚣‍♂️ 使用PaddleNLP在NPU下跑通baichuan2-13b模型

##  🚀 快速开始 🚀

### （0）在开始之前，您需要有一台昇腾NPU机器，对此机器的系统要求如下：

 | 芯片类型 | 驱动版本 | CANN版本 |
 | --- | --- | --- |
 | 昇腾910 | 23.0.3 | CANN 8.0.RC1 |

**注：本示例使用8卡机器，并通过微调训练+推理的流程演示运行方法**
**注：如果要验证您的机器是否为昇腾910B芯片，只需系统环境下输入命令，看是否有输出：**
```
lspci | grep d802

#例如：$ lspci | grep d802 , 输出如下
28:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
29:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
38:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
39:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
48:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
49:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
59:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
5a:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
98:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
99:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
b8:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
b9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
c8:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
c9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
d9:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
da:00.0 Processing accelerators: Huawei Technologies Co., Ltd. Device d802 (rev 20)
```

### （1）环境准备：(这将花费您5～15min时间)
1. 拉取镜像
```
# 注意此镜像仅为开发环境，镜像中不包含预编译的飞桨安装包
docker pull registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39
```
2. 参考如下命令启动容器，可以通过设置 ASCEND_RT_VISIBLE_DEVICES 指定容器可见的昇腾卡号
```
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py39 /bin/bash
```
3. 安装paddle
```
# paddlepaddle『飞桨』深度学习框架，提供运算基础能力
python -m pip install paddlepaddle==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```
4. 安装paddleCustomDevice
```
# paddleCustomDevice是paddlepaddle『飞桨』深度学习框架的自定义硬件接入实现，提供NPU的算子实现。
python -m pip install paddle-custom-npu==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/npu/
# 如想源码编译安装，请参考https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README_cn.md
```
5. 克隆PaddleNLP仓库代码，并安装依赖
```
# PaddleNLP是基于paddlepaddle『飞桨』的自然语言处理和大语言模型(LLM)开发库，存放了基于『飞桨』框架实现的各种大模型，llama2-13B模型也包含其中。为了便于您更好地使用PaddleNLP，您需要clone整个仓库。
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
python -m pip install -r requirements.txt
python -m pip install -e .
```
6. 安装 paddlenlp_ops
```
# PaddleNLP仓库内置了部分昇腾专用的融合算子，以便用户享受到极致压缩的推理成本
cd csrc/npu
python setup.py build bdist_wheel
pip install dist/paddlenlp_ops-0.0.0-cp39-cp39-linux_x86_64.whl
cd -
```
### （2）数据准备：(这将花费您2～5min时间)
sft为精调策略，我们提供了广告生成数据集demo便于您调试使用
```
#精调：为了方便测试，我们也提供了广告生成数据集可以直接使用：
cd llm/npu/baichuan
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz
tar -zxvf AdvertiseGen.tar.gz
```
我们支持的精调数据格式是每行包含一个字典的json文件，每个字典包含以下字段：
- `src`: `str, List(str)`，指模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt`: `str, List(str)`，指模型的输出。
样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
#您可以根据此格式自行制作精调数据。
```

### （3）训练：(这将花费您约4小时的时间)
我们在本目录中提供了对应Pretrain/SFT/LoRA的三个入口脚本，并已经按照8张910芯片的训练资源优化了并行策略等配置供您参考。启动微调训练的详细步骤如下：
```
# 运行sft策略
bash baichuan2_13b_npu_sft_N1C8.sh
```
### （4）推理
推理前需要准备推理用的配置文件，在merge好参数的路径下(本教程下路径为：`./output/sft_bf16_baichuan_N1C8`)将`config.json`更改为下面的内容：
```
{
  "architectures": [
    "LlamaForCausalLM"
  ],
  "tokenizer_class": "LlamaTokenizer",
  "bos_token_id": 1,
  "eos_token_id": 2,
  "gradient_checkpointing": false,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 13696,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "tie_word_embeddings": false,
  "dtype": "bfloat16",
  "use_cache": true,
  "alibi": true,
  "vocab_size": 125696
}
```
从训练产出的动态图模型中导出静态图模型，执行如下命令进行导出：
```
bash export_npu.sh ./output/sft_bf16_baichuan_N1C8/ ./inference
```
最终，我们通过静态图的模型执行推理：
```
# 执行推理代码
bash predict_npu.sh ./inference
```
```
***********Source**********
睡不着该怎么办？
***********Target**********

***********Output**********
如果你睡不着，可以尝试以下方法：

1. 保持放松：尝试深呼吸、渐进性肌肉松弛或冥想等放松技巧。
2. 改变环境：调整房间的温度、光线和噪音，创造一个舒适的睡眠环境。
3. 避免咖啡因、尼古丁和酒精：这些物质可能会影响你的睡眠质量。
4. 限制白天小憩：如果你白天打盹过多，可能会影响到晚上的睡眠。
5. 运动：定期进行有氧运动，如散步、跑步或游泳，可以帮助改善睡眠质量。
6. 保持规律的作息时间：尽量每天在相同的时间上床睡觉和起床。
7. 限制使用电子设备：在睡前一小时内避免使用手机、电脑等电子设备，因为它们发出的蓝光可能会干扰你的睡眠。
8. 保持卧室整洁：一个干净、整洁的卧室有助于创造一个良好的睡眠环境。
9. 尝试渐进性入睡技巧：如数数法、听轻音乐或阅读轻松的文章。
10. 如果长时间无法入睡，不要强迫自己：起床做一些轻松的活动，如阅读或听音乐，直到感到困意再回到床上。

如果尝试了这些方法仍然无法改善你的睡眠质量，建议咨询专业医生或睡眠专家。
```



