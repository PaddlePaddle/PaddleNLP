# PaddleNLP 大模型新手指南-量化

我们在 Ai Studio 上同步公开了项目，也可以点击[链接](https://aistudio.baidu.com/projectdetail/9185161)在线体验大模型量化。

## 1. 量化
在大模型预训练的指南中在遇到显存不足时我们曾经提到了模型量化，大模型量化（Quantization）是一种重要的模型压缩技术，它通过将模型中的参数从高精度（如 FP32）转换为低精度（如 INT8 或 FP16），以减小模型的体积、降低显存占用、加快推理速度，同时保持较好的模型性能。

模型量化的目标是在尽量不损失精度的前提下，让模型尽量更小更快。常见的量化类型如下：

* 权重量化（Weight Quantization）：对模型参数进行压缩，例如将模型的参数值由 FP32转为 INT8。
* 激活量化（Activation Quantization）：对推理时中间激活值量化。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fe8f941b-4b35-48ca-814f-96533d7e24ce">
</div>
<div align="center">
    <font size ="1">
    飞桨大模型量化算法
     </font>
</div>

在进行量化之前，需要安装 PaddlePaddle、PaddleNLP 与 PaddleSlim。

安装最新版本的 paddlepaddle 与 paddlenlp:
```shell
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

安装 develop 版本的 PaddleSlim：
```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim

python setup.py install
```

## 2. GPTQ
GPTQ 是一种后训练量化（PTQ）方法，基本思想就是直接对模型进行量化压缩，然后再通过一些校准数据验证压缩以后对模型的影响较小。GPTQ 的流程如下：
* 初始状态：未量化的模型、一个校准数据集
* 输入校准数据，获得每层的激活值
* 对每一层进行量化
* 选择对输出误差影响最小的量化值

校正数据集使用与 SFT 类似的数据集，本次使用[CEVAL](https://cevalbenchmark.com/index_zh.html)数据集，这是一个中文选择题数据集，样例如下：
```
id: 1
question: 25 °C时，将pH=2的强酸溶液与pH=13的强碱溶液混合，所得混合液的pH=11，则强酸溶液与强碱溶液 的体积比是(忽略混合后溶液的体积变化)____
A: 11:1
B: 9:1
C: 1:11
D: 1:9
answer: B
explanation:
1. pH=13的强碱溶液中c(OH-)=0.1mol/L, pH=2的强酸溶液中c(H+)=0.01mol/L，酸碱混合后pH=11，即c(OH-)=0.001mol/L。
2. 设强酸和强碱溶液的体积分别为x和y，则：c(OH-)=(0.1y-0.01x)/(x+y)=0.001，解得x:y=9:1。
```

下载并解压数据集：
```shell
cd ~/PaddleNLP/
mkdir dataset
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/c-eval/ceval-exam.zip
unzip ceval-exam.zip -d dataset/ceval
```

抽取 C-Eval 样本作为校准数据集：
```shell
cd llm/experimental/ceval/default
python prepare_data_for_ptq.py
```

我们需要在本地新建一个配置文件 ```gptq_argument.json```，将下面的内容粘贴进去：
```
{
    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "eval_accumulation_steps":16,
    "src_length": 512,
    "max_length": 1024,
    "fp16": false,
    "fp16_opt_level": "O0",
    "dataset_name_or_path": "/home/aistudio/PaddleNLP/dataset/ceval_ptq_test",
    "output_dir": "./checkpoints/gptq_ckpts",
    "do_eval": true,
    "eval_with_do_generation": false,
    "do_gptq": true,
    "unified_checkpoint": true,
    "gptq_step": 4
  }
```
对比 ```PaddleNLP/llm/config/llama/gptq_argument.json```，做出了如下几个关键改动：
1. 将模型由 LLaMa3-7B 更换为更小的 Qwen2.5-0.5B-Instruct，避免超出显存限制；
2. 将数据集地址更改为绝对地址，防止如下报错：
```
huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: './data'.
```
3. 禁用了 bf16与 fp16：V100及之前的显卡无法使用 bf16，使用 fp16也会导致量化出现 NaN。
```
[    INFO] - GPTQ layer: 6, qwen2.layers.0.mlp.up_proj
[    INFO] -   Num examples = 128
[    INFO] -   Total GPTQ steps = 4
[    INFO] -   Pre device batch size = 8
[    INFO] -   Total Batch size = 8
quant linear_5
time 0.54
error nan
```

执行 GPTQ 量化:
```shell
cd ~/PaddleNLP/llm/
python run_quantization.py gptq_argument.json
```

看到如下的提示，并且没有出现 NaN，量化就正常运行了：
```
[    INFO] - GPTQ layer: 118, qwen2.layers.16.mlp.up_proj
[    INFO] -   Num examples = 128
[    INFO] -   Total GPTQ steps = 4
[    INFO] -   Pre device batch size = 8
[    INFO] -   Total Batch size = 8
quant linear_117
time 0.56
error 27387.02734375
```

最后结果：
```
[    INFO] - ***** eval metrics *****
[    INFO] -   eval_accuracy           =     0.8314
[    INFO] -   eval_loss               =     0.9088
[    INFO] -   eval_ppl                =     2.4813
[    INFO] -   eval_runtime            = 0:00:14.00
[    INFO] -   eval_samples_per_second =     9.1368
[    INFO] -   eval_steps_per_second   =     1.1421
```

## 3. 其他量化方法
飞桨大模型套件还支持其他的一些量化方法，如[AWQ](https://arxiv.org/abs/2306.00978)及 PaddleSlim 团队自研的自适应 PiecewiseSearchSmooth(PSS)量化算法 PTQ。

<div align="center">
    <img width="500" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/37530985/969b62db-9692-4d50-b91a-85cff305d153">
</div>
<div align="center">
    <font size ="1">
    飞桨 W4和 W8A8量化算法效果展示
     </font>
</div>
<div align="center">
    <img width="300" alt="llm" src="https://github.com/user-attachments/assets/ab8d04ba-d589-4f54-acf1-b00c0fd9159e">
</div>
<div align="center">
    <font size ="1">
    飞桨 W8A8C8和 FP8量化效果展示
     </font>
</div>

PTQ 量化启动命令参考
```shell
python run_quantization.py ./config/llama/ptq_argument.json
```



awq 量化启动命令参考
```shell
python run_quantization.py ./config/llama/awq_argument.json
```

W8A8C8(INT)量化启动命令参考
```shell
python run_quantization.py ./config/llama/ptq_c8_argument.json
```
W8A8(FP8)量化启动命令参考
```shell
python run_quantization.py ./config/llama/fp8_ptq_argument.json
```

更多技术细节和模型量化使用详见[量化文档](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/docs/quantization.md)。
