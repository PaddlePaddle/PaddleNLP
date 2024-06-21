# 大模型量化教程

## 1.算法介绍

大模型量化将16位、32位浮点数的模型参数或激活量化为4位或8位整数能够有效降低模型存储空间和计算资源需求，同时加速推理速度。工具链量化算法包含：
- **PTQ**。PaddleSlim 团队自研的自适应PiecewiseSearchSmooth(PSS)量化算法，在[SmoothQuant](https://arxiv.org/abs/2211.10438)和[Outlier Suppression+](https://arxiv.org/abs/2304.09145)基础上
新增PieceWiseSearch参数搜索算法并将算法扩展至**所有线性层**，对模型权重和激活分布进行调整，减少后续A8W8 PTQ量化损失。


- **GPTQ**。[GPTQ](https://arxiv.org/abs/2210.17323)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。

- **AWQ**。[GPTQ](https://arxiv.org/abs/2306.00978)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fe8f941b-4b35-48ca-814f-96533d7e24ce">
</div>
<div align="center">
    <font size ="1">
    飞桨大模型量化算法
     </font>
</div>

更多PaddleSlim实现细节详见[量化策略详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md)



## 2. 快速开始

### 2.1 环境准备

- PaddleSlim develop
- PaddlePaddle develop
- PaddleNLP  develop

git clone 代码到本地，即可开始。

```bash
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP 使用develop版本
    cd PaddleNLP/llm
    # 到达运行目录
```

### 2.2 数据准备

量化中默认使用训练集作为校正（Calibartion）数据集，开发集作为评估数据集。为了方便用户测试，我们也提供示例数据集[广告生成数据集](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz)。如果希望使用其他数据作为校正数据集，则在数据目录下新增`quant.json`文件，用户也可以仿照数据集的格式制作自己的数据集进行精调。我们支持的数据格式是每行包含一个字典，每个字典包含以下字段：

- `src` : `str, List(str)`, 模型的输入指令（instruction）、提示（prompt），模型应该执行的任务。
- `tgt` : `str, List(str)`, 模型的输出。

样例数据：
```
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```


### 2.3 PTQ 量化

```
python  run_finetune.py ./config/llama/ptq_argument.json
```

### 2.4 GPTQ 量化

```
python  run_finetune.py ./config/llama/gptq_argument.json
```

### 2.5 AWQ 量化

```
python  run_finetune.py ./config/llama/awq_argument.json
```

### 2.6 量化参数介绍

<summary>&emsp; 量化参数（QuantArgument）</summary><div>

- `quant_type`: PTQ,QAT量化类型，默认为A8W8。支持A8W8,WINT4，WINT8：A8W8指对激活（输入）进行INT8量化，对模型权重进行INT8量化；WINT4指仅对模型权重进行INT4量化，后续使用WeightOnly进行推理；WINT8指仅对模型权重进行INT8量化，后续使用WeightOnly进行推理。
- `do_ptq`: 是否进行PTQ量化，默认为False。
- `weight_quant_method`: 权重量化方式，现可选groupwise或者abs_max_channel_wise。
- `ptq_step`: PTQ量化步数，也即模型前向次数，默认为32。
- `shift`: 是否在PTQ量化前进行[Shift策略](https://arxiv.org/abs/2304.09145)，默认为False。使用Shift策略需要设`do_ptq`为True。
- `shift_all_linear`: 是否对模型中所有Linear层应用Shift，如果为True，将会对非LayerNorm-Linear组合的Linear进行Shift，并且添加两个op，默认为False
- `shift_sampler`: Shift策略使用的sampler，默认为none。可选none，ema：none指直接利用MinMax计算Shift中的零点；ema指使用指数平均计算Shift中零点。
- `shift_step`: Shift采样步数，也即模型前向次数，默认为32。
- `smooth`: 是否在PTQ量化前进行[SmoothQuant策略](https://arxiv.org/abs/2211.10438)，默认为False。使用Smooth策略需要设`do_ptq`为True。
- `smooth_all_linears`: 是否对模型中所有Linear层应用Smooth，如果为True，将会对非LayerNorm-Linear组合的Linear进行Smooth，并且添加两个op，默认为False
- `smooth_sampler`: Smooth策略使用的sampler，默认为none，可选none，multi_step。multi_step会保存多轮前向结果进行计算，需要更大的显存。
- `smooth_step`: Smooth采样步数，也即模型前向次数，默认为32。
- `smooth_piecewise_search`: Smooth是否进行分段搜索,默认为False。分段搜索根据数值大小将激活分成K段，对于每一段进行alhpa和scale的搜索。
- `smooth_k_piece`: 使用分段搜索功能时分段数量，默认为3。根据经验建议10B模型设置为3，100B模型设置为6。
- `smooth_search_piece`: 使用分段搜索功能时，是否搜索分段数量，默认为False。设为True时，`smooth_k_piece`建议设为6，搜索分段数量耗时较长，如需加速Smooth过程建议关闭。
- `do_gptq`: 是否进行GPTQ量化，GPTQ对模型进行WINT4量化，相比于普通PTQ量化精度更高，量化时间较长。默认为False。
- `gptq_step`: GPTQ量化步数，也即模型前向次数，默认为8。
- `do_awq`: 是否进行AWQ量化，AWQ对模型进行WINT4量化，相比于普通PTQ量化精度更高。默认为False。
- `auto_clip`: AWQ时是否进行自动搜索截断值并对模型权重进行截断操作，截断操作有利于量化模型精度，但搜索速度较慢。默认为False。
- `autoclip_step`: AutoClip步数，也即模型前向次数，采样时默认concat每轮数据用来搜索截断值，默认为8。


</div>


<summary>&emsp; 其他参数</summary><div>

- `per_device_train_batch_size`: 量化前向批大小，默认为8。量化过程只有模型前向，相比于普通训练需要显存较少。

更多参数详见[精调文档](./finetune.md)中精调参数介绍。

</div>
