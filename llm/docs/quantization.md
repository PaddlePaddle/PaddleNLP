# 大模型量化教程

## 1.算法介绍

大模型量化将16位、32位浮点数的模型参数或激活量化为4位或8位整数能够有效降低模型存储空间和计算资源需求，同时加速推理速度。量化算法包含：

- **PTQ**。PaddleSlim 团队自研的自适应 PiecewiseSearchSmooth(PSS)量化算法，在[SmoothQuant](https://arxiv.org/abs/2211.10438)和[Outlier Suppression+](https://arxiv.org/abs/2304.09145)基础上
新增 PieceWiseSearch 参数搜索算法并将算法扩展至**所有线性层**，对模型权重和激活分布进行调整，减少后续 A8W8 PTQ 量化损失。
- **GPTQ**。[GPTQ](https://arxiv.org/abs/2210.17323)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。
- **AWQ**。[AWQ](https://arxiv.org/abs/2306.00978)是业界主流的权重量化算法，可以将大模型权重进行4位整数无损量化，提高模型推理速度。

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fe8f941b-4b35-48ca-814f-96533d7e24ce">
</div>
<div align="center">
    <font size ="1">
    飞桨大模型量化算法
     </font>
</div>

更多 PaddleSlim 实现细节详见[量化策略详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md)

## 2. 快速开始

### 2.1 环境准备

- PaddleSlim develop
- PaddlePaddle develop
- PaddleNLP  develop

git clone 代码到本地，即可开始。

```shell
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

```text
{"src": "类型#裙*颜色#蓝色*风格#清新*图案#蝴蝶结", "tgt": "裙身处采用立体蝴蝶结装饰辅以蓝色条带点缀，令衣身造型饱满富有层次的同时为其注入一丝甜美气息。将女孩清新娇俏的一面衬托而出。"}
...
```

除了上述数据集，也可以使用抽取 ceval 部分训练数据集作为校准数据。通过下述命令下载数据到当前文件夹并解压
```shell
mkdir dataset
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
unzip ceval-exam.zip -d dataset/ceval
```
使用下述脚本和命令抽取 C-Eval 样本作为校准数据集：
```shell
cd llm/experimental/ceval/default
python prepare_data_for_ptq.py
```
默认生成的校准数据集位于`dataset/ceval_ptq`。

### 2.3 PTQ 量化

```shell
python  run_quantization.py ./config/llama/ptq_argument.json
```

### 2.4 GPTQ 量化

```shell
python  run_quantization.py ./config/llama/gptq_argument.json
```

### 2.5 AWQ 量化

```shell
python  run_quantization.py ./config/llama/awq_argument.json
```

### 2.6 W8A8C8(INT8)量化

```shell
python  run_quantization.py ./config/llama/ptq_c8_argument.json
```

### 2.7 W8A8(FP8)量化

```shell
python  run_quantization.py ./config/llama/fp8_ptq_argument.json
```

### 2.8 量化参数介绍

<summary>&emsp; 量化参数（QuantArgument）</summary>

<div>

- `quant_type`: PTQ，QAT 量化类型，默认为 a8w8(不区分大小写)。支持 a8w8，a8w8c8，a8w8_fp8，wint4/weight_only_int4，wint8/weight_only_int8:
    - a8w8指对激活（输入）进行 8位量化，对模型权重进行 INT8量化
    - a8w8c8指对激活、权重、kvcache 进行 INT8量化
    - a8w8_fp8指对激活、权重进行 FP8量化
    - wint4/weight_only_int4指仅对模型权重进行 INT4量化，后续使用 WeightOnly 进行推理
    - wint8/weight_only_int8指仅对模型权重进行 INT8量化，后续使用 WeightOnly 进行推理
- `fp8_type`: FP8量化类型，指定 activatin，weight 的 fp8类型，默认为`["e4m3","e4m3"]`。
- `do_ptq`: 是否进行 PTQ 量化，默认为 False。
- `weight_quant_method`: 权重量化方式，INT8量化可选 groupwise 或者 abs_max_channel_wise，FP8量化可选 abs_max 或 avg。
- `act_quant_method`: 激活量化方式，INT8可选 avg 或者 abs_max，FP8量化可选 abs_max 或 avg。
- `cachekv_quant_method`: kvcache 量化方式，现可选 abs_max_headwise, avg_headwise。
- `ptq_step`: PTQ 量化步数，也即模型前向次数，默认为32。
- `shift`: 是否在 PTQ 量化前进行[Shift 策略](https://arxiv.org/abs/2304.09145)，默认为 False。使用 Shift 策略需要设`do_ptq`为 True。
- `shift_all_linear`: 是否对模型中所有 Linear 层应用 Shift，如果为 True，将会对非 LayerNorm-Linear 组合的 Linear 进行 Shift，并且添加两个 op，默认为 False
- `shift_sampler`: Shift 策略使用的 sampler，默认为 none。可选 none，ema：none 指直接利用 MinMax 计算 Shift 中的零点；ema 指使用指数平均计算 Shift 中零点。
- `shift_step`: Shift 采样步数，也即模型前向次数，默认为32。
- `smooth`: 是否在 PTQ 量化前进行[SmoothQuant 策略](https://arxiv.org/abs/2211.10438)，默认为 False。使用 Smooth 策略需要设`do_ptq`为 True。
- `smooth_all_linears`: 是否对模型中所有 Linear 层应用 Smooth，如果为 True，将会对非 LayerNorm-Linear 组合的 Linear 进行 Smooth，并且添加两个 op，默认为 False
- `smooth_sampler`: Smooth 策略使用的 sampler，默认为 none，可选 none，multi_step。multi_step 会保存多轮前向结果进行计算，需要更大的显存。
- `smooth_step`: Smooth 采样步数，也即模型前向次数，默认为32。
- `smooth_piecewise_search`: Smooth 是否进行分段搜索,默认为 False。分段搜索根据数值大小将激活分成 K 段，对于每一段进行 alpha 和 scale 的搜索。
- `smooth_k_piece`: 使用分段搜索功能时分段数量，默认为3。根据经验建议10B 模型设置为3，100B 模型设置为6。
- `smooth_search_piece`: 使用分段搜索功能时，是否搜索分段数量，默认为 False。设为 True 时，`smooth_k_piece`建议设为6，搜索分段数量耗时较长，如需加速 Smooth 过程建议关闭。
- `search_alpha_min`: 分段搜索时 alpha 最小值，默认为0.2。
- `search_alpha_max`: 分段搜索时 alpha 最大值，默认为0.8。
- `search_scale_min`: 分段搜索时 scale 最小值，默认为1.0。
- `search_scale_max`: 分段搜索时 scale 最大值，默认为5.0。
- `load_quant_model`: 是否加载量化模型，默认为 False。用于验证量化后的模型效果， 若设为 True，则从 output_dir 中加载权重。启动该过程需要设`do_ptq`为 False。如果量化时使用了 smooth 或 shift，加载时需要保持相同的配置（shift_step/search_step 可设为8）。注意，当前该函数只支持 pdparams 格式加载，若要使用该功能，设置`"unified_checkpoint": false`。
- `skip_list_names`: 需要量化跳过的层名称列表，默认为空列表。可以使用层名的部分字符串作为匹配，如['down_proj']表示跳过所有 ffn2层。
- `do_gptq`: 是否进行 GPTQ 量化，GPTQ 对模型进行 WINT4量化，相比于普通 PTQ 量化精度更高，量化时间较长。默认为 False。
- `gptq_step`: GPTQ 量化步数，也即模型前向次数，默认为8。
- `do_awq`: 是否进行 AWQ 量化，AWQ 对模型进行 WINT4量化，相比于普通 PTQ 量化精度更高。默认为 False。
- `auto_clip`: AWQ 时是否进行自动搜索截断值并对模型权重进行截断操作，截断操作有利于量化模型精度，但搜索速度较慢。默认为 False。
- `autoclip_step`: AutoClip 步数，也即模型前向次数，采样时默认 concat 每轮数据用来搜索截断值，默认为8。

</div>

<summary>&emsp; 其他参数</summary>
<div>

- `per_device_train_batch_size`: 量化前向批大小，默认为8。量化过程只有模型前向，相比于普通训练需要显存较少。

更多参数详见[精调文档](./finetune.md)中精调参数介绍。

</div>
