## 6. 量化

量化算法可以将模型权重和激活转为更低比特数值类型表示，能够有效减少显存占用和计算开销。下面我们提供GPTQ和PaddleSlim自研的PTQ策略，分别实现WINT4和W8A8量化。更多技术细节详见[量化策略详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md)

### 6.1 环境安装
- PaddleSlim develop版本
- PaddlePaddle develop版本

### 6.2 数据准备

量化中默认使用训练集作为校正（Calibartion）数据集，开发集作为评估数据集。如果希望使用其他数据作为校正数据集，则在数据目录下新增`quant.json`文件，文件格式请参照精调训练数据格式。

### 6.3 PTQ 量化

```
python  finetune_generation.py ./llama/ptq_argument.json
```

### 6.4 GPTQ 量化

```
python  finetune_generation.py ./llama/gptq_argument.json
```

### 6.5 量化参数介绍

<summary>&emsp; 量化参数（QuantArgument）</summary><div>

- `quant_type`: PTQ,QAT量化类型，默认为A8W8。支持A8W8,WINT4，WINT8：A8W8指对激活（输入）进行INT8量化，对模型权重进行INT8量化；WINT4指仅对模型权重进行INT4量化，后续使用WeightOnly进行推理；WINT8指仅对模型权重进行INT8量化，后续使用WeightOnly进行推理。
- `do_ptq`: 是否进行PTQ量化，默认为False。
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
</div>


<summary>&emsp; 其他参数</summary><div>

- `per_device_train_batch_size`: 量化前向批大小，默认为8。量化过程只有模型前向，相比于普通训练需要显存较少。

- 更多参数详见精调参数介绍。

</div>