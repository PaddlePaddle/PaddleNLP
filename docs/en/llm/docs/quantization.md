# Large Model Quantization Tutorial

## 1. Algorithm Introduction

Large model quantization converts 16-bit and 32-bit floating-point model parameters or activations into 4-bit or 8-bit integers, effectively reducing model storage space and computational resource requirements while accelerating inference. The quantization algorithms include:

- **PTQ**. The adaptive PiecewiseSearchSmooth (PSS) quantization algorithm independently developed by the PaddleSlim team. Based on [SmoothQuant](https://arxiv.org/abs/2211.10438) and [Outlier Suppression+](https://arxiv.org/abs/2304.09145), it introduces the PieceWiseSearch parameter search algorithm and extends it to **all linear layers**, adjusting model weights and activation distributions to reduce subsequent A8W8 PTQ quantization loss.
- **GPTQ**. [GPTQ](https://arxiv.org/abs/2210.17323) is an industry-leading weight quantization algorithm that enables lossless 4-bit integer quantization of large model weights to improve inference speed.
- **AWQ**. [AWQ](https://arxiv.org/abs/2306.00978) is another industry-leading weight quantization algorithm that allows lossless 4-bit integer quantization of large model weights to enhance inference speed.

<div align="center">
    <img width="800" alt="llm" src="https://github.com/PaddlePaddle/PaddleNLP/assets/63761690/fe8f941b-4b35-48ca-814f-96533d7e24ce">
</div>
<div align="center">
    <font size ="1">
    PaddlePaddle Large Model Quantization Algorithms
     </font>
</div>

For more implementation details about PaddleSlim, please refer to the [Detailed Quantization Strategy Tutorial](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/quant/advanced_quantization.md).

## 2. Quick Start

### 2.1 Environment Setup

- PaddleSlim develop
- PaddlePaddle develop
- PaddleNLP develop

Clone the code repository to get started:

```shell
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    # pip install ./PaddleNLP to use the develop version
    cd PaddleNLP/llm
    # Navigate to the execution directory
```

### 2.2 Data Preparation

By default, the quantization process uses the training set as the calibration dataset and the development set as the evaluation dataset. For user convenience, we provide an example dataset [Advertisement Generation Dataset](https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz). To use other data as the calibration dataset, create a `quant.json` file in the data directory. Users can also format their own datasets by following our data structure. The supported data format requires each line to contain a dictionary with the following fields:

- `src`: `str, List(str)`, the model's input instruction (instruction), prompt, or task description.
`tgt`: `str, List(str)`, model's output.

Sample data:

```text
{"src": "type#dress*color#blue*style#fresh*pattern#bow", "tgt": "The dress features 3D bow decorations with blue ribbon accents, creating a full-bodied silhouette with layered details while infusing a touch of sweetness. This highlights the girl's fresh and charming appearance."}
...
```

In addition to the above dataset, you can also extract part of the C-Eval training dataset as calibration data. Use the following commands to download the data to the current folder and unzip it:
```shell
mkdir dataset
wget https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/c-eval/ceval-exam.zip
unzip ceval-exam.zip -d dataset/ceval
```
Use the following script and commands to extract C-Eval samples as calibration dataset:
```shell
cd llm/experimental/ceval/default
python prepare_data_for_ptq.py
```
The default generated calibration dataset is located at `dataset/ceval_ptq`.

### 2.3 PTQ Quantization

```shell
python run_quantization.py ./config/llama/ptq_argument.json
```

### 2.4 GPTQ Quantization

```shell
python run_quantization.py ./config/llama/gptq_argument.json
```

### 2.5 AWQ Quantization

```shell
python run_quantization.py ./config/llama/awq_argument.json
```

### 2.6 W8A8C8(INT8) Quantization

```shell
python run_quantization.py ./config/llama/ptq_c8_argument.json
```

### 2.7 W8A8(FP8) Quantization

```shell
python run_quantization.py ./config/llama/fp8_ptq_argument.json
```

### 2.8 Quantization Parameters

<summary>&emsp; Quantization Parameters (QuantArgument)</summary>

<div>

- `quant_type`: PTQ, QAT quantization type, default a8w8 (case insensitive). Supported types: a8w8, a8w8c8, a8w8_fp8, wint4/weight_only_int4, wint8/weight_only_int8:
    - a8w8: 8-bit quantization for activations (input) and INT8 quantization for model weights
    - a8w8c8: INT8 quantization for activations, weights, and kvcache
    - a8w8_fp8: FP8 quantization for activations and weights
    - wint4/weight_only_int4: INT4 weight-only quantization for model weights, followed by WeightOnly inference
    - wint8/weight_only_int8: INT8 weight-only quantization for model weights, followed by WeightOnly inference
- `fp8_type`: FP8 quantization type, specifies fp8 types for activation and weight, default `["e4m3","e4m3"]`.
- `do_ptq`
Whether to perform PTQ quantization, defaults to False.
- `weight_quant_method`: Weight quantization method. For INT8 quantization, options are groupwise or abs_max_channel_wise; for FP8 quantization, options are abs_max or avg.
- `act_quant_method`: Activation quantization method. For INT8, options are avg or abs_max; for FP8, options are abs_max or avg.
- `cachekv_quant_method`: kvcache quantization method, currently options are abs_max_headwise or avg_headwise.
- `ptq_step`: Number of steps for PTQ quantization (i.e., forward passes), defaults to 32.
- `shift`: Whether to apply [Shift strategy](https://arxiv.org/abs/2304.09145) before PTQ quantization, defaults to False. Requires `do_ptq` to be True.
- `shift_all_linear`: Whether to apply Shift to all Linear layers in the model. If True, Shift will be applied to Linear layers not in LayerNorm-Linear combinations, and two additional ops will be added, defaults to False.
- `shift_sampler`: Sampler for Shift strategy, defaults to none. Options: none, ema. "none" means directly use MinMax to calculate zero points in Shift; "ema" uses exponential moving average to calculate zero points.
- `shift_step`: Number of sampling steps for Shift (i.e., forward passes), defaults to 32.
- `smooth`: Whether to apply [SmoothQuant strategy](https://arxiv.org/abs/2211.10438) before PTQ quantization, defaults to False. Requires `do_ptq` to be True.
- `smooth_all_linears`: Whether to apply Smooth to all Linear layers in the model. If True, Smooth will be applied to Linear layers not in LayerNorm-Linear combinations, and two additional ops will be added, defaults to False.
- `smooth_sampler`: Sampler for Smooth strategy, defaults to none. Options: none, multi_step. "multi_step" saves multiple forward passes for computation, requiring more memory.
- `smooth_step`: Number of sampling steps for Smooth (i.e., forward passes), defaults to 32.
- `smooth_piecewise_search`: Whether to perform piecewise search in Smooth, defaults to False. Piecewise search divides activations into K segments based on value ranges, and searches for alpha and scale per segment.
- `smooth_k_piece`: Number of segments for piecewise search, defaults to 3. Empirical suggestion: 3 for 10B models, 6 for 100B models.
- `smooth_search_piece`: Whether to search for optimal number of segments during piecewise search, defaults to False. When True, it's recommended to set `smooth_k_piece` to 6. Segment number search is time-consuming; disable for faster Smooth process.
- `search_alpha_min`: Minimum alpha value for piecewise search, defaults to 0.2.
- `search_alpha_max`: Maximum alpha value for piecewise search, defaults to 0.8.
- `search_scale_min`
- `search_scale_min`: Minimum scale value for segmented search, default is 1.0.
- `search_scale_max`: Maximum scale value for segmented search, default is 5.0.
- `load_quant_model`: Whether to load quantized model, default is False. Used to validate the effect of quantized models. When set to True, weights will be loaded from output_dir. Enabling this requires setting `do_ptq` to False. If smooth or shift was used during quantization, the same configuration must be maintained when loading (shift_step/search_step can be set to 8). Note: Currently only supports loading in pdparams format. To use this feature, set `"unified_checkpoint": false`.
- `skip_list_names`: List of layer names to skip during quantization, default is empty list. Partial string matching can be used, e.g. ['down_proj'] will skip all ffn2 layers.
- `do_gptq`: Whether to perform GPTQ quantization. GPTQ implements WINT4 quantization, providing higher precision than regular PTQ but requiring longer quantization time. Default is False.
- `gptq_step`: Number of steps for GPTQ quantization (i.e., forward passes), default is 8.
- `do_awq`: Whether to perform AWQ quantization. AWQ implements WINT4 quantization, providing higher precision than regular PTQ. Default is False.
- `auto_clip`: Whether to perform automatic truncation value search and weight truncation during AWQ. Truncation improves quantized model precision but slows down search. Default is False.
- `autoclip_step`: Number of steps for AutoClip (i.e., forward passes). During sampling, data is concatenated for truncation value search, default is 8.

</div>

<summary>&emsp; Other Parameters</summary>
<div>

- `per_device_train_batch_size`: Batch size for quantization forward pass, default is 8. Quantization process only involves model forward passes, requiring less memory compared to regular training.

More parameters can be found in the [Fine-tuning document](./finetune.md).

</div>
