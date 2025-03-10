[简体中文🀄](./README.md) | **English🌎**

<p align="center">
  <img src="https://user-images.githubusercontent.com/1371212/175816733-8ec25eb0-9af3-4380-9218-27c154518258.png" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="https://paddlenlp.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/paddlenlp/badge/?version=latest">
    <a href="https://github.com/PaddlePaddle/PaddleNLP/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleNLP?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleNLP?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleNLP?color=3af"></a>
    <a href="https://pypi.org/project/paddlenlp/"><img src="https://img.shields.io/pypi/dm/paddlenlp?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleNLP?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleNLP/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleNLP?color=ccf"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
</p>

<h4 align="center">
    <a href=#Features> Features </a> |
    <a href=#Support-Models> Supported Models </a> |
    <a href=#Installation> Installation </a> |
    <a href=#Quick-start> Quick Start </a> |
    <a href=#community> Community </a>
</h4>

**PaddleNLP** is a Large Language Model (LLM) development suite based on the PaddlePaddle deep learning framework, supporting efficient large model training, lossless compression, and high-performance inference on various hardware devices. With its **simplicity** and **ultimate performance**, PaddleNLP is dedicated to helping developers achieve efficient industrial applications of large models.

## News 📢

* **2024.06.27 [PaddleNLP v3.0 Beta](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v3.0.0-beta0)**：Embrace large models and experience a complete upgrade. With a unified large model suite, we achieve full-process access to domestically produced computing chips. We fully support industrial-level application processes for large models, such as PaddlePaddle's 4D parallel configuration, efficient fine-tuning strategies, efficient alignment algorithms, and high-performance reasoning. Our developed RsLoRA+ algorithm, full checkpoint storage mechanism Unified Checkpoint, and generalized support for FastFNN and FusedQKV all contribute to the training and inference of large models. We continuously support updates to mainstream models for providing efficient solutions.

* **2024.04.24 [PaddleNLP v2.8](https://github.com/PaddlePaddle/PaddleNLP/releases/tag/v2.8.0)**：Our self-developed RsLoRA+ algorithm with extreme convergence significantly improves the convergence speed and training effectiveness of PEFT training. By introducing high-performance generation acceleration into the RLHF PPO algorithm, we have broken through the generation speed bottleneck in PPO training, achieving a significant lead in PPO training performance. We generally support multiple large model training performance optimization methods such as FastFFN and FusedQKV, making large model training faster and more stable.

## Features

### <a href=#Integrated training and inference on multiple hardware platforms> 🔧 Integrated training and inference on multiple hardware platforms </a>

Our development suit supports large model training and inference on multiple hardware platforms, including NVIDIA GPUs, Kunlun XPUs, Ascend NPUs, Enflame GCUs, and Hygon DCUs. The toolkit's interface allows for quick hardware switching, significantly reducing research and development costs associated with hardware transitions.

### <a href=Efficient and easy-to-use pre-training> 🚀 Efficient and easy-to-use pre-training </a>

We support 4D high-performance training with data parallelism, sharding parallelism, tensor parallelism, and pipeline parallelism. The Trainer supports configurable distributed strategies, reducing the cost associated with complex distributed combinations. The Unified Checkpoint large model storage format supports dynamic scaling of model parameter distribution during training, thereby reducing the migration cost caused by hardware switching.

### <a href=#Efficient fine-tuning> 🤗 Efficient fine-tuning </a>

The fine-tuning algorithms are deeply integrated with zero-padding data streams and high-performance FlashMask operators, reducing invalid data padding and computation during training, and significantly improving the throughput of fine-tuning training.

### <a href=#Lossless compression and high-performance inference> 🎛️ Lossless compression and high-performance inference </a>

The high-performance inference module of the large model toolkit incorporates dynamic insertion and operator fusion strategies throughout the entire process, greatly accelerating parallel inference speed. The underlying implementation details are encapsulated, enabling out-of-the-box high-performance parallel inference capabilities.

## Documentation
For detailed documentation, visit the [PaddleNLP Documentation](https://paddlenlp.readthedocs.io/).

------------------------------------------------------------------------------------------

## Support Models

Detailed list 👉 [Supported Model List](https://github.com/PaddlePaddle/PaddleNLP/issues/8663)

## Installation

### Prerequisites

* python >= 3.8
* paddlepaddle >= 3.0.0b0

### Pip Installation

```shell
pip install --upgrade paddlenlp==3.0.0b3
```

or you can install the latest develop branch code with the following command:

```shell
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

More information about PaddlePaddle installation please refer to [PaddlePaddle's Website](https://www.paddlepaddle.org.cn).

------------------------------------------------------------------------------------------

## Quick Start

### Text generation with large language model

PaddleNLP provides a convenient and easy-to-use Auto API, which can quickly load models and Tokenizers. Here, we use the `Qwen/Qwen2-0.5B` large model as an example for text generation:

```python
>>> from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
>>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
>>> input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
>>> outputs = model.generate(**input_features, max_length=128)
>>> print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
['我是一个AI语言模型，我可以回答各种问题，包括但不限于：天气、新闻、历史、文化、科学、教育、娱乐等。请问您有什么需要了解的吗？']
```

### Pre-training for large language model

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # if cloned or downloaded, can skip this step
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_pretrain.py ./config/llama/pretrain_argument.json
```

### SFT finetuning forlarge language model

```shell
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # if cloned or downloaded, can skip this step
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
cd .. # change folder to PaddleNLP/llm
python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" run_finetune.py ./config/llama/sft_argument.json
```

For more steps in the entire large model process, please refer to the[Large Model Full-Process Suite](./llm).

For more PaddleNLP content, please refer to:

* [Model Library](./slm/model_zoo)，which includes end-to-end usage of high-quality pre-trained models.
* [Multi-scenario Examples](./slm/examples)，to understand how to use PaddleNLP to solve various NLP technical problems, including basic techniques, system applications, and extended applications.
* [Interactive Tutorial](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/574995)，to quickly learn PaddleNLP on the free computing platform AI Studio.

------------------------------------------------------------------------------------------

## Community

### Slack

To connect with other users and contributors, welcome to join our [Slack channel](https://paddlenlp.slack.com/).

### WeChat

Scan the QR code below with your Wechat⬇️. You can access to official technical exchange group. Look forward to your participation.

<div align="center">
    <img src="https://github.com/user-attachments/assets/3a58cc9f-69c7-4ccb-b6f5-73e966b8051a" width="150" height="150" />
</div>

## Citation

If you find PaddleNLP useful in your research, please consider citing

```bibtext
@misc{=paddlenlp,
    title={PaddleNLP: An Easy-to-use and High Performance NLP Library},
    author={PaddleNLP Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleNLP}},
    year={2021}
}
```

## Acknowledge

We have borrowed from Hugging Face's [Transformers](https://github.com/huggingface/transformers)🤗 excellent design on pretrained models usage, and we would like to express our gratitude to the authors of Hugging Face and its open source community.

## License

PaddleNLP is provided under the [Apache-2.0 License](./LICENSE).
