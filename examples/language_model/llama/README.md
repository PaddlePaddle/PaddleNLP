# LLaMA inplementation

**目录**
- [1. 模型下载及权重转换](#1)
- [2. 微调](#2)
- [3. 动转静](#3)
- [4. 推理](#4)

<a name="1"></a>

## 模型下载及权重转换

[LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) Paddle实现，填写 [google form](https://forms.gle/jk851eBVbX1m5TAv5) 申请获得模型下载权限。

通过以下方式获得paddle模型权重：

```shell
python convert_llama_weights_to_pd.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

模型加载：

```python
from tokenizer import LLaMATokenizer
from modeling import LLaMAForCausalLM

tokenizer = LLaMATokenizer.from_pretrained("./llama-7b/")
model = LLaMAForCausalLM.from_pretrained("./llama-7b/", load_state_as_np=True)
```

<a name="2"></a>

## 微调

```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" finetune_generation.py \
    --model_name_or_path ./llama-7b \
    --output_dir output \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 32 \
    --sharding_degree 4 \
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --global_batch_size 4 \
    --micro_batch_size 4 \
    --output_dir output \
    --max_steps 200 \
    --use_pure_fp16 True \
    --device gpu
```

<a name="3"></a>

## 动转静

```shell
python export_generation_model.py \
    --model_path output/200/splits_mp_02_sharding_04 \
    --output_path inference/llama
```

<a name="4"></a>

## 推理

```shell
python infer_generation.py \
    --model_dir inference \
    --model_prefix llama
```
