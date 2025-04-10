# Yuan 2.0

## 1. Model Introduction

[Yuan 2.0](https://github.com/IEIT-Yuan/Yuan-2.0) is a new generation of fundamental language models released by Inspur Information. Based on Yuan 1.0, Yuan 2.0 leverages more diverse high-quality pretraining data and instruction fine-tuning datasets, endowing the model with enhanced comprehension capabilities across different aspects including semantics, mathematics, reasoning, code, and knowledge.

Currently, Yuan 2.0's adaptation to PaddlePaddle only supports data parallelism and tensor parallelism. Subsequent features are under development.

**Supported Model Weights:**

| Model             |
|-------------------|
| IEITYuan/Yuan2-2B |
| IEITYuan/Yuan2-51B |
| IEITYuan/Yuan2-102B |

## 2. Inference Guide

### · 2B

Inference script:

```python
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "IEITYuan/Yuan2-2B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype="bfloat16")
model.eval()
input_features = tokenizer("Qingdao travel recommendations?", return_tensors="pd")
print("Question:", tokenizer.batch_decode(input_features["input_ids"]))
outputs = model.generate(**input_features, do_sample=False, max_length=1024)
print("Answer:", tokenizer.batch_decode(outputs[0]))
# <sep>Qingdao is a famous tourist city in China with many renowned attractions and activities. Here are some recommended places:\n1. Zhanqiao Pier: A symbol of Qingdao, part of the Badaguan Scenic Area. Here you can enjoy beautiful coastlines and spectacular city views.\n2. Tsingtao Beer Museum: Located on Laoshan Mountain, offering panoramic sea views and cityscapes.\n3. Badaguan Scenic Area: Contains famous landmarks like Zhanqiao Pier, Music Square, and Tsingtao Beer Museum.\n4. Qingdao Olympic Sailing Center: Hosted two successful sailing competitions, an important event for Qingdao citizens.\n5. Qingdao Old Streets: Feature rich history and unique architecture, with small vendors offering souvenirs.\n6. Underwater World: Discover beautiful corals and diverse fish species in China's largest underwater cave on Laoshan Mountain.\n7. Laoshan Scenic Area: A UNESCO World Heritage site with abundant natural and cultural resources.\nWhichever you choose, you'll experience breathtaking scenery and rich cultural activities. Hope you enjoy your trip to Qingdao!<eod>
```

### · 51B

Inference script:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    test_tp.py
```

test_tp.py:

```python
# (Original code block remains unchanged)
# The actual content should be filled here according to specific implementation
```
```python
from paddle.distributed import fleet
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1,
    "mp_degree": 8,
    "pp_degree": 1,
    "sharding_degree": 1,
}

fleet.init(is_collective=True, strategy=strategy)
hcg = fleet.get_hybrid_communicate_group()
tensor_parallel_rank = hcg.get_model_parallel_parallel_rank()

model_path = "IEITYuan/Yuan2-51B"
tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    tensor_parallel_degree=8,
    tensor_parallel_rank=tensor_parallel_rank,
    dtype="bfloat16"
)
model.eval()

input_features = tokenizer("Xiamen travel recommendations?", return_tensors="pd")
print("Question:", tokenizer.batch_decode(input_features["input_ids"]))

outputs = model.generate(**input_features, do_sample=False, max_length=1024)
print("Answer:", tokenizer.batch_decode(outputs[0]))
```
### · 102B

The inference script is consistent with the 51B model.

## 3. Pretraining Introduction

Please refer to [LLM Full-Process Tool Introduction](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)

## 4. Fine-tuning Introduction

Please refer to [LLM Full-Process Tool Introduction](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)
