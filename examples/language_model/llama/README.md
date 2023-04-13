# LLaMA inplementation

**目录**
- [1. 模型下载及权重转换](#1)
- [2. 微调](#2)
- [3. 动转静](#3)
- [4. 推理](#4)

<a name="1"></a>

## 模型加载：

```python
from tokenizer import LLaMATokenizer
from modeling import LLaMAForCausalLM

tokenizer = LLaMATokenizer.from_pretrained("facebook/llama-7b")
model = LLaMAForCausalLM.from_pretrained("facebook/llama-7b", load_state_as_np=True)
```

<a name="2"></a>

## 微调

```shell
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3" finetune_generation.py \
    --model_name_or_path facebook/llama-7b \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --tensor_parallel_degree 4 \
    --overwrite_output_dir \
    --output_dir ./checkpoints/ \
    --logging_steps 10 \
    --fp16 \
    --fp16_opt_level O2 \
    --gradient_accumulation_steps 32 \
    --recompute \
    --learning_rate 3e-5 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --warmup_steps 20
```

<a name="3"></a>

## 动转静

```shell
python export_generation_model.py \
    --model_path checkpoints/ \
    --output_path inference/llama
```

<a name="4"></a>

## 推理

```shell
python infer_generation.py \
    --model_dir inference \
    --model_prefix llama
```

结果：

```text
answer: linebacker context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: What was von Miller's job title?
--------------------
answer: five context: The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles. </s>

question: How many total tackles did von Miller get in the Super Bowl?
--------------------
```
