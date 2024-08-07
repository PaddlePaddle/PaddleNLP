# 源2.0

## 1. 模型介绍

[源2.0](https://github.com/IEIT-Yuan/Yuan-2.0)是浪潮信息发布的新一代基础语言大模型。源2.0是在源1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同方面具备更强的理解能力。

目前源2.0对 PaddlePaddle 的适配仅支持数据并行和张量并行，后续功能正在开发中。

**支持模型权重:**

| Model             |
|-------------------|
| IEITYuan/Yuan2-2B |
| IEITYuan/Yuan2-51B |
| IEITYuan/Yuan2-102B |

## 2. 推理介绍

### · 2B

```python
from paddlenlp.transformers import  AutoModelForCausalLM, AutoTokenizer
model_path = "/pytorch/params" #from https://huggingface.co/IEITYuan/Yuan2-2B-hf
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,  dtype="bfloat16", convert_from_torch=True)
model.eval()
input_features = tokenizer("青岛推荐去哪玩？", return_tensors="pd")
print("问题：", tokenizer.batch_decode(input_features["input_ids"]))
outputs = model.generate(**input_features, do_sample=False, max_length=1024)
print("回答：", tokenizer.batch_decode(outputs[0]))
# <sep>青岛是中国著名的旅游城市，有许多著名的景点和活动。以下是一些值得推荐的地方：\n1. 栈桥：栈桥是青岛的象征之一，是八大关风景区的一部分。在这里可以欣赏到美丽的海岸线和壮观的城市风光。\n2. 青岛啤酒博物馆：这座博物馆位于崂山山顶上，可以欣赏到美丽的海景和壮观的城市景象。\n3. 八大关风景区：这里有许多知名的景点，如栈桥、音乐广场、青岛啤酒博物馆等。\n4. 青岛奥帆中心：这个帆船比赛已经在青岛成功举办了两届，是青岛市民的一项重要活动。\n5. 青岛老街：这里有丰富的历史和独特的建筑风格，还有许多小摊贩可以帮助游客找到纪念品。\n6. 海底世界：崂山是中国最大的海底岩洞，这里可以看到美丽的珊瑚和各种鱼类。\n7. 崂山风景名胜区：这个区域被联合国教科文组织列为世界遗产地，有丰富的自然和文化资源。\n无论您选择哪个地方，都可以欣赏到美丽的景色和体验到丰富的文化活动。希望您有机会去青岛旅游！<eod>
```

注意：在进行张量并行推理时，源2.0需要对权重文件进行特殊处理，在权重路径下找到源2.0的相关权重文件，使用本目录 utils 中提供的脚本，设置张量并行数、隐藏层的数目以及源权重和目标权重的路径，运行脚本即可完成权重文件的处理

### · 51B

由于 pdparams 格式的权重占用内存过大，需将权重保存为 safetensors 的格式。由于此权重较大，一般需要开启张量并行才能顺利推理，因此在保存之前需要完成上文提到的权重文件的特殊处理。

```python
from paddlenlp.transformers import  AutoModelForCausalLM
model_path = "/pytorch/params" #from https://huggingface.co/IEITYuan/Yuan2-51B-hf
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,  dtype="bfloat16", convert_from_torch=True) #如果显存不足就设置device='cpu'，并设置数据类型为float32
model.save_pretrained("/workspace/yuan_51B_tp8_safetensors", safe_serialization=True )
```

推理脚本如下 :

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --devices "0,1,2,3,4,5,6,7" \
    test_tp.py
```

test_tp.py :

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
tensor_parallel_rank = hcg.get_model_parallel_rank()
model_path = "/workspace/yuan_51B_tp8_safetensors"
tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(model_path, tensor_parallel_degree= 8, tensor_parallel_rank=tensor_parallel_rank, dtype="bfloat16")
model.eval()
input_features = tokenizer("青岛推荐去哪玩？", return_tensors="pd")
print("问题：", tokenizer.batch_decode(input_features["input_ids"]))
outputs = model.generate(**input_features, do_sample=False, max_length=1024)
print("回答：", tokenizer.batch_decode(outputs[0]))
```

### · 102B

与51B 的执行方法一致

## 3. 预训练介绍

请参考[LLM 全流程工具介绍](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)

## 4. 微调介绍

请参考[LLM 全流程工具介绍](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)
