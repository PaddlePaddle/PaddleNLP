# 大模型训练快速上手

## 大模型 SFT 精调

快速微调，您现在只需复制者几行代码，即可开启大模型全量微调流程。
```python
from paddlenlp.trl import SFTConfig, SFTTrainer
from datasets import load_dataset

dataset = load_dataset("ZHUI/alpaca_demo", split="train")

training_args = SFTConfig(output_dir="Qwen/Qwen2.5-0.5B-SFT", device="gpu")
trainer = SFTTrainer(
    args=training_args,
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=dataset,
)
trainer.train()
```

同时，我们还提供了更加高性能微调脚本，clone paddlenlp 即可开启训练。

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # 如已clone或下载PaddleNLP可跳过
cd llm && wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json
```


## 大模型预训练
如果你想从随机初始化训练您的模型，或者在原来模型的基础上，加入额外语料继续训练。我们提供了高性能的预训练脚本。git clone 即可开始训练。
```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # 如已clone或下载PaddleNLP可跳过
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. # change folder to PaddleNLP/llm
python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json
```
