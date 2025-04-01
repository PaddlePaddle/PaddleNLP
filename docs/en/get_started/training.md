# Quick Start for Large Model Training

## Large Model SFT Fine-Tuning

Rapid fine-tuning: You can now start the full fine-tuning process for large models by simply copying these few lines of code.

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

Additionally, we provide higher-performance fine-tuning scripts. Clone PaddleNLP to start training.

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # Skip if already cloned
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/datasets/examples/AdvertiseGen.tar.gz && tar -zxvf AdvertiseGen.tar.gz
cd .. # Change folder to PaddleNLP/llm
python -u run_finetune.py ./config/qwen/sft_argument_0p5b.json
```

## Large Model Pre-training

If you want to train your model from random initialization or continue training with additional corpus on an existing model, we provide high-performance pre-training scripts. Clone to start training.

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git && cd PaddleNLP # Skip if already cloned
mkdir -p llm/data && cd llm/data
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
cd .. # Change folder to PaddleNLP/llm
python -u run_pretrain.py ./config/qwen/pretrain_argument_0p5b.json
```
