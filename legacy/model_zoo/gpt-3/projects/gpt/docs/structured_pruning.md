# GPT模型结构化稀疏

本项目对语言模型 GPT 进行结构化稀疏（以下简称稀疏）。在 GPT 模型中，我们对 fused-qkv、out-linear、ffn1 和 ffn2 四层的权重进行了通道稀疏，其中，fused-qkv 和 ffn1 是在输出通道进行稀疏，out-linear 和 ffn2 是在输入通道进行稀疏。如果您需要自定义稀疏的层和通道，可以通过重写 ppfleetx/utils/compression_helper.py 中的 get_pruned_params() 函数实现。

下面是本例涉及的文件及说明：

```text
.
├── prune_gpt_345M_single_card.sh            # 单卡345M稀疏训练入口
├── eval_prune_gpt_345M_single_card.sh       # 单卡345M稀疏模型验证入口
├── export_prune_gpt_345M_single_card.sh     # 单卡345M稀疏模型导出入口
```


### 环境依赖和数据准备
环境依赖和数据准备请参考[GPT训练文档](./README.md)。

特别的，本示例需要依赖 PaddleSlim develop版本。安装命令如下：

```shell
git clone https://github.com/PaddlePaddle/PaddleSlim.git & cd PaddleSlim
pip install -r requirements.txt
python setup.py install
```


### 预训练模型准备
稀疏训练需加载[GPT-345M](https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz) 的预训练模型。

**预训练模型下载命令**
```shell
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M.tar.gz
tar xf GPT_345M.tar.gz
```

### 稀疏训练

- [345M模型稀疏训练](../prune_gpt_345M_single_card.sh)

快速启动：
```shell
bash ./projects/gpt/prune_gpt_345M_single_card.sh
```

或如下启动：
```shell
export CUDA_VISIBLE_DEVICES=0
python ./tools/train.py \
    -c ./ppfleetx/configs/nlp/gpt/prune_gpt_345M_single_card.yaml \
    -o Engine.max_steps=100000 \
    -o Optimizer.lr.decay_steps=72000 \
    -o Optimizer.weight_decay=0.0 \
    -o Optimizer.lr.max_lr=2.5e-5 \
    -o Optimizer.lr.min_lr=5.0e-6 \
    -o Compress.pretrained='./PaddleFleetX_GPT_345M_220826'

```

### 模型验证
```shell
# 下载验证数据
wget https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl
export CUDA_VISIBLE_DEVICES=0
python ./tools/eval.py \
    -c ./ppfleetx/configs/nlp/gpt/eval_pruned_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./output' \
    -o Offline_Eval.eval_path=./lambada_test.jsonl \
    -o Offline_Eval.cloze_eval=True
```

### 模型导出
```shell
export CUDA_VISIBLE_DEVICES=0
python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/generation_pruned_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./output'
```
