
# 推理部署

模型训练完成后，可使用飞桨高性能推理引擎Paddle Inference通过如下方式进行推理部署。

## 1. 模型导出

### 1.1 非量化模型导出

以`GPT-3(345M)`模型为例，通过如下方式下载PaddleFleetX发布的训练好的权重。若你已下载或使用训练过程中的权重，可跳过此步。

```bash
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_FP16.tar.gz
tar -zxvf GPT_345M_FP16.tar.gz
```

通过如下方式进行推理模型导出
导出单卡`GPT-3(345M)`模型：
```bash
sh projects/gpt/auto_export_gpt_345M_single_card.sh
```

导出单卡`GPT-3(6.7B)`模型：
```bash
sh projects/gpt/auto_export_gpt_6.7B_mp1.sh
```

导出8卡`GPT-3(175B)`模型：
```bash
sh projects/gpt/auto_export_gpt_175B_mp8.sh
```

### 1.2 量化模型导出

导出单卡`GPT-3(345M)`量化模型：

```shell
# 为了方便快速体验，这里给出345M量化训练的模型，若已有量化模型，则无需下载
wget https://paddlefleetx.bj.bcebos.com/model/nlp/gpt/GPT_345M_QAT_wo_analysis.tar
tar xf GPT_345M_QAT_wo_analysis.tar

export CUDA_VISIBLE_DEVICES=0
python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/generation_qat_gpt_345M_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0 \
    -o Engine.save_load.ckpt_dir='./GPT_345M_QAT_wo_analysis/'
```

导出单卡`GPT-3(6.7B)`量化模型：

```shell
export CUDA_VISIBLE_DEVICES=0
python ./tools/export.py \
    -c ./ppfleetx/configs/nlp/gpt/generation_qat_gpt_6.7B_single_card.yaml \
    -o Model.hidden_dropout_prob=0.0 \
    -o Model.attention_probs_dropout_prob=0.0
```

## 2. 推理部署

模型导出后，可通过`tasks/gpt/inference.py`脚本进行推理部署。

单卡推理
```bash
bash projects/gpt/inference_gpt_single_card.sh
```

多卡推理(以8卡为例)

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MP=8
bash projects/gpt/inference_gpt_multigpu.sh
```


## 3. Benchmark
- 导出模型
修改配置文件
PaddleFleetX/ppfleetx/configs/nlp/gpt/auto/generation_gpt_6.7B_mp1.yaml，将`Generation/early_finish`选项设置为False(关闭提前终止，仅适用于测速场景)

执行导出
```bash
sh projects/gpt/auto_export_gpt_6.7B_mp1.sh
```
如果打开了topp_sampling,则需要安装自定义算子：
```bash
cd ppfleetx/ops && python setup_cuda.py install && cd ../..
```

- 运行benchmark脚本
```
bash projects/gpt/run_benchmark.sh
```

| 模型          | 输入长度 | 输出长度 | batch size | GPU卡数 | FP16推理时延 | INT8推理时延 |
| :------------ | :------: | :------: | :--------: | :-----: | :----------: | :----------: |
| GPT-3(345M)   |    128   |    8     |     1      |    1    |   18.91ms    |   18.30ms    |
| GPT-3(345M)   |    128   |    8     |     2      |    1    |   20.01ms    |   18.88ms    |
| GPT-3(345M)   |    128   |    8     |     4      |    1    |   20.83ms    |   20.77ms    |
| GPT-3(345M)   |    128   |    8     |     8      |    1    |   24.06ms    |   23.90ms    |
| GPT-3(345M)   |    128   |    8     |    16      |    1    |   29.32ms    |   27.95ms    |
| GPT-3(6.7B)   |    128   |    8     |     1      |    1    |   84.93ms    |   63.96ms    |
| GPT-3(6.7B)   |    128   |    8     |     2      |    1    |   91.93ms    |   67.25ms    |
| GPT-3(6.7B)   |    128   |    8     |     4      |    1    |   105.50ms   |   78.98ms    |
| GPT-3(6.7B)   |    128   |    8     |     8      |    1    |   138.56ms   |   99.54ms    |
| GPT-3(6.7B)   |    128   |    8     |    16      |    1    |   204.33ms   |   140.97ms   |
| GPT-3(175B)   |    128   |    8     |     1      |    8    |   327.26ms   |   230.11ms   |
| GPT-3(175B)   |    128   |    8     |     2      |    8    |   358.61ms   |   244.23ms   |
| GPT-3(175B)   |    128   |    8     |     4      |    8    |   428.93ms   |   278.63ms   |
| GPT-3(175B)   |    128   |    8     |     8      |    8    |   554.28ms   |   344.00ms   |
| GPT-3(175B)   |    128   |    8     |    16      |    8    |   785.92ms   |   475.19ms   |

以上性能数据基于PaddlePaddle[每日版本](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-develop) ，依赖CUDA 11.6测试环境。
