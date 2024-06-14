# 无监督语义匹配模型 [DiffCSE](https://arxiv.org/pdf/2204.10298.pdf)

借鉴 [DiffCSE](https://arxiv.org/pdf/2204.10298.pdf) 的思路，实现了 DiffCSE 模型。相比于 SimCSE 模型，DiffCSE模型会更关注语句之间的差异性，具有精确的向量表示能力。DiffCSE 模型同样适合缺乏监督数据，但是又有大量无监督数据的匹配和检索场景。

## 快速开始
### 代码结构说明

以下是本项目主要代码结构及说明：

```
DiffCSE/
├── model.py # DiffCSE 模型组网代码
├── custom_ernie.py # 为适配 DiffCSE 模型，对ERNIE模型进行了部分修改
├── data.py # 无监督语义匹配训练数据、测试数据的读取逻辑
├── run_diffcse.py # 模型训练、评估、预测的主脚本
├── utils.py # 包括一些常用的工具式函数
├── run_train.sh # 模型训练的脚本
├── run_eval.sh # 模型评估的脚本
└── run_infer.sh # 模型预测的脚本
```

### 模型训练
默认使用无监督模式进行训练 DiffCSE，模型训练数据的数据样例如下所示，每行表示一条训练样本：
```shell
全年地方财政总收入3686.81亿元，比上年增长12.3%。
“我对案情并不十分清楚，所以没办法提出批评，建议，只能希望通过质询，要求检察院对此做出说明。”他说。
据调查结果显示：2015年微商行业总体市场规模达到1819.5亿元，预计2016年将达到3607.3亿元，增长率为98.3%。
前往冈仁波齐需要办理目的地包含日喀则和阿里地区的边防证，外转沿途有一些补给点，可购买到干粮和饮料。
```

可以运行如下命令，开始模型训练并且进行模型测试。

```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_train"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_diffcse.py \
    --mode "train" \
    --encoder_name "rocketqa-zh-dureader-query-encoder" \
    --generator_name "ernie-3.0-base-zh" \
    --discriminator_name "ernie-3.0-base-zh" \
    --max_seq_length "128" \
    --output_emb_size "32" \
    --train_set_file "your train_set path" \
    --eval_set_file "your dev_set path" \
    --save_dir "./checkpoints" \
    --log_dir ${log_dir} \
    --save_steps "50000" \
    --eval_steps "1000" \
    --epochs "3" \
    --batch_size "32" \
    --mlm_probability "0.15" \
    --lambda_weight "0.15" \
    --learning_rate "3e-5" \
    --weight_decay "0.01" \
    --warmup_proportion "0.01" \
    --seed "0" \
    --device "gpu"
```

可支持配置的参数：
* `mode`：可选，用于指明本次运行是模型训练、模型评估还是模型预测，仅支持[train, eval, infer]三种模式；默认为 infer。
* `encoder_name`：可选，DiffCSE模型中用于向量抽取的模型名称；默认为 ernie-3.0-base-zh。
* `generator_name`: 可选，DiffCSE模型中生成器的模型名称；默认为 ernie-3.0-base-zh。
* `discriminator_name`: 可选，DiffCSE模型中判别器的模型名称；默认为 rocketqa-zh-dureader-query-encoder。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `output_emb_size`：可选，向量抽取模型输出向量的维度；默认为32。
* `train_set_file`：可选，用于指定训练集的路径。
* `eval_set_file`：可选，用于指定验证集的路径。
* `save_dir`：可选，保存训练模型的目录；
* `log_dir`：可选，训练训练过程中日志的输出目录；
* `save_steps`：可选，用于指定模型训练过程中每隔多少 step 保存一次模型。
* `eval_steps`：可选，用于指定模型训练过程中每隔多少 step，使用验证集评估一次模型。
* `epochs`: 模型训练轮次，默认为3。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `mlm_probability`：可选，利用生成器预测时，控制单词掩码的比例，默认为0.15。
* `lambda_weight`：可选，控制RTD任务loss的占比，默认为0.15。
* `learning_rate`：可选，Fine-tune 的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.01。
* `warmup_proportion`：可选，学习率 warmup 策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到 learning_rate, 而后再缓慢衰减，默认为0.01。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选 cpu 或 gpu。如使用 gpu 训练则参数 gpus 指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。
如：
```text
checkpoints/
├── best
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
└── ...
```

### 模型评估
在模型评估时，需要使用带有标签的数据，以下展示了几条模型评估数据样例，每行表示一条训练样本，每行共计包含3列，分别是query1， query2， label：
```shell
右键单击此电脑选择属性，如下图所示   右键单击此电脑选择属性，如下图所示   5
好医生解密||是什么，让美洲大蠊能美容还能救命    解密美洲大蠊巨大药用价值        1
蒜香蜜汁烤鸡翅的做法    外香里嫩一口爆汁蒜蓉蜜汁烤鸡翅的做法    3
项目计划书 篇2  简易项目计划书（参考模板）      2
夏天幼儿园如何正确使用空调？    老师们该如何正确使用空调，让孩子少生病呢？      3
```


可以运行如下命令，进行模型评估。

```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_eval"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_diffcse.py \
    --mode "eval" \
    --encoder_name "rocketqa-zh-dureader-query-encoder" \
    --max_seq_length "128" \
    --output_emb_size "32" \
    --eval_set_file "your dev_set path" \
    --ckpt_dir "./checkpoints/best" \
    --batch_size "32" \
    --seed "0" \
    --device "gpu"
```
可支持配置的参数：
* `ckpt_dir`: 用于指定进行模型评估的checkpoint路径。

其他参数解释同上。

### 基于动态图模型预测
在模型预测时，需要给定待预测的两条文本，以下展示了几条模型预测的数据样例，每行表示一条训练样本，每行共计包含2列，分别是query1， query2：
```shell
韩国现代摩比斯2015招聘  韩国现代摩比斯2015校园招聘信息
《DNF》封号减刑方法 被封一年怎么办?     DNF封号减刑方法 封号一年怎么减刑
原神手鞠游戏三个刷新位置一览    手鞠游戏三个刷新位置一览
```

可以运行如下命令，进行模型预测：
```shell
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

log_dir="log_infer"
python -u -m paddle.distributed.launch --gpus ${gpu_ids} --log_dir ${log_dir} \
    run_diffcse.py \
    --mode "infer" \
    --encoder_name "rocketqa-zh-dureader-query-encoder" \
    --max_seq_length "128" \
    --output_emb_size "32" \
    --infer_set_file "your test_set path \
    --ckpt_dir "./checkpoints/best" \
    --save_infer_path "./infer_result.txt" \
    --batch_size "32" \
    --seed "0" \
    --device "gpu"
```

可支持配置的参数：
* `infer_set_file`: 可选，用于指定测试集的路径。
* `save_infer_path`: 可选，用于保存模型预测结果的文件路径。

其他参数解释同上。 待模型预测结束后，会将结果保存至save_infer_path参数指定的文件中。


## Reference
[1] Chuang Y S ,  Dangovski R ,  Luo H , et al. DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings[J]. arXiv e-prints, 2022. https://arxiv.org/pdf/2204.10298.pdf.
