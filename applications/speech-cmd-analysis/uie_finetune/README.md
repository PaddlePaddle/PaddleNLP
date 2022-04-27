# UIE

[Universal Information Extraction（UIE）](https://universal-ie.github.io/)将各种类型的信息抽取任务统一转化为自然语言的形式，并进行多任务联合训练。该模型支持多种类型的开放抽取任务，包括但不限于命名实体、关系、事件论元、事件描述片段、评价、评价维度、观点词、情感倾向等。

## 快速开始

### 代码结构说明

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── metric.py         # 模型效果验证指标脚本
├── doccano.py        # 数据标注脚本
├── train.py          # 模型训练脚本
├── evaluate.py       # 模型评估脚本
├── run_train.sh      # 模型训练命令
├── run_evaluate.sh   # 模型评估命令
└── README.md
```

### 模型输入数据格式

prompt为`实体类别标签`:

```text
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "完美和喜悦在我心中", "start": 1, "end": 10}], "prompt": "作品名"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "海天出版社", "start": 17, "end": 22}], "prompt": "机构名"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "阿兰科恩", "start": 31, "end": 35}], "prompt": "人物名"}
```

prompt为`实体名称` + 的 + `关系类别标签`:

```text
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "海天出版社", "start": 17, "end": 22}], "prompt": "完美和喜悦在我心中的出版社名称"}
{"content": "《完美和喜悦在我心中》是2003年海天出版社出版的图书，作者是阿兰科恩", "result_list": [{"text": "阿兰科恩", "start": 31, "end": 35}], "prompt": "完美和喜悦在我心中的作者"}
```

### 数据标注

我们推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本案例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。为达到这个目的，您需要按以下标注规则在doccano平台上标注数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/164374314-9beea9ad-08ed-42bc-bbbc-9f68eb8a40ee.png />
    <p>图2 数据标注样例图<p/>
</div>

- 在doccano平台上，定义实体标签类别和关系标签类别，上例中需要定义的实体标签有`作品名`、`机构名`和`人物名`，关系标签有`出版社名称`和`作者`。
- 使用以上定义的标签开始标注数据，图2展示了一个标注样例。
- 当标注完成后，在 doccano 平台上导出 `jsonl` 形式的文件，并将其重命名为 `doccano.json` 后，放入 `./data` 目录下。
- 通过 [doccano.py](./doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。

```shell
python doccano.py \
    --doccano_file ./data/doccano.json \
    --save_dir ./data/ext_data \
    --negative_ratio 5
```

**备注：**
- 默认情况下 [doccano.py](./doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](./doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。

### 模型训练

下载训练好的[UIE模型](https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie/model_state.pdparams)并放入`./uie_model`中:

通过运行以下命令进行自定义UIE模型训练：

```shell
python train.py \
    --train_path ./data/ext_data/train.txt \
    --dev_path ./data/ext_data/dev.txt \
    --save_dir ./checkpoint \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 512 \
    --num_epochs 50 \
    --init_from_ckpt ./uie_model/model_state.pdparams \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu
```

### 模型评估

通过运行以下命令进行模型评估：

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best/model_state.pdparams \
    --test_path ./data/ext_data/test.txt \
    --batch_size 16 \
    --max_seq_len 512
```

### Taskflow一键预测

通过`schema`自定义抽取目标，`task_path`指定使用标注数据训练的UIE模型。

```python
from paddlenlp import Taskflow

schema = [{"作品名": ["作者", "出版社名称"]}]

# 为任务实例设定抽取目标和定制化模型权重路径
my_ie = Taskflow("information_extraction", schema=schema, task_path='./checkpoint/model_best')
```

## Reference

Unified Structure Generation for Universal Information Extraction. Yaojie Lu, Qing Liu, Dai Dai, Xinyan Xiao, Hongyu Lin, Xianpei Han, Le Sun and Hua Wu. ACL 2022. [[arxiv](https://arxiv.org/abs/2203.12277)]
