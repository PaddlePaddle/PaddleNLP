# RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering

## 模型介绍

RocketQA是百度开源的一个检索式问答模型，可以用于智能问答，信息检索等领域，RocketQA提出了跨批次负采样和去噪的强负采样和数据增强的方法，并且开放了中文领域的预训练模型，在学术界和工业落地上都有着不错的表现。本项目是 RocketQA 的 PaddlePaddle 实现。

RocketQA在Dureader Retrieval的实验结果如下：

| Model |  MRR@10 | recall@1 | recall@50 |
| --- | --- | --- | --- |
| dual-encoder (retrieval) | 56.17 | 45 | 91.55|
| cross-encoder (re-ranking) | 65.62 | 55.50 | 91.75|

### 代码导航

以下是本项目目录结构及说明：

```
├── cross_encoder
│   ├── evaluate_cross.sh # Cross Encoder评估bash文件
│   ├── cross_model.py # 模型文件
│   ├── predict_ce.sh  # 预测 bash文件
│   ├── predict.py # 预测脚本
│   ├── train_ce.py # 训练脚本
│   ├── train_ce.sh # 训练的bash文件
│   └── data.py # 数据处理函数
├── dual_encoder
│   ├── create_index.py # 构建索引脚本
│   ├── data.py # 数据处理函数
│   ├── index_search.py  # 索引检索脚本
│   ├── inference_de.py # 预测脚本
│   ├── evaluate_de.sh # 评估bash文件
│   ├── extract_embeddings.sh # 向量抽取，索引构建和检索的bash文件
│   ├── merge.py # 结果融合脚本
│   ├── dual_model.py # 模型文件
│   ├── train_de.py # 训练文件
│   └── train_de.sh # 训练的bash脚本
├── metric
│   ├── utils.py  # dual encoder和cross encoder的结果转换成json文件的脚本
│   └── evaluation.py # 评估脚本
└── README.md
```

# 开始运行

## 环境要求

安装环境依赖paddlepaddle和paddlenlp

+ paddlenlp                          2.4.0
+ paddlepaddle-gpu                   2.3.2
+ faiss-cpu                      1.7.2

## 数据准备

### 基线数据

本项目的数据通过运行下面的命令下载：

```
wget -nv https://dataset-bj.cdn.bcebos.com/qianyan/dureader-retrieval-baseline-dataset.tar.gz
tar -zxvf dureader-retrieval-baseline-dataset.tar.gz
rm dureader-retrieval-baseline-dataset.tar.gz
```
下载数据集并解压以后，数据集就会存放在`dureader-retrieval-baseline-dataset/`目录下，解压后的目录结构如下：

```
dureader-retrieval-baseline-dataset/
├── auxiliary
│   ├── dev.retrieval.top50.res.id_map.tsv #  从dual encoder检索的top50的文章，用于评估cross encoder
│   └── dev.retrieval.top50.res.tsv # 文章的id和query的id的映射
├── dev
│   ├── dev.json # 评估样本
│   ├── dev.q.format # 评估集合的query文本
│   └── q2qid.dev.json # # 评估集文本和id的映射
├── License.docx
├── passage-collection # 篇章数据，包含800万条数据，分成了4个部分，便于数据并行
│   ├── part-00
│   ├── part-01
│   ├── part-02
│   ├── part-03
│   └── passage2id.map.json # 篇章的id映射文件
├── readme.md # 数据集的README文档
└── train
    ├── cross.train.demo.tsv # 训练 cross_encoder的demo数据集
    ├── cross.train.tsv # 训练 cross_encoder的全量数据集
    ├── dual.train.demo.tsv # 训练 dual_encoder的demo数据集
    └── dual.train.tsv # 训练 dual_encoder的全量数据集
```

`dual.train.tsv`和`dual.train.demo.tsv`的数据格式为：`query null para_text_pos null para_text_neg null`，用`\t`来分开，`null`表示不合法的列，
`para_text_pos` 表示的是正样本， `para_text_neg` 表示的是负样本。

数据示例为：

```
微信分享链接打开app        iOS里,把一个页面链接分享给微信好友(会话),好友在微信里打开这个链接,也就是打开了一个网页,点击网页里的某个地方后(比如网页中“打开xx应用程序”的按钮),代码里怎么设置可以跳回到第三方app?知乎的ios客户端就有这种功能,在微信里分享链接后,点开链接,再点网页中的某处,就可以打开知乎客户端显示全部微信中不能用自定义url的方式,微信提供了打开第三方应用的接口:launch3rdApp谢。一般用自带浏览器可以调用起app没问题。微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的        随便找个能够分享外链的网站，不要网盘什么的，最好找免费的主机或者免费空间，能拿到绝对路径就ok，出来像这样：http://23,44,23,21/upload/my.apk；http://www.liantu.com/举个例子，在联图网上直接复制你的URL，点击生成就ok了，这样用户一扫就能自动打开apk下载链接下载安装了 # 使用应用商店，私有协议 很多app都支持私有协议，私有协议以【appName】:// 开头。安卓应用商店也有私有协议，一般是market://，把你的应用上传到商店，然后由商店提供二维码，或者拿到私有协议的这个URL，参考上面的2步骤生成二维码都可以。 望采纳。    0
微信分享链接打开app        iOS里,把一个页面链接分享给微信好友(会话),好友在微信里打开这个链接,也就是打开了一个网页,点击网页里的某个地方后(比如网页中“打开xx应用程序”的按钮),代码里怎么设置可以跳回到第三方app?知乎的ios客户端就有这种功能,在微信里分享链接后,点开链接,再点网页中的某处,就可以打开知乎客户端显示全部微信中不能用自定义url的方式,微信提供了打开第三方应用的接口:launch3rdApp谢。一般用自带浏览器可以调用起app没问题。微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的        大小:107 MB版本:2.9.0.108 官方版环境:WinXP, Win7, WinAll通过这几个简单的操作步骤就可以解决在微信中出现的“请在微信客户端打开链接”的提示了,完成了设置之后重新单击链接并选择浏览器之后就可以顺利的打开了。若是你遇到了这个故障还没有解决,不妨试一试喔!希望这个教程能够帮助到大家。    0
微信分享链接打开app        iOS里,把一个页面链接分享给微信好友(会话),好友在微信里打开这个链接,也就是打开了一个网页,点击网页里的某个地方后(比如网页中“打开xx应用程序”的按钮),代码里怎么设置可以跳回到第三方app?知乎的ios客户端就有这种功能,在微信里分享链接后,点开链接,再点网页中的某处,就可以打开知乎客户端显示全部微信中不能用自定义url的方式,微信提供了打开第三方应用的接口:launch3rdApp谢。一般用自带浏览器可以调用起app没问题。微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的        首先我们需要先上百度，搜索找到带有微信多开APP的网站。打开safari浏览器，输入网址，然后点击下载。（最好使用wifi下载，APP会消耗大概100Mb流量）下载安装好了以后，我们还需要对软件进行开发者认证。此方式适用与ios10-11-12等系统，其他的方式也类是。在手机上找到【设置】-进去后找到【通用】-然后在找到【描述文件与设备管理】在【描述文件与设备管理】里面找到对应的开发者证书，点击证书-选择【信任】然后就可以返回到桌面。点击APP就可以正常打开。输入验证数字，登录帐号。就可以正常使用。注意：此方法仅仅适用于ios10以上的系统，太老的机型跟系统都不支持。    0
```

`cross.train.tsv`和 `cross.train.demo.tsv`的数据格式为：`query null para_text label` ，用`\t`来分开，`null`表示不合法的列，数据示例为：

```
微信分享链接打开app        iOS里,把一个页面链接分享给微信好友(会话),好友在微信里打开这个链接,也就是打开了一个网页,点击网页里的某个地方后(比如网页中“打开xx应用程序”的按钮),代码里怎么设置可以跳回到第三方app?知乎的ios客户端就有这种功能,在微信里分享链接后,点开链接,再点网页中的某处,就可以打开知乎客户端显示全部微信中不能用自定义url的方式,微信提供了打开第三方应用的接口:launch3rdApp谢。一般用自带浏览器可以调用起app没问题。微信里面能调出app的,是和腾讯有合作的应用,其他会被过滤掉。有一个公司的产品,叫 魔窗,免费可以接入的    1
微信分享链接打开app        百度经验:jingyan.baidu.com主要思路就是用一个可以在电脑上面打开手机软件的模拟器,来打开微信。经验内容仅供参考,如果您需解决具体问题(尤其法律、医学等领域),建议您详细咨询相关领域专业人士。个性签名:分享经验,帮助更多人。    0
微信分享链接打开app        在里面上传GIF动图或普通图片,生成短链接,在朋友圈评论时,粘贴链接即可。更多实用小程序,参考这篇文章:    0
```
对于 passage-collection/里面的数据part-0x，`null null passage_text null`，用`\t`来分开，`null`表示不合法的列，数据示例为：

```
-               现实总是残酷的戳心,富在深山有远亲,穷在闹市无人问,因为人情薄如纸,没人在乎你苦不苦,累不累,他们只在乎你飞得高不高,活得炫不炫;你的伤,你的痛,你的泪,别人看见只会笑话你的落魄,你的倾诉只会让别人嘲笑你的懦弱。真正的强者,不言悲伤,不言痛苦,不言苦累,从不高估自己和任何人的关系,更不高估自己在别人心中的位置,不会轻易把自己艰辛的心路历程展示给别人看,那些关注你的人,等着看你笑话的人很多,真正希望你过得好的人,真心没几个。当你落魄一次,就知道人情冷暖,世态炎凉,平日里往来密切,称兄道弟的人,都会绕道而行,避着你,躲着你,心里小瞧着你。        0
-               1、按下Win+R键打开运行,输入regedit,回车打开注册表编辑器; 2、依次展开到路径HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Run, 在右侧新建一个名为systray.exe的字符串值; 3、双击打开该键值,将其数据数值修改为C:\Windows\System32\systray.exe; 4、重启电脑即可解决问题。    0
```

对于 `passage2id.map.json`，为json数据，即  `passage_line_index` 和 `passage_id`的映射。

```
{
    "0": "67236474b99b2215c296a6942ad6e04c",
    "1": "49e1730c70e3cc22771d75be410cbd18",
    "2": "8cd8cf7ebe40f0ab812e85bfd5c4c6c1",
    "3": "28eff9f9985e27adf416887dd92e616b",
```
对于 `dev.json`,数据为json格式，示例数据如下：

```
{"question_id": "edb58f525bd14724d6f490722fa8a657", "question": "国家法定节假日共多少天", "answer_paragraphs": [{"paragraph_id": "2c4fe63d3378ac39907b6b2648eb40c5", "paragraph_text": "一年国家法定节假日为11天。根据公布的国家法定节假日调整方案，调整的主要内容包括：元旦放假1天不变；春节放假3天，放假时间为农历正月初一、初二、初三；“五一”国际劳动节1天不变；“十一”国庆节放假3天；清明节、端午节、中秋节增设为国家法定节假日，各放假1天(农历节日如遇闰月，以第一个月为休假日)。3、允许周末上移下错，与法定节假日形成连休。"},...
```
对于 `dev.q.format`, 格式为`query_text null null null`，用`\t`来分开，`null`表示不合法的列，示例数据如下：

```
国家法定节假日共多少天  -       -       0
如何查看好友申请        -       -       0
哪个网站有湖南卫视直播  -       -       0
功和功率的区别  -       -       0
徐州旅行社哪家好        -       -       0
为什么要平衡摩擦力      -       -       0
```

对于 `q2qid.dev.json`，格式为JSON格式，示例数据如下：

```
{
    "国家法定节假日共多少天": "edb58f525bd14724d6f490722fa8a657",
    "如何查看好友申请": "a451acd1e9836b04b16664e9f0c290e5",
    "哪个网站有湖南卫视直播": "48a8338aefaff17573048a088a08de70",
    "功和功率的区别": "7871706c5cb1a1d6912ff8222434ccbd",
    "徐州旅行社哪家好": "2b22b972c4e30776b9160fc83ee05367",
    ....
```
对于 `dev.retrieval.top50.res.tsv`，格式为`query_text null passage_text null`，用`\t`来分开，`null`表示不合法的列，示例数据如下表示：
```
国家法定节假日共多少天          请问国家规定每年法定节假日是几天 国家法定节假日2015年国家法定节假日2015年国家法定假日,用人单位安排加班的,须在正常支付员工工资的基础上,按不低于员工本人日或小时工资的300%另行支付加班工资。也就是说,10月1日、2日、3日三天,加班费按3倍标准执行,同时,由于3日出现法定...国家法定节假日天数全年有多少天国家法定节假日天数全年有多少天目前国家法定节假日天数全年一共有11天,国家法定节假日天数曾作出调整,国家法定节假日天数在总天数里增加了1天,由10天变成现在的11天,这一调整对旅游和传统文化都产生了一定的影响。国家法定...关于节假日的最新规定(最新)        0
国家法定节假日共多少天          现在法定假日是元旦1天，春节3天，清明节1天，五一劳动节1天，端午节1天，国庆节3天，中秋节1天，共计11天。法定休息日每年52个周末总共104天。合到一起总计115天。这些日历上面都有吧，自己看呀！人生日历上面的假期很全很准，可以试一下。        0
```

对于 `dev.retrieval.top50.res.id_map.tsv`，格式为`query_id passage_id`，用`\t`来分开，示例数据如下：

```
edb58f525bd14724d6f490722fa8a657        a0ac26837aef3497f4fe08b6243b8eed
edb58f525bd14724d6f490722fa8a657        6e67389d07da8ce02ed97167d23baf9d
edb58f525bd14724d6f490722fa8a657        06031941b9613d2fde5cb309bbefaf88
edb58f525bd14724d6f490722fa8a657        f794a6dc6c26ba7a64d14ddfea5954ff
......
```


## 运行RocketQA

RocketQA的检索分成两步：

+ 第一步是使用dual-encoder做段落检索
+ 第二步是使用cross-encoder来做段落排序

由于训练和预测过程比较复杂，这里推荐使用脚本的方式运行。

## 第一步 Dual-encoder

### 训练

训练的命令如下，需要开启cross_batch策略和recompute策略：

```
TRAIN_SET="../dureader-retrieval-baseline-dataset/train/dual.train.tsv"
python -u -m paddle.distributed.launch --gpus "0,1,2,3" \
                    train_de.py \
                   --train_set_file ${TRAIN_SET} \
                   --save_dir ./checkpoint \
                   --batch_size 128 \
                   --save_steps 8685 \
                   --query_max_seq_length 32 \
                   --title_max_seq_length 384 \
                   --learning_rate 3e-5 \
                   --epochs 10 \
                   --weight_decay 0.0 \
                   --warmup_proportion 0.1 \
                   --use_cross_batch \
                   --seed 1 \
                   --use_amp \
                   --use_recompute
```

其中参数释义如下：
- `train_set_file` 训练数据的路径。
- `device` 使用 cpu/gpu 进行训练
- `save_dir` 模型保存的路径。
- `model_name_or_path` 预训练语言模型名，用于模型的初始化，默认ernie-1.0。
- `batch_size` 批次的大小。
- `save_steps` 保存模型的step间隔。
- `output_emb_size` Transformer 顶层输出的文本向量维度
- `query_max_seq_length` 输入的query文本的最大长度。
- `title_max_seq_length` 输入的paragraph的最大长度。
- `learning_rate` 模型的学习率
- `epochs` 模型训练的回合数。
- `weight_decay` 优化器的权重衰减系数。
- `warmup_proportion` 学习率warmup参数。
- `use_cross_batch` 是否开启 cross batch的训练。
- `log_steps` 日志保存的step数。
- `seed` 随机种子。
- `init_from_ckpt` 加载模型继续进行训练。
- `use_amp` 开启混合精度策略。
- `use_recompute` 开启recompute的策略。
- `scale_loss` 表示自动混合精度训练的参数。

推荐使用 bash 脚本的方式运行：
```
cd dual_encoder
bash train_de.sh
```

### 向量抽取

向量抽取过程比较复杂，需要分别抽取query和title的向量，并建立索引，然后进行检索，推荐使用脚本的方式：

```
bash extract_embeddings.sh
```

### 评估预测

```
DATA_PATH="../dureader-retrieval-baseline-dataset/passage-collection"
TOP_K=50
para_part_cnt=`cat $DATA_PATH/part-00 | wc -l`
python merge.py --para_part_cnt ${para_part_cnt} \
                --top_k ${TOP_K} \
                --total_part 4 \
                --inputfiles output/res.top50-part0 output/res.top50-part1 output/res.top50-part2 output/res.top50-part3 \
                --outfile output/dev.res.top${TOP_K}

QUERY2ID="../dureader-retrieval-baseline-dataset/dev/q2qid.dev.json"
PARA2ID="../dureader-retrieval-baseline-dataset/passage-collection/passage2id.map.json"
MODEL_OUTPUT="output/dev.res.top50"
# python metric/convert_recall_res_to_json.py $QUERY2ID $PARA2ID $MODEL_OUTPUT
python ../metric/utils.py --q2id_map $QUERY2ID \
                       --p2id_map $PARA2ID \
                       --recall_result $MODEL_OUTPUT \
                       --outputf output/dual_res.json

REFERENCE_FIEL="../dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/dual_res.json"
python ../metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
```
推荐使用 bash脚本的方式：

```
bash evaluate_de.sh
```

## 第二步 Cross-encoder

### 训练

训练的命令如下：
```
TRAIN_SET="../dureader-retrieval-baseline-dataset/train/cross.train.tsv"
node=4
epoch=3
lr=1e-5
batch_size=32
train_exampls=`cat $TRAIN_SET | wc -l`
save_steps=$[$train_exampls/$batch_size/$node]
new_save_steps=$[$save_steps*$epoch/2]

python -u -m paddle.distributed.launch --gpus "0,1,2,3" train_ce.py \
        --device gpu \
        --train_set ${TRAIN_SET} \
        --save_dir ./checkpoints \
        --batch_size ${batch_size} \
        --save_steps ${new_save_steps} \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs ${epoch}
```

其中参数释义如下：
- `train_set` 训练数据的路径。
- `device` 使用 cpu/gpu 进行训练
- `model_name_or_path` 预训练语言模型名，用于模型的初始化，默认ernie-1.0。
- `save_dir` 模型保存的路径。
- `batch_size` 批次的大小。
- `save_steps` 保存模型的step间隔。
- `max_seq_len` 文本序列的最大长度。
- `learning_rate` 模型的学习率。
- `weight_decay` 优化器的权重衰减系数。
- `epochs` 模型训练的回合数。
- `warmup_proportion` 学习率warmup参数。
- `logging_steps` 日志保存的step数。
- `use_amp` 开启混合精度策略。
- `scale_loss` 表示自动混合精度训练的参数。
- `seed` 随机种子。
- `init_from_ckpt` 加载模型继续进行训练。

推荐使用 bash脚本运行：

```
cd cross_encoder
bash train_ce.sh
```

### 预测

模型的预测命令如下：

```
TEST_SET="../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.tsv"
MODEL_PATH="./checkpoints/model_26040/model_state.pdparams"

python predict.py \
                --device 'gpu' \
                --params_path ${MODEL_PATH} \
                --test_set ${TEST_SET} \
                --batch_size 128 \
                --max_seq_length 384
```
其中参数释义如下：
- `test_set` 预测数据的路径。
- `device` 使用 cpu/gpu 进行训练
- `params_path` 模型的加载路径
- `batch_size` 批次的大小。
- `save_dir` 预测结果的保存路径。
- `max_seq_len` 文本序列的最大长度。

推荐使用 bash脚本运行：
```
bash predict_ce.sh
```

### 评估

评估的命令如下：

```
MODEL_OUTPUT="output/result.txt"
ID_MAP="../dureader-retrieval-baseline-dataset/auxiliary/dev.retrieval.top50.res.id_map.tsv"
python ../metric/utils.py --score_f $MODEL_OUTPUT \
                       --id_f $ID_MAP \
                       --mode rank \
                       --outputf output/cross_res.json
REFERENCE_FIEL="../dureader-retrieval-baseline-dataset/dev/dev.json"
PREDICTION_FILE="output/cross_res.json"
python ../metric/evaluation.py $REFERENCE_FIEL $PREDICTION_FILE
```
推荐使用 bash脚本运行：
```
bash evaluate_cross.sh
```

## 参考文献

- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

- [DuReader_retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine](https://arxiv.org/abs/2203.10232)
