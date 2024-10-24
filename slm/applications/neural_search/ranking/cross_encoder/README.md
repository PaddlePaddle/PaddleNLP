
 **目录**

* [背景介绍](#背景介绍)
* [CrossEncoder](#CrossEncoder)
    * [1. 技术方案和评估指标](#技术方案)
    * [2. 环境依赖](#环境依赖)
    * [3. 代码结构](#代码结构)
    * [4. 数据准备](#数据准备)
    * [5. 模型训练](#模型训练)
    * [6. 评估](#开始评估)
    * [7. 预测](#预测)
    * [8. 部署](#部署)

<a name="背景介绍"></a>

# 背景介绍

基于 RocketQA 的 CrossEncoder 训练的单塔模型，该模型用于搜索的排序阶段，对召回的结果进行重新排序的作用。


<a name="CrossEncoder"></a>

# CrossEncoder

<a name="技术方案"></a>

## 1. 技术方案和评估指标

### 技术方案

加载基于 ERNIE 3.0训练过的 RocketQA 单塔 CrossEncoder 模型。


### 评估指标

（1）采用 AUC 指标来评估排序模型的排序效果。

**效果评估**

|  训练方式 |  模型  | AUC |
| ------------ | ------------ |------------ |
| pairwise|  ERNIE-Gram  |0.801 |
|  CrossEncoder |  rocketqa-base-cross-encoder  |**0.835** |

<a name="环境依赖"></a>

## 2. 环境依赖和安装说明

**环境依赖**

* python >= 3.7
* paddlepaddle >= 2.3.7
* paddlenlp >= 2.3
* pandas >= 0.25.1
* scipy >= 1.3.1

<a name="代码结构"></a>

## 3. 代码结构

以下是本项目主要代码结构及说明：

```
ernie_matching/
├── deply # 部署
    ├── cpp
        ├── rpc_client.py # RPC 客户端的bash脚本
        ├── http_client.py # http 客户端的bash文件
        └── start_server.sh # 启动C++服务的脚本
    └── python
        ├── deploy.sh # 预测部署bash脚本
        ├── config_nlp.yml # Pipeline 的配置文件
        ├── web_service.py # Pipeline 服务端的脚本
        ├── rpc_client.py # Pipeline RPC客户端的脚本
        └── predict.py # python 预测部署示例
|—— scripts
    ├── export_model.sh # 动态图参数导出静态图参数的bash文件
    ├── export_to_serving.sh # 导出 Paddle Serving 模型格式的bash文件
    ├── train_ce.sh # 匹配模型训练的bash文件
    ├── evaluate_ce.sh # 评估验证文件bash脚本
    ├── predict_ce.sh # 匹配模型预测脚本的bash文件
├── export_model.py # 动态图参数导出静态图参数脚本
├── export_to_serving.py # 导出 Paddle Serving 模型格式的脚本
├── data.py #  训练样本的转换逻辑
├── train_ce.py # 模型训练脚本
├── evaluate.py # 评估验证文件
├── predict.py # Pair-wise 模型预测脚本，输出文本对是相似度

```

<a name="数据准备"></a>

## 4. 数据准备

### 数据集说明

样例数据如下:
```
(小学数学教材比较) 关键词:新加坡        新加坡与中国数学教材的特色比较数学教材,教材比较,问题解决        0
徐慧新疆肿瘤医院        头颈部非霍奇金淋巴瘤扩散加权成像ADC值与Ki-67表达相关性分析淋巴瘤,非霍奇金,头颈部肿瘤,磁共振成像 1
抗生素关性腹泻  鼠李糖乳杆菌GG防治消化系统疾病的研究进展鼠李糖乳杆菌,腹泻,功能性胃肠病,肝脏疾病,幽门螺杆菌      0
德州市图书馆    图书馆智慧化建设与融合创新服务研究图书馆;智慧化;阅读服务;融合创新       1
维生素c 综述    维生素C防治2型糖尿病研究进展维生素C;2型糖尿病;氧化应激;自由基;抗氧化剂  0
(白藜芦醇) 关键词:2型糖尿病     2型糖尿病大鼠心肌缺血再灌注损伤转录因子E2相关因子2/血红素氧合酶1信号通路的表达及白藜芦醇的干预研究糖尿病,2型,心肌缺血,再灌注损伤,白藜芦醇       1
融资偏好        创新型企业产业风险、融资偏好与融资选择融资偏好;产业风险;融资选择        1
星载激光雷达    星载激光雷达望远镜主镜超轻量化结构设计超轻量化;拓扑优化;集成优化;RMS;有限元仿真 1
```


### 数据集下载


- [literature_search_rank](https://paddlenlp.bj.bcebos.com/applications/literature_search_rank.zip)

```
├── data # 排序数据集
    ├── test.csv   # 测试集
    ├── dev_pairwise.csv    # 验证集
    └── train.csv  # 训练集
```

<a name="模型训练"></a>

## 5. 模型训练

**排序模型下载链接：**


|Model|训练参数配置|硬件|MD5|
| ------------ | ------------ | ------------ |-----------|
|[ERNIE-Gram-Sort](https://bj.bcebos.com/v1/paddlenlp/models/ernie_gram_sort.zip)|<div style="width: 150pt">epoch:3 lr:5E-5 bs:64 max_len:64 </div>|<div style="width: 100pt">4卡 v100-16g</div>|d24ece68b7c3626ce6a24baa58dd297d|


### 训练环境说明


- NVIDIA Driver Version: 440.64.00
- Ubuntu 16.04.6 LTS (Docker)
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz


### 单机单卡训练/单机多卡训练

这里采用单机多卡方式进行训练，通过如下命令，指定 GPU 0,1,2,3 卡。如果采用单机单卡训练，只需要把`--gpu`参数设置成单卡的卡号即可

训练的命令如下：

```
unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir="logs" train_ce.py \
        --device gpu \
        --train_set data/train.csv \
        --test_file data/dev_pairwise.csv \
        --save_dir ./checkpoints \
        --model_name_or_path rocketqa-base-cross-encoder \
        --batch_size 32 \
        --save_steps 10000 \
        --max_seq_len 384 \
        --learning_rate 1E-5 \
        --weight_decay  0.01 \
        --warmup_proportion 0.0 \
        --logging_steps 10 \
        --seed 1 \
        --epochs 3 \
        --eval_step 1000
```
也可以运行 bash 脚本：

```
sh scripts/train_ce.sh
```

<a name="评估"></a>

## 6. 评估


```
python evaluate.py --model_name_or_path rocketqa-base-cross-encoder \
                   --init_from_ckpt checkpoints/model_80000/model_state.pdparams \
                   --test_file data/dev_pairwise.csv
```
也可以运行 bash 脚本：

```
sh scripts/evaluate_ce.sh
```


成功运行后会输出下面的指标：

```
eval_dev auc:0.829
```

<a name="预测"></a>

## 7. 预测

### 准备预测数据

待预测数据为 tab 分隔的 tsv 文件，每一行为 1 个文本 Pair，和文本 pair 的语义索引相似度，(该相似度由召回模型算出，仅供参考)，部分示例如下:

```
中西方语言与文化的差异  第二语言习得的一大障碍就是文化差异。    0.5160342454910278
中西方语言与文化的差异  跨文化视角下中国文化对外传播路径琐谈跨文化,中国文化,传播,翻译   0.5145505666732788
中西方语言与文化的差异  从中西方民族文化心理的差异看英汉翻译语言,文化,民族文化心理,思维方式,翻译        0.5141439437866211
中西方语言与文化的差异  中英文化差异对翻译的影响中英文化,差异,翻译的影响        0.5138794183731079
中西方语言与文化的差异  浅谈文化与语言习得文化,语言,文化与语言的关系,文化与语言习得意识,跨文化交际      0.5131710171699524
```



### 开始预测

以上述 demo 数据为例，运行如下命令基于我们开源的 rocketqa 模型开始计算文本 Pair 的语义相似度:

```shell
unset CUDA_VISIBLE_DEVICES
python predict.py \
                --device 'gpu' \
                --params_path checkpoints/model_80000/model_state.pdparams \
                --model_name_or_path rocketqa-base-cross-encoder \
                --test_set data/test.csv \
                --topk 10 \
                --batch_size 128 \
                --max_seq_length 384
```
也可以直接执行下面的命令：

```
sh scripts/predict_ce.sh
```
得到下面的输出，分别是 query，title 和对应的预测概率：

```
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '高校\\十四五\\规划中学科建设要处理好五对关系\\十四五\\规划,学科建设,科技创新,人才培养', 'pred_prob': 0.7076062}
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '校企科研合作项目管理模式创新校企科研合作项目,管理模式,问题,创新', 'pred_prob': 0.64633846}
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '科研项目管理策略科研项目,项目管理,实施,必要性,策略', 'pred_prob': 0.63166416}
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '高校科研项目经费管理流程优化研究——以z大学为例高校,科研项目经费\\全流程\\管理,流程优化', 'pred_prob': 0.60351866}
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '关于推进我院科研发展进程的相关问题研究医院科研,主体,环境,信息化', 'pred_prob': 0.5688347}
{'text_a': '加强科研项目管理有效促进医学科研工作', 'text_b': '医学临床科研选题原则和方法医学临床,科学研究,选题', 'pred_prob': 0.55190295}
```

<a name="部署"></a>

## 8. 部署

### 动转静导出

首先把动态图模型转换为静态图：

```
python export_model.py \
                       --params_path checkpoints/model_80000/model_state.pdparams \
                       --model_name_or_path rocketqa-base-cross-encoder \
                       --output_path=./output
```
也可以运行下面的 bash 脚本：

```
sh scripts/export_model.sh
```

### Paddle Inference

使用 PaddleInference

```
python deploy/python/predict.py --model_dir ./output \
                                --input_file data/test.csv \
                                --model_name_or_path rocketqa-base-cross-encoder
```
也可以运行下面的 bash 脚本：

```
sh deploy/python/deploy.sh
```
得到下面的输出，输出的是样本的 query，title 以及对应的概率：

```
Data: {'query': '加强科研项目管理有效促进医学科研工作', 'title': '科研项目管理策略科研项目,项目管理,实施,必要性,策略'}   prob: 0.5479063987731934
Data: {'query': '加强科研项目管理有效促进医学科研工作', 'title': '关于推进我院科研发展进程的相关问题研究医院科研,主体,环境,信息化'}      prob: 0.5151925086975098
Data: {'query': '加强科研项目管理有效促进医学科研工作', 'title': '深圳科技计划对高校科研项目资助现状分析与思考基础研究,高校,科技计划,科技创新'}          prob: 0.42983829975128174
Data: {'query': '加强科研项目管理有效促进医学科研工作', 'title': '普通高校科研管理模式的优化与创新普通高校,科研,科研管理'}       prob: 0.465454638004303
```

### Paddle Serving 部署

Paddle Serving 的详细文档请参考 [Pipeline_Design](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Python_Pipeline/Pipeline_Design_CN.md)和[Serving_Design](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Serving_Design_CN.md),首先把静态图模型转换成 Serving 的格式：

```
python export_to_serving.py \
    --dirname "output" \
    --model_filename "inference.pdmodel" \
    --params_filename "inference.pdiparams" \
    --server_path "serving_server" \
    --client_path "serving_client" \
    --fetch_alias_names "predict"

```

参数含义说明
* `dirname`: 需要转换的模型文件存储路径，Program 结构文件和参数文件均保存在此目录。
* `model_filename`： 存储需要转换的模型 Inference Program 结构的文件名称。如果设置为 None ，则使用 `__model__` 作为默认的文件名
* `params_filename`: 存储需要转换的模型所有参数的文件名称。当且仅当所有模型参数被保>存在一个单独的二进制文件中，它才需要被指定。如果模型参数是存储在各自分离的文件中，设置它的值为 None
* `server_path`: 转换后的模型文件和配置文件的存储路径。默认值为 serving_server
* `client_path`: 转换后的客户端配置文件存储路径。默认值为 serving_client
* `fetch_alias_names`: 模型输出的别名设置，比如输入的 input_ids 等，都可以指定成其他名字，默认不指定
* `feed_alias_names`: 模型输入的别名设置，比如输出 pooled_out 等，都可以重新指定成其他模型，默认不指定

也可以运行下面的 bash 脚本：
```
sh scripts/export_to_serving.sh
```
Paddle Serving 的部署有两种方式，第一种方式是 Pipeline 的方式，第二种是 C++的方式，下面分别介绍这两种方式的用法：

#### Pipeline 方式

修改对应预训练模型的`Tokenizer`：

```
self.tokenizer = AutoTokenizer.from_pretrained('rocketqa-base-cross-encoder')
```

启动 Pipeline Server:

```
python web_service.py
```

启动客户端调用 Server。

首先修改 rpc_client.py 中需要预测的样本：

```
list_data = [{"query":"加强科研项目管理有效促进医学科研工作","title":"科研项目管理策略科研项目,项目管理,实施,必要性,策略"}]`
```
然后运行：
```
python rpc_client.py
```
模型的输出为：

```
PipelineClient::predict pack_data time:1662354188.422532
PipelineClient::predict before time:1662354188.423034
time to cost :0.016808509826660156 seconds
(1,)
[0.5479064]
```
可以看到客户端发送了1条文本，这条文本的相似的概率值。

#### C++的方式

启动 C++的 Serving：

```
python -m paddle_serving_server.serve --model serving_server --port 8600 --gpu_id 0 --thread 5 --ir_optim True
```
也可以使用脚本：

```
sh deploy/cpp/start_server.sh
```
Client 可以使用 http 或者 rpc 两种方式，rpc 的方式为：

```
python deploy/cpp/rpc_client.py
```
运行的输出为：

```
I0905 05:38:28.876770 28507 general_model.cpp:490] [client]logid=0,client_cost=158.124ms,server_cost=156.385ms.
time to cost :0.15848731994628906 seconds
[0.54790646]
```
可以看到服务端返回了相似度结果

或者使用 http 的客户端访问模式：

```
python deploy/cpp/http_client.py
```
运行的输出为：
```
time to cost :0.13054680824279785 seconds
0.5479064707850817
```
可以看到服务端返回了相似度结果


## Reference

[1] Xiao, Dongling, Yu-Kun Li, Han Zhang, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. “ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding.” ArXiv:2010.12148 [Cs].

[2] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, Haifeng Wang:
RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. NAACL-HLT 2021: 5835-5847
