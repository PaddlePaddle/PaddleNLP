# WINDOWS 环境下搭建端到端 FAQ 智能问答系统
以下的流程都是使用的 Anaconda 的环境进行的搭建，Anaconda 安装好以后，进入 `Anaconda Powershell Prompt`（由于环境变量设置不兼容的原因，暂不支持使用`cmd`执行下面的命令），然后执行下面的流程。

## 1. 快速开始: 快速搭建 FAQ 智能问答系统

### 1.1 运行环境和安装说明

a. 依赖安装：
我们预置了基于[ 8000 多条保险行业问答数据](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/baoxianzhidao/intro.ipynb)搭建保险 FAQ 智能问答的代码示例，您可以通过如下命令快速体验智能问答的效果
```bash
git clone https://github.com/tvst/htbuilder.git
cd htbuilder/
python setup install
# pip 一键安装
pip install --upgrade paddle-pipelines -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或者源码进行安装最新版本
cd ${HOME}/PaddleNLP/pipelines/
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python setup.py install
```

```
# 下载pipelines源代码
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP/pipelines
```

【注意】以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录
### 1.2 数据说明
我们预置了基于[ 8000 多条保险行业问答数据](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/baoxianzhidao/intro.ipynb)搭建保险 FAQ 智能问答的代码示例，您可以通过如下命令快速体验智能问答的效果

### 1.3 一键体验 FAQ 智能问答系统

```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
python examples/FAQ/dense_faq_example.py --device gpu
# 如果只有 CPU 机器，安装CPU版本的Paddle后，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
python examples/FAQ/dense_faq_example.py --device cpu
```

### 1.4 构建 Web 可视化 FAQ 系统

整个 Web 可视化 FAQ 智能问答系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的 FAQ 智能问答系统。

#### 1.4.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动 ES 服务
把`xpack.security.enabled` 设置成 false，如下：
```
xpack.security.enabled: false
```

然后直接双击 bin 目录下的 elasticsearch.bat 即可启动。

3. elasticsearch 可视化工具 Kibana（可选）
为了更好的对数据进行管理，可以使用 Kibana 可视化工具进行管理和分析，下载链接为[Kibana](https://www.elastic.co/cn/downloads/kibana)，下载完后解压，直接双击运行 `bin\kibana.bat`即可。

#### 1.4.2 文档数据写入 ANN 索引库
```
# 以DuReader-Robust 数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name insurance --doc_dir data/insurance --split_answers --delete_index --query_embedding_model rocketqa-zh-nano-query-encoder --passage_embedding_model rocketqa-zh-nano-para-encoder --embedding_dim 312
```
参数含义说明
* `index_name`: 索引的名称
* `doc_dir`: txt 文本数据的路径
* `host`: Elasticsearch 的 IP 地址
* `port`: Elasticsearch 的端口号
* `delete_index`: 是否删除现有的索引和数据，用于清空 es 的数据，默认为 false


运行结束后，可使用 Kibana 查看数据

#### 1.4.3 启动 RestAPI 模型服务

**注意** dense_faq.yaml 里面的检索模型需要与前面使用 offline_ann.py 建库的时候使用的检索模型一致

```bash
# 指定FAQ智能问答系统的Yaml配置文件
$env:PIPELINE_YAML_PATH='rest_api/pipeline/dense_faq.yaml'
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

#### 1.4.4 启动 WebUI
```bash
# 配置模型服务地址
$env:API_ENDPOINT='http://127.0.0.1:8891'
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_faq.py --server.port 8502
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验 FAQ 智能问答系统服务了。

#### 1.4.5 数据更新

数据更新的方法有两种，第一种使用前面的 `utils/offline_ann.py`进行数据更新，另一种是使用前端界面的文件上传进行数据更新，支持 txt，pdf，image，word 的格式，以 txt 格式的文件为例，每段文本需要使用空行隔开，程序会根据空行进行分段建立索引，示例数据如下(demo.txt)：

```
兴证策略认为，最恐慌的时候已经过去，未来一个月市场迎来阶段性修复窗口。

从海外市场表现看，
对俄乌冲突的恐慌情绪已显著释放，
海外权益市场也从单边下跌转入双向波动。

长期，继续聚焦科技创新的五大方向。1)新能源(新能源汽车、光伏、风电、特高压等)，2)新一代信息通信技术(人工智能、大数据、云计算、5G等)，3)高端制造(智能数控机床、机器人、先进轨交装备等)，4)生物医药(创新药、CXO、医疗器械和诊断设备等)，5)军工(导弹设备、军工电子元器件、空间站、航天飞机等)。
```

如果安装遇见问题可以查看[FAQ 文档](../../FAQ.md)
