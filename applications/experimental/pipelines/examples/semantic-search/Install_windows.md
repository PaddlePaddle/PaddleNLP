# WINDOWS环境下搭建端到端语义检索系统
以下的流程都是使用的Anaconda的环境进行的搭建，Anaconda安装好以后，进入 `Anaconda Powershell Prompt`，然后执行下面的流程。

## 1. 快速开始: 快速搭建语义检索系统

### 1.1 运行环境和安装说明

a. 依赖安装：
```bash
pip install -r requirements.txt
# 1) 安装 pipelines package
cd ${HOME}/PaddleNLP/applications/experimental/pipelines/
python setup.py install
```
### 1.2 数据说明
语义检索数据库的数据来自于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)，共包含 46972 个段落文本。


### 1.3 一键体验语义检索系统
我们预置了基于[DuReader-Robust数据集](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)搭建语义检索系统的代码示例，您可以通过如下命令快速体验语义检索系统的效果
```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/semantic-search/semantic_search_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/semantic-search/semantic_search_example.py --device cpu
```

### 1.4 构建 Web 可视化语义检索系统

整个 Web 可视化语义检索系统主要包含 3 大组件: 1. 基于 ElasticSearch 的 ANN 服务 2. 基于 RestAPI 构建模型服务 3. 基于 Streamlit 构建 WebUI，接下来我们依次搭建这 3 个服务并最终形成可视化的语义检索系统。

#### 1.4.1 启动 ANN 服务
1. 参考官方文档下载安装 [elasticsearch-8.3.2](https://www.elastic.co/cn/downloads/elasticsearch) 并解压。
2. 启动 ES 服务
把`xpack.security.enabled` 设置成false，如下：
```
xpack.security.enabled: false
```

然后直接双击bin目录下的elasticsearch.bat即可启动。

3. elasticsearch可视化工具Kibana（可选）
为了更好的对数据进行管理，可以使用Kibana可视化工具进行管理和分析，下载链接为[Kibana](https://www.elastic.co/cn/downloads/kibana)，下载完后解压，直接双击运行 `bin\kibana.bat`即可。

#### 1.4.2 文档数据写入 ANN 索引库
```
# 以DuReader-Robust 数据集为例建立 ANN 索引库
python utils/offline_ann.py --index_name dureader_robust_query_encoder --doc_dir data/dureader_robust_processed
```
运行结束后，可使用Kibana查看数据

#### 1.4.3 启动 RestAPI 模型服务
```bash
# 指定语义检索系统的Yaml配置文件
$env:PIPELINE_YAML_PATH='rest_api/pipeline/semantic_search.yaml'
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

#### 1.4.4 启动 WebUI
```bash
# 配置模型服务地址
$env:API_ENDPOINT='http://127.0.0.1:8891'
# 在指定端口 8502 启动 WebUI
python -m streamlit run ui/webapp_semantic_search.py --server.port 8502
```

到这里您就可以打开浏览器访问 http://127.0.0.1:8502 地址体验语义检索系统服务了。
