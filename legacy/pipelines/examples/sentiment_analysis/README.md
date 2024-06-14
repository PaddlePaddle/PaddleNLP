# 端到端情感分析系统

## 1. 系统介绍

情感分析（sentiment analysis）是近年来国内外研究的热点，旨在对带有情感色彩的主观性文本进行分析、处理、归纳和推理，其广泛应用于消费决策、舆情分析、个性化推荐等领域，具有很高的商业价值。按照分析粒度可以大致分为三类：篇章级的情感分析（Document-Level Sentiment Classification）、语句级的情感分析（Sentence-Level Sentiment Classification）和属性级的情感分析（Aspect-Level Sentiment Classification）。

本项目更多聚焦于属性级的情感分析，支持文本评论中关于属性、观点词和情感倾向方面的分析。同时为方便用户使用，本项目提供了基于UIE模型提供了从输入数据到情感分析结果可视化的解决方案，用户只需上传自己的业务数据，就可以使用本项目开放的情感分析解决方案，快速搭建一个针对自己业务数据的情感分析系统，并提供基于[Gradio](https://gradio.app/) 的 Web 可视化服务。

<div align="center">
    <img src="https://user-images.githubusercontent.com/35913314/208049755-8fac879e-c544-443f-b127-5a83899a6d6f.png" />
</div>


## 2. 快速开始

以下是针对mac和linux的安装流程：


### 2.1 运行环境

**安装PaddlePaddle：**

 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。

**安装PaddleNLP：**

```bash
pip install paddlenlp==2.4.1
```

**安装Paddle-Pipelines：**

安装相关依赖：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

pip 一键安装Paddle-Pipelines：
```bash
pip install paddle-pipelines==0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或使用源码安装Paddle-Pipelines最新版本：
```bash
cd ${HOME}/PaddleNLP/pipelines/
python setup.py install
```

【注意】**以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录**

### 2.2 一键体验情感分析系统
您可以通过如下命令快速体验开放情感分析系统的效果。

```bash
# 建议在 GPU 环境下运行本示例，运行速度较快
export CUDA_VISIBLE_DEVICES=0
python examples/sentiment_analysis/senta_example.py \
    --file_path "your file path"
```

在运行结束后，可视化结果将存放到与`file_path` 文件所在目录的子目录`images`目录下。

### 2.3 构建 Web 可视化开放文档抽取问答系统

整个 Web 可视化情感分析系统主要包含两大组件:  1. 基于 RestAPI 构建模型服务 2. 基于 Gradio 构建 WebUI。接下来我们依次搭建这 2 个服务并串联构成可视化的情感分析系统。

#### 2.3.1 启动 RestAPI 模型服务
```bash
# 指定语义检索系统的Yaml配置文件
export CUDA_VISIBLE_DEVICES=0
export PIPELINE_YAML_PATH=rest_api/pipeline/senta.yaml
export QUERY_PIPELINE_NAME=senta_pipeline

# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/sentiment_analysis/run_senta_server.sh
```

#### 2.3.2 启动 WebUI

```bash
python ui/webapp_senta.py --serving_port 8891
```

Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/sentiment_analysis/run_senta_web.sh
```

接下来，您就可以打开浏览器访问 http://127.0.0.1:7860 地址体验情感分析系统服务了。
