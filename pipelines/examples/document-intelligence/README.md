# 端到端开放文档抽取问答系统

## 1. 系统介绍

开放文档抽取问答主要指对于网页、数字文档或扫描文档所包含的文本以及丰富的排版格式等信息，通过人工智能技术进行理解、分类、提取以及信息归纳的过程。开放文档抽取问答技术广泛应用于金融、保险、能源、物流、医疗等行业，常见的应用场景包括财务报销单、招聘简历、企业财报、合同文书、动产登记证、法律判决书、物流单据等多模态文档的关键信息抽取、问题回答等。

本项目提供了低成本搭建端到端开放文档抽取问答系统的能力。用户只需要处理好自己的业务数据，就可以使用本项目预置的开放文档抽取问答系统模型(文档OCR预处理模型、文档抽取问答模型)快速搭建一个针对自己业务数据的文档抽取问答系统，并提供基于[Gradio](https://gradio.app/) 的 Web 可视化服务。

![gradio](https://user-images.githubusercontent.com/63761690/197500524-17013358-8d19-43c4-9796-abac1e2d675f.gif)

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

### 2.2 一键体验问答系统
您可以通过如下命令快速体验开放文档抽取问答系统的效果。


```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
export CUDA_VISIBLE_DEVICES=0
python examples/document-intelligence/docprompt_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
unset CUDA_VISIBLE_DEVICES
python examples/document-intelligence/docprompt_example.py --device cpu
```

### 2.3 构建 Web 可视化开放文档抽取问答系统

整个 Web 可视化问答系统主要包含两大组件:  1. 基于 RestAPI 构建模型服务 2. 基于 Gradio 构建 WebUI。接下来我们依次搭建这 2 个服务并串联构成可视化的开放文档抽取问答系统。

#### 2.3.1 启动 RestAPI 模型服务
```bash
# 指定智能问答系统的Yaml配置文件
export PIPELINE_YAML_PATH=rest_api/pipeline/docprompt.yaml
export QUERY_PIPELINE_NAME=query_documents
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```
Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/document-intelligence/run_docprompt_server.sh
```
启动后可以使用curl命令验证是否成功运行：

```
curl --request POST --url 'http://0.0.0.0:8891/query_documents' -H "Content-Type: application/json"  --data '{"meta": {"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}}'
```

#### 2.3.2 启动 WebUI

```bash
python ui/webapp_docprompt_gradio.py  --serving_port 8891
```

Linux 用户推荐采用 Shell 脚本来启动服务：

```bash
sh examples/document-intelligence/run_docprompt_web.sh
```

到这里您就可以打开浏览器访问 http://127.0.0.1:7860 地址体验开放文档抽取问答系统系统服务了。
