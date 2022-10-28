# WINDOWS环境下搭建端到端开放文档抽取问答系统

## 1. 快速开始

### 1.1 运行环境

**安装PaddlePaddle：**

 环境中paddlepaddle-gpu或paddlepaddle版本应大于或等于2.3, 请参见[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)根据自己需求选择合适的PaddlePaddle下载命令。

**安装PaddleNLP：**

```bash
pip install paddlenlp==2.4.1
```

**安装Paddle-Pipelines：**

安装htbuilder：
```bash
git clone https://github.com/tvst/htbuilder.git
cd htbuilder/
python setup.py install
```

pip 一键安装Paddle-Pipelines：
```bash
pip install paddle-pipelines==0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或使用源码安装Paddle-Pipelines最新版本：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ${HOME}/PaddleNLP/pipelines/
python setup.py install
```

【注意】**以下的所有的流程都只需要在`pipelines`根目录下进行，不需要跳转目录**

### 1.2 一键体验问答系统
您可以通过如下命令快速体验开放文档抽取问答系统的效果。

```bash
# 我们建议在 GPU 环境下运行本示例，运行速度较快
# 设置 1 个空闲的 GPU 卡，此处假设 0 卡为空闲 GPU
python examples/document-intelligence/docprompt_example.py --device gpu
# 如果只有 CPU 机器，可以通过 --device 参数指定 cpu 即可, 运行耗时较长
python examples/document-intelligence/docprompt_example.py --device cpu
```

### 1.3 构建 Web 可视化开放文档抽取问答系统

整个 Web 可视化问答系统主要包含两大组件:  1. 基于 RestAPI 构建模型服务 2. 基于 Gradio 构建 WebUI。接下来我们依次搭建这 2 个服务并串联构成可视化的开放文档抽取问答系统。

#### 1.3.1 启动 RestAPI 模型服务
```bash
# 指定智能问答系统的Yaml配置文件
$env:PIPELINE_YAML_PATH='rest_api/pipeline/docprompt.yaml'
$env:QUERY_PIPELINE_NAME='query_documents'
# 使用端口号 8891 启动模型服务
python rest_api/application.py 8891
```

启动后可以使用curl命令验证是否成功运行：

```
curl --request POST --url 'http://0.0.0.0:8891/query_documents' -H "Content-Type: application/json"  --data '{"meta": {"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg", "prompt": ["发票号码是多少?", "校验码是多少?"]}}'
```

#### 1.3.2 启动 WebUI

```bash
python ui/webapp_docprompt_gradio.py  --serving_port 8891
```

到这里您就可以打开浏览器访问 http://127.0.0.1:7860 地址体验开放文档抽取问答系统系统服务了。
